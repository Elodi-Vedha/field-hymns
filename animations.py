"""Render Field Hymns state animations."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.gridspec import GridSpec
import numpy as np

from engine import (
    INTERVALS,
    MAX_WIND,
    N_CYCLES,
    N_K,
    N_OBJ,
    N_T,
    NOTES,
    PSO_ROUNDS,
    STEP_DUR,
    run_autonomous,
    save_wav,
    text_to_notes,
)


DEFAULT_TEXT = "I am autonomous"
DEFAULT_SEED = 1
DEFAULT_OUTPUT_DIR = Path("outputs/animations")
VIDEO_DPI = 120
PAGE_BG = "#0a0a0f"
AX_BG = "#0f0f18"
TEXT = "#e8e8e8"
MUTED = "#777777"
SPINE = "#2a2a3a"
GRID = "#333344"
ZERO_LINE = "#555566"


@dataclass
class AnimationRenderResult:
    """Paths and simulation metadata from an animation render."""

    text: str
    seed: int
    notes: List[str]
    cycle: int
    n: int
    C: int
    cycle_ns: List[int]
    cycle_cherns: List[int]
    bloch_path: Path
    audio_synced_path: Path
    swarm_path: Path
    audio_path: Path


def has_ffmpeg() -> bool:
    """Return whether the ffmpeg binary needed for MP4 rendering is present."""

    return shutil.which("ffmpeg") is not None


def require_ffmpeg() -> None:
    """Raise a helpful error if ffmpeg is not installed."""

    if not has_ffmpeg():
        raise RuntimeError(
            "ffmpeg is required to render MP4 animations. Install it with "
            "`brew install ffmpeg` on macOS or your platform package manager."
        )


def select_cycle(cycles: List[Dict], prefer_nontrivial: bool = True) -> int:
    """Return the index of the cycle to visualize."""

    if prefer_nontrivial:
        for idx, cycle in enumerate(cycles):
            if cycle["n"] != 0 and cycle["C"] != 0:
                return idx
        for idx, cycle in enumerate(cycles):
            if cycle["n"] != 0:
                return idx
    return 0


def run_recorded_autonomous(
    text: str,
    seed: int = DEFAULT_SEED,
    require_nontrivial: bool = False,
    max_attempts: int = 24,
    sr: int = 44100,
):
    """Run autonomous simulation with rich state needed for animations."""

    notes = text_to_notes(text)
    attempts = max(1, max_attempts if require_nontrivial else 1)
    tried = []
    for offset in range(attempts):
        trial_seed = seed + offset
        audio, cycles = run_autonomous(notes, sr=sr, seed=trial_seed,
                                       record_fields=True)
        summary = [(cycle["n"], cycle["C"]) for cycle in cycles]
        tried.append((trial_seed, summary))
        if not require_nontrivial:
            return notes, trial_seed, audio, cycles
        if any(cycle["n"] != 0 and cycle["C"] != 0 for cycle in cycles):
            return notes, trial_seed, audio, cycles

    details = "; ".join(
        f"seed {trial_seed}: {summary}" for trial_seed, summary in tried
    )
    raise RuntimeError(
        "No nontrivial autonomous cycle found. Tried " + details
    )


def _dhat(cycle: Dict) -> np.ndarray:
    d_vec = np.stack(
        [cycle["vx_grid"], cycle["vy_grid"], cycle["eps_grid"]],
        axis=-1,
    )
    norm = np.linalg.norm(d_vec, axis=-1, keepdims=True)
    return d_vec / np.maximum(norm, 1e-12)


def _interval_from_occ(occ: float) -> float:
    dist = min(abs(float(occ) - 0.5) * 2.0, 1.0)
    idx = min(int(dist * len(INTERVALS)), len(INTERVALS) - 1)
    return INTERVALS[idx]


def _style_figure(fig) -> None:
    fig.patch.set_facecolor(PAGE_BG)


def _style_2d_axis(ax) -> None:
    ax.set_facecolor(AX_BG)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(color=GRID, linewidth=0.35, alpha=0.35)
    for spine in ax.spines.values():
        spine.set_color(SPINE)


def _style_3d_axis(ax) -> None:
    ax.set_facecolor(AX_BG)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.zaxis.label.set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=6, pad=0)
    pane = mcolors.to_rgba(AX_BG, 0.55)
    grid = mcolors.to_rgba(GRID, 0.35)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(pane)
        axis.pane.set_edgecolor(SPINE)
        axis._axinfo["grid"]["color"] = grid


def _draw_sphere(ax) -> None:
    u = np.linspace(0, 2 * np.pi, 28)
    v = np.linspace(0, np.pi, 14)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color="#7a7a8a", linewidth=0.35, alpha=0.33)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("dx/|d|", fontsize=7)
    ax.set_ylabel("dy/|d|", fontsize=7)
    ax.set_zlabel("dz/|d|", fontsize=7)
    _style_3d_axis(ax)


def _draw_loop(ax, d_hat: np.ndarray, t_idx: int, title: str,
               color: str = "#44c2ff") -> None:
    _draw_sphere(ax)
    for past in range(0, t_idx + 1, 4):
        loop = d_hat[past]
        ax.plot(loop[:, 0], loop[:, 1], loop[:, 2],
                color=color, alpha=0.08, linewidth=0.7)
    loop = d_hat[t_idx]
    closed = np.vstack([loop, loop[0]])
    ax.plot(closed[:, 0], closed[:, 1], closed[:, 2],
            color=color, linewidth=2.0)
    ax.scatter(loop[:, 0], loop[:, 1], loop[:, 2],
               c=np.linspace(0, 1, len(loop)), cmap="twilight",
               s=12, alpha=0.9)
    ax.set_title(title, fontsize=10, color=TEXT)
    ax.view_init(elev=24, azim=35 + 360 * t_idx / N_T)


def render_bloch_sphere_wrapping(
    cycles: List[Dict],
    cycle_index: int,
    output_path: Path,
    text: str,
    seed: int,
    fps: float = 1.0 / STEP_DUR,
) -> Path:
    """Render the selected cycle's k-loop wrapping around the Bloch sphere."""

    require_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cycle = cycles[cycle_index]
    d_hat = _dhat(cycle)

    fig = plt.figure(figsize=(12, 6.4), facecolor=PAGE_BG)
    _style_figure(fig)
    writer = FFMpegWriter(
        fps=fps,
        bitrate=2400,
        metadata={"title": "Field Hymns Bloch wrapping"},
    )
    with writer.saving(fig, str(output_path), dpi=VIDEO_DPI):
        for t_idx in range(N_T):
            fig.clear()
            _style_figure(fig)
            ax = fig.add_subplot(111, projection="3d")
            _draw_loop(
                ax,
                d_hat,
                t_idx,
                "Autonomous Bloch-sphere wrapping\n"
                f'text="{text}" · seed={seed} · cycle={cycle["cycle"]} · '
                f'n={cycle["n"]:+d} · C={cycle["C"]:+d}',
                color="#7ecfb0",
            )
            fig.text(
                0.5,
                0.03,
                "Each frame is one t-step; colored points are the periodic "
                "k-loop on d(k,t)/|d|.",
                ha="center",
                fontsize=9,
                color=MUTED,
            )
            writer.grab_frame()
    plt.close(fig)
    return output_path


def render_audio_synced_state(
    audio: np.ndarray,
    cycles: List[Dict],
    notes: List[str],
    output_path: Path,
    text: str,
    seed: int,
    sr: int = 44100,
    fps: float = 1.0 / STEP_DUR,
) -> Path:
    """Render all autonomous cycles with the generated audio muxed in."""

    require_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    d_hats = [_dhat(cycle) for cycle in cycles]
    audio_path = output_path.with_suffix(".wav")
    silent_path = output_path.with_name(output_path.stem + "_silent.mp4")
    save_wav(str(audio_path), audio, sr=sr)

    def draw_frame(fig, global_t: int) -> None:
        fig.clear()
        _style_figure(fig)
        gs = GridSpec(
            3,
            2,
            figure=fig,
            width_ratios=[1.1, 1.0],
            height_ratios=[1, 1, 1],
            left=0.045,
            right=0.975,
            bottom=0.08,
            top=0.86,
            wspace=0.24,
            hspace=0.42,
        )
        cycle_idx = global_t // N_T
        t_idx = global_t % N_T
        cycle = cycles[cycle_idx]
        occ_trace = np.array(cycle["occ_trace"])
        eps_trace = np.array(cycle["eps_trace"])
        note = notes[global_t % len(notes)]
        interval = _interval_from_occ(occ_trace[t_idx])

        ax_sphere = fig.add_subplot(gs[:, 0], projection="3d")
        _draw_loop(
            ax_sphere,
            d_hats[cycle_idx],
            t_idx,
            f'Bloch state · cycle {cycle_idx + 1}\n'
            f'n={cycle["n"]:+d}, C={cycle["C"]:+d}',
            color="#7ecfb0",
        )

        x = np.arange(cycle_idx * N_T, cycle_idx * N_T + N_T)
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.plot(x, occ_trace, color="#2ca25f", linewidth=1.6)
        ax1.axhline(0.5, color=ZERO_LINE, linestyle=":", linewidth=0.9)
        ax1.axvline(global_t, color=TEXT, linewidth=1.3)
        ax1.set_xlim(cycle_idx * N_T, cycle_idx * N_T + N_T - 1)
        ax1.set_ylim(-0.03, 1.03)
        ax1.set_title("occupation selects chord interval", fontsize=9)
        ax1.set_ylabel("occupation", fontsize=8)
        _style_2d_axis(ax1)

        ax2 = fig.add_subplot(gs[1, 1])
        ax2.plot(x, eps_trace, color="#5e81d1",
                 linewidth=1.4, label="epsilon")
        ax2.plot(x, cycle["vx_trace"], color="#d95f02",
                 linewidth=1.4, label="mean Vx")
        ax2.axvline(global_t, color=TEXT, linewidth=1.3)
        ax2.set_xlim(cycle_idx * N_T, cycle_idx * N_T + N_T - 1)
        ax2.set_ylim(-1.28, 1.28)
        ax2.set_title("drive and feedback", fontsize=9)
        _style_2d_axis(ax2)
        ax2.legend(fontsize=7, loc="upper right", facecolor="#111118",
                   edgecolor=SPINE, labelcolor=TEXT)

        ax3 = fig.add_subplot(gs[2, 1])
        sample_t = int(global_t * STEP_DUR * sr)
        half = int(0.10 * sr)
        start = max(0, sample_t - half)
        stop = min(len(audio), sample_t + half)
        wx = np.linspace(start / sr, stop / sr, stop - start)
        ax3.plot(wx, audio[start:stop], color="#6a51a3", linewidth=0.8)
        ax3.axvline(sample_t / sr, color=TEXT, linewidth=1.2)
        ax3.set_xlim(
            max(0, sample_t / sr - 0.10),
            min(len(audio) / sr, sample_t / sr + 0.10),
        )
        ax3.set_ylim(-0.24, 0.24)
        ax3.set_title("generated audio waveform", fontsize=9)
        ax3.set_xlabel("seconds", fontsize=8)
        _style_2d_axis(ax3)

        fig.suptitle(
            f'Audio-synced autonomous state · "{text}" · seed={seed}',
            fontsize=12,
            y=0.965,
            color=TEXT,
        )
        meta = (
            f't={global_t:03d}/{N_T * N_CYCLES - 1:03d}   '
            f'cycle={cycle_idx + 1}   n={cycle["n"]:+2d}   '
            f'C={cycle["C"]:+2d}   note={note:<3s}   '
            f'interval={interval:5.3f}   occupation={occ_trace[t_idx]:5.3f}'
        )
        fig.text(
            0.5,
            0.905,
            meta,
            ha="center",
            va="center",
            fontsize=10.5,
            family="monospace",
            weight="bold",
            color=TEXT,
        )

    fig = plt.figure(figsize=(12, 7), facecolor=PAGE_BG)
    _style_figure(fig)
    writer = FFMpegWriter(
        fps=fps,
        bitrate=3200,
        metadata={"title": "Field Hymns audio synced state"},
    )
    with writer.saving(fig, str(silent_path), dpi=VIDEO_DPI):
        for global_t in range(N_T * N_CYCLES):
            draw_frame(fig, global_t)
            writer.grab_frame()
    plt.close(fig)

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(silent_path),
                "-i",
                str(audio_path),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"ffmpeg failed to mux animation audio: {detail}") from exc
    finally:
        silent_path.unlink(missing_ok=True)
    return output_path


def render_swarm_negotiation_prelude(
    cycles: List[Dict],
    cycle_index: int,
    output_path: Path,
    text: str,
    fps: float = 6.0,
) -> Path:
    """Render the PSO vote negotiation that precedes one autonomous cycle."""

    require_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history = cycles[cycle_index]["pso_history"]
    if not history:
        raise ValueError("pso_history is empty; run with record_fields=True.")

    fig = plt.figure(figsize=(11, 6), facecolor=PAGE_BG)
    _style_figure(fig)
    writer = FFMpegWriter(
        fps=fps,
        bitrate=2600,
        metadata={"title": "Field Hymns swarm negotiation"},
    )
    note_colors = {note: plt.cm.Set3(i / len(NOTES))
                   for i, note in enumerate(NOTES)}
    theta = 2 * np.pi * np.arange(N_OBJ) / N_OBJ
    xs = np.cos(theta)
    ys = np.sin(theta)

    with writer.saving(fig, str(output_path), dpi=130):
        for snapshot in history:
            fig.clear()
            _style_figure(fig)
            gs = GridSpec(
                2,
                2,
                figure=fig,
                width_ratios=[1.05, 1.0],
                height_ratios=[1, 1],
                left=0.055,
                right=0.965,
                bottom=0.10,
                top=0.84,
                wspace=0.28,
                hspace=0.42,
            )
            ax_ring = fig.add_subplot(gs[:, 0])
            ax_bar = fig.add_subplot(gs[0, 1])
            ax_mean = fig.add_subplot(gs[1, 1])
            positions = np.array(snapshot["positions"])
            fitness = np.array(snapshot["fitness"])
            ax_ring.set_facecolor(AX_BG)

            for idx in range(N_OBJ):
                nxt = (idx + 1) % N_OBJ
                ax_ring.plot([xs[idx], xs[nxt]], [ys[idx], ys[nxt]],
                             color=SPINE, linewidth=1)
            sizes = 420 + 350 * np.clip(fitness, 0, 1)
            colors = [note_colors[note] for note in snapshot["notes"]]
            ax_ring.scatter(xs, ys, s=sizes, c=colors,
                            edgecolor="0.2", linewidth=1.0, zorder=3)
            for idx, (x0, y0) in enumerate(zip(xs, ys)):
                ax_ring.text(
                    x0,
                    y0,
                    f'{idx}\n{snapshot["notes"][idx]}',
                    ha="center",
                    va="center",
                    fontsize=8,
                    weight="bold",
                )
                scale = 0.16 * positions[idx]
                ax_ring.arrow(
                    x0,
                    y0,
                    scale * np.cos(theta[idx]),
                    scale * np.sin(theta[idx]),
                    head_width=0.035,
                    color=TEXT,
                    alpha=0.75,
                    length_includes_head=True,
                )
            ax_ring.set_xlim(-1.42, 1.42)
            ax_ring.set_ylim(-1.42, 1.42)
            ax_ring.set_aspect("equal", adjustable="box")
            ax_ring.set_anchor("C")
            ax_ring.set_autoscale_on(False)
            ax_ring.axis("off")
            ax_ring.set_title(
                "local message-passing ring\n"
                "node size = personal-best fitness",
                fontsize=10,
                color=TEXT,
            )

            bar_colors = ["#7ecfb0" if vote >= 0 else "#e8a070"
                          for vote in positions]
            ax_bar.bar(np.arange(N_OBJ), positions, color=bar_colors)
            ax_bar.axhline(0, color=ZERO_LINE, linewidth=0.8)
            consensus = round(snapshot["mean_vote"])
            ax_bar.axhline(consensus, color=TEXT, linestyle="--",
                           linewidth=1.0, label="rounded consensus")
            ax_bar.set_xlim(-0.6, N_OBJ - 0.4)
            ax_bar.set_ylim(-(MAX_WIND + 0.8), MAX_WIND + 0.8)
            ax_bar.set_title("winding votes by object", fontsize=10)
            ax_bar.set_xlabel("object", fontsize=8)
            ax_bar.set_ylabel("vote", fontsize=8)
            _style_2d_axis(ax_bar)
            ax_bar.legend(fontsize=7, loc="upper right", facecolor="#111118",
                          edgecolor=SPINE, labelcolor=TEXT)

            means = [
                item["mean_vote"]
                for item in history[:snapshot["step"] + 1]
            ]
            ax_mean.plot(range(len(means)), means,
                         color="#5e81d1", linewidth=1.7)
            ax_mean.axhline(0, color=ZERO_LINE, linestyle=":",
                            linewidth=0.8)
            ax_mean.axhline(consensus, color=TEXT, linestyle="--",
                            linewidth=1.0)
            ax_mean.set_xlim(0, PSO_ROUNDS - 1)
            ax_mean.set_ylim(-(MAX_WIND + 0.8), MAX_WIND + 0.8)
            ax_mean.set_title(f"mean vote -> n = {consensus:+d}",
                              fontsize=10)
            ax_mean.set_xlabel("PSO round", fontsize=8)
            ax_mean.set_ylabel("mean vote", fontsize=8)
            _style_2d_axis(ax_mean)

            fig.suptitle(
                f'Swarm negotiation prelude · "{text}" · '
                f'cycle {cycle_index + 1}',
                fontsize=12,
                y=0.96,
                color=TEXT,
            )
            fig.text(
                0.5,
                0.895,
                f'round={snapshot["step"] + 1:02d}/{PSO_ROUNDS:02d}   '
                f'mean_vote={snapshot["mean_vote"]:+6.3f}   '
                f"consensus_n={consensus:+2d}",
                ha="center",
                family="monospace",
                fontsize=10.5,
                weight="bold",
                color=TEXT,
            )
            fig.text(
                0.5,
                0.035,
                "Each object only receives neighbor emissions; consensus n "
                "later controls Vy(k,t) and the topology/audio.",
                ha="center",
                fontsize=9,
                color=MUTED,
            )
            writer.grab_frame()
    plt.close(fig)
    return output_path


def render_all(
    text: str = DEFAULT_TEXT,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    seed: int = DEFAULT_SEED,
    require_nontrivial: bool = False,
    max_attempts: int = 24,
    sr: int = 44100,
) -> AnimationRenderResult:
    """Render all supported Field Hymns animations."""

    require_ffmpeg()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    notes, used_seed, audio, cycles = run_recorded_autonomous(
        text=text,
        seed=seed,
        require_nontrivial=require_nontrivial,
        max_attempts=max_attempts,
        sr=sr,
    )
    cycle_index = select_cycle(cycles)
    cycle = cycles[cycle_index]
    safe_text = "".join(ch if ch.isalnum() else "_" for ch in text.lower())
    safe_text = "_".join(part for part in safe_text.split("_") if part) or "field_hymns"
    prefix = f"{safe_text[:32]}_seed_{used_seed}"

    bloch_path = output_dir / f"{prefix}_bloch_wrapping.mp4"
    audio_path = output_dir / f"{prefix}_audio_synced.mp4"
    swarm_path = output_dir / f"{prefix}_swarm_prelude.mp4"

    render_bloch_sphere_wrapping(
        cycles, cycle_index, bloch_path, text, used_seed
    )
    render_audio_synced_state(
        audio, cycles, notes, audio_path, text, used_seed, sr=sr
    )
    render_swarm_negotiation_prelude(
        cycles, cycle_index, swarm_path, text
    )

    return AnimationRenderResult(
        text=text,
        seed=used_seed,
        notes=notes,
        cycle=cycle["cycle"],
        n=cycle["n"],
        C=cycle["C"],
        cycle_ns=[item["n"] for item in cycles],
        cycle_cherns=[item["C"] for item in cycles],
        bloch_path=bloch_path,
        audio_synced_path=audio_path,
        swarm_path=swarm_path,
        audio_path=audio_path.with_suffix(".wav"),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Field Hymns Bloch, audio-synced, and swarm animations."
    )
    parser.add_argument("--text", default=DEFAULT_TEXT,
                        help="Text to map into Solresol DNA.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Seed for deterministic swarm negotiation.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory for rendered MP4/WAV files.")
    parser.add_argument(
        "--require-nontrivial",
        action="store_true",
        help="Try seeds until at least one autonomous cycle has n != 0 and C != 0.",
    )
    parser.add_argument("--max-attempts", type=int, default=24,
                        help="Maximum seed attempts with --require-nontrivial.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = render_all(
        text=args.text,
        output_dir=args.output_dir,
        seed=args.seed,
        require_nontrivial=args.require_nontrivial,
        max_attempts=args.max_attempts,
    )
    print(f'Text: "{result.text}"')
    print(f"Seed: {result.seed}")
    print(f"Selected cycle: {result.cycle}  n={result.n:+d}  C={result.C:+d}")
    print(f"All cycles n={result.cycle_ns}  C={result.cycle_cherns}")
    print("Rendered:")
    print(f"  {result.bloch_path}")
    print(f"  {result.audio_synced_path}")
    print(f"  {result.swarm_path}")
    print(f"  {result.audio_path}")


if __name__ == "__main__":
    main()
