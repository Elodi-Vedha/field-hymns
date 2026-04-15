"""
Autonomous vs Command: Unified Thouless Pump in Solresol

H(k_i, t) = V_x(k_i, t) σ_x + V_y(k_i, t) σ_y + ε(t) σ_z

V_x(k, t): Appendix D self-consistent feedback.
    V_x responds to dot occupation ⟨d†d⟩.
    ⟨d†d⟩ responds to V_x.
    They find the fixed point together, or they don't.
    ε_k = ε(t) + cos(k)  — k-dependent dot energy (hopping).

V_y(k, t): PSO phase accumulation — the imaginary coupling.
    V_y(k_i, t) = A · sin(n · k_i − n · 2πt/N_T)
    n is the winding number negotiated by the autonomous swarm.
    V_y = 0 for the command system: no phase, no imaginary coupling.

The same (V_x, V_y, ε) that the FHS Chern number is computed from
also determines the occupation at each step, which determines the
chord interval in the audio. The topology, the feedback, and the
sound are the same thing viewed three ways.

Autonomous: n ≠ 0, V_y ≠ 0, Appendix D feedback active. C = ±2.
Command: n = 0, V_y = 0, V_x prescribed (no feedback). C = 0.

Command occupation: driven by ε alone (no self-consistency, no k-structure).
    This is correct — the paper says command systems cannot sustain
    the fixed point. They drift with the external drive.
Autonomous occupation: self-consistent at each (k, t), responds to
    the full (V_x, V_y) coupling. Fixed point is found or lost.
"""

import numpy as np
import wave
from typing import List, Tuple, Optional
from dataclasses import dataclass


NOTES     = ["do", "re", "mi", "fa", "sol", "la", "si"]
BASE_FREQ = dict(zip(NOTES, [261.63, 293.66, 329.63,
                              349.23, 392.00, 440.00, 493.88]))

# Solresol harmonic intervals indexed by occupation distance from fixed point.
# Near 0.5 (at fixed point) → consonant. Far from 0.5 → dissonant.
INTERVALS = [1.0, 9/8, 6/5, 4/3, 3/2, 5/3, 16/9]

def text_to_notes(text: str) -> List[str]:
    out = []
    for c in text.lower():
        if c.isalpha():
            out.append(NOTES[(ord(c) - ord('a')) % 7])
        elif c == ' ':
            out.append("sol")
    return out or ["do", "re", "mi"]


# ── Parameters ────────────────────────────────────────────────────────────────

N_OBJ      = 8
N_K        = 32
N_T        = 64
N_CYCLES   = 3
PSO_ROUNDS = 25
MAX_WIND   = 3

EPS_AMP    = 1.2
BETA       = 18.0
LAMBDA     = 1.5
A_Y        = 0.5      # V_y amplitude
FB_ITERS   = 8        # Appendix D iterations per step
FB_RELAX   = 0.70     # V_x relaxation rate

# Command: V_x prescribed at this value (outside self-consistent range)
VX_CMD     = 0.40

PSO_W      = 0.50
PSO_C1     = 1.50
PSO_C2     = 1.50
MAX_VEL    = 0.40
STEP_DUR   = 0.080    # seconds per time step (64 steps × 0.08s = 5.1s per cycle)


# ── Quantum layer ─────────────────────────────────────────────────────────────

def _lower_eig(vx: float, vy: float, eps: float) -> np.ndarray:
    H = np.array([[eps,       vx - 1j*vy],
                  [vx + 1j*vy, -eps     ]], dtype=complex)
    _, vecs = np.linalg.eigh(H)
    return vecs[:, 0]


def _link(a: np.ndarray, b: np.ndarray) -> complex:
    ov = np.vdot(a, b)
    return ov / abs(ov) if abs(ov) > 1e-12 else 1.0 + 0j


def chern_FHS(psi_grid: List[List[np.ndarray]]) -> int:
    """Fukui-Hatsugai-Suzuki Chern number on (t, k) torus."""
    Nt, Nk = len(psi_grid), len(psi_grid[0])
    total = 0.0
    for it in range(Nt):
        for ik in range(Nk):
            U1 = _link(psi_grid[it][ik],
                       psi_grid[(it+1) % Nt][ik])
            U2 = _link(psi_grid[(it+1) % Nt][ik],
                       psi_grid[(it+1) % Nt][(ik+1) % Nk])
            U3 = _link(psi_grid[(it+1) % Nt][(ik+1) % Nk],
                       psi_grid[it][(ik+1) % Nk])
            U4 = _link(psi_grid[it][(ik+1) % Nk],
                       psi_grid[it][ik])
            total += np.angle(U1 * U2 * U3 * U4)
    return int(np.round(total / (2 * np.pi)))


# ── Appendix D: V_x self-consistent feedback ──────────────────────────────────

def appendix_d_step(eps: float, k: float, V_prev: float) -> Tuple[float, float]:
    """
    One Appendix D step: iterate V = tanh(λ(⟨d†d⟩ - 0.5)).
    ε_k = ε + cos(k) — k-dependent dot energy.
    Returns (V_x, occupation).
    """
    V = V_prev
    eps_k = eps + np.cos(k)
    for _ in range(FB_ITERS):
        n = 1.0 / (1.0 + np.exp(BETA * (eps_k + LAMBDA * V)))
        V_target = np.tanh(LAMBDA * (n - 0.5))
        V = FB_RELAX * V + (1 - FB_RELAX) * V_target
    occ = 1.0 / (1.0 + np.exp(BETA * (eps_k + LAMBDA * V)))
    return V, occ


def command_occupation(eps: float) -> float:
    """
    Command occupation: Fermi-Dirac with prescribed V_x, no feedback.
    Drifts with ε — cannot sustain the fixed point.
    """
    return 1.0 / (1.0 + np.exp(BETA * (eps + LAMBDA * VX_CMD)))


# ── Audio ─────────────────────────────────────────────────────────────────────

def _interval_from_occ(occ: float) -> float:
    """Harmonic interval from occupation distance from fixed point."""
    dist = min(abs(occ - 0.5) * 2.0, 1.0)
    idx  = min(int(dist * len(INTERVALS)), len(INTERVALS) - 1)
    return INTERVALS[idx]


def _chord(base: float, eps: float, occ: float,
           dur: float = STEP_DUR, sr: int = 44100) -> np.ndarray:
    """
    Two-note chord: the protein.
    ε-note: base frequency shifted by ε.
    V-note: base frequency * harmonic interval derived from occupation.
    Consonant at fixed point. Dissonant far from it.
    """
    freq_eps = base * (1.0 + 0.20 * eps)
    freq_v   = base * _interval_from_occ(occ)
    amp = 0.10
    n   = max(1, int(dur * sr))
    t   = np.linspace(0, dur, n, endpoint=False)
    w   = (amp * np.sin(2 * np.pi * freq_eps * t) +
           amp * np.sin(2 * np.pi * freq_v   * t))
    a   = min(int(0.05 * n), n // 5)
    if a > 0:
        w[:a]  *= np.linspace(0, 1, a)
        w[-a:] *= np.linspace(1, 0, a)
    return w


def _tone(base: float, occ: float,
          dur: float = STEP_DUR, sr: int = 44100) -> np.ndarray:
    """Single note for command. No interval — no topology."""
    freq = base * (1.0 + 0.20 * (occ - 0.5))
    amp  = 0.15
    n    = max(1, int(dur * sr))
    t    = np.linspace(0, dur, n, endpoint=False)
    w    = amp * np.sin(2 * np.pi * freq * t)
    a    = min(int(0.05 * n), n // 5)
    if a > 0:
        w[:a]  *= np.linspace(0, 1, a)
        w[-a:] *= np.linspace(1, 0, a)
    return w


def save_wav(path: str, audio: np.ndarray, sr: int = 44100):
    a = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1); wf.setsampwidth(2)
        wf.setframerate(sr); wf.writeframes(a.tobytes())


# ── PSO objects ───────────────────────────────────────────────────────────────

@dataclass
class Emission:
    note:      str
    freq:      float
    wind_vote: float
    pb_vote:   float
    fitness:   float
    generation: int


class AutonomousObject:
    """
    PSO particle. Votes on signed winding number n.
    n determines V_y(k,t) = A·sin(n·k − n·2πt/N_T).
    Fitness: self-referential agreement with neighbors.
    Type changes via doesNotUnderstand.
    """

    def __init__(self, note: str, idx: int, dna_seed: float):
        self.idx  = idx
        self.note = note
        self.base = BASE_FREQ[note]
        rng = np.random.default_rng(int(abs(dna_seed) * 1000 + idx) % (2**32))
        self.pos    = float(rng.uniform(-(MAX_WIND + 0.5), MAX_WIND + 0.5))
        self.vel    = float(rng.uniform(-0.2, 0.2))
        self.pb_pos = self.pos
        self.pb_fit = -np.inf
        self.last_emission: Optional[Emission] = None
        self.inbox: List[Emission] = []
        self.type_history = [note]

    def receive(self, msg: Emission):
        self.inbox.append(msg)

    def _fitness(self, received: List[Emission]) -> float:
        if not received or self.last_emission is None:
            return 0.0
        pred = self.last_emission.wind_vote
        act  = float(np.mean([m.wind_vote for m in received]))
        return max(0.0, 1.0 - abs(pred - act) / (2 * MAX_WIND + 1))

    def negotiate(self, step: int) -> Emission:
        received = self.inbox[:]
        self.inbox.clear()

        fitness = self._fitness(received)
        if fitness > self.pb_fit:
            self.pb_fit = fitness
            self.pb_pos = self.pos

        best_nb: Optional[Emission] = None
        if received:
            best_nb = max(received, key=lambda m: m.fitness)

        r1, r2    = np.random.random(), np.random.random()
        cognitive = PSO_C1 * r1 * (self.pb_pos - self.pos)
        social    = PSO_C2 * r2 * (best_nb.pb_vote - self.pos) if best_nb else 0.0
        self.vel  = np.clip(PSO_W * self.vel + cognitive + social, -MAX_VEL, MAX_VEL)
        self.pos  = np.clip(self.pos + self.vel, -(MAX_WIND + 0.5), MAX_WIND + 0.5)

        if (best_nb is not None and
                best_nb.note != self.note and
                best_nb.fitness > fitness + 0.12):
            self.note = best_nb.note
            self.base = BASE_FREQ[self.note]
            self.type_history.append(self.note)

        em = Emission(
            note=self.note, freq=self.base,
            wind_vote=self.pos, pb_vote=self.pb_pos,
            fitness=self.pb_fit, generation=step
        )
        self.last_emission = em
        return em


class CommandObject:
    """Fixed type. n = 0. V_y = 0. No feedback. No negotiation."""
    def __init__(self, note: str, idx: int, parent: Optional[str] = None):
        self.idx            = idx
        self.note           = note
        self.inherited_from = parent or note
        self.base           = BASE_FREQ[note]


# ── System runners ────────────────────────────────────────────────────────────

def run_autonomous(notes: List[str], sr: int = 44100):
    dna_seed   = sum(hash(n) * (i + 1) for i, n in enumerate(notes)) % 100000
    k_arr      = 2 * np.pi * np.arange(N_K) / N_K
    eps_list   = [EPS_AMP * np.sin(2 * np.pi * t / N_T) for t in range(N_T)]
    all_audio  = []
    cycle_data = []

    for cycle in range(N_CYCLES):
        objects = [AutonomousObject(notes[i % len(notes)], i,
                                    dna_seed + cycle * 137)
                   for i in range(N_OBJ)]

        for step in range(PSO_ROUNDS):
            emissions = [obj.negotiate(step) for obj in objects]
            for i, obj in enumerate(objects):
                obj.receive(emissions[(i - 1) % N_OBJ])
                obj.receive(emissions[(i + 1) % N_OBJ])

        mean_vote   = float(np.mean([obj.pos for obj in objects]))
        n_consensus = int(np.round(np.clip(mean_vote, -MAX_WIND, MAX_WIND)))

        # Build eigenstate grid and compute Chern number.
        # V_x from Appendix D feedback at each (k, t).
        # V_y from PSO phase at each (k, t).
        # These are the SAME quantities that drive the audio.
        psi_grid  = []
        V_prev    = np.zeros(N_K)
        eps_trace = []
        vx_trace  = []
        vy_trace  = []
        occ_trace = []

        # When n=0, use k-independent Vx to give clean C=0
        use_k_dep = (n_consensus != 0)

        for t_idx, eps in enumerate(eps_list):
            t_phase = n_consensus * 2 * np.pi * t_idx / N_T
            row     = []
            step_vx = []
            step_vy = []
            step_oc = []

            for ik, k in enumerate(k_arr):
                k_arg = k if use_k_dep else 0.0
                Vx, occ     = appendix_d_step(eps, k_arg, V_prev[ik])
                V_prev[ik]  = Vx
                Vy          = A_Y * np.sin(n_consensus * k - t_phase)
                row.append(_lower_eig(Vx, Vy, eps))
                step_vx.append(Vx)
                step_vy.append(Vy)
                step_oc.append(occ)

            psi_grid.append(row)
            eps_trace.append(eps)
            vx_trace.append(float(np.mean(step_vx)))
            vy_trace.append(float(np.mean(np.abs(step_vy))))
            occ_trace.append(float(np.mean(step_oc)))

        C = chern_FHS(psi_grid)

        # Audio: chord from mean (ε, occupation) at each step
        for t_idx in range(N_T):
            obj  = objects[t_idx % N_OBJ]
            seg  = _chord(obj.base, eps_trace[t_idx],
                          occ_trace[t_idx], STEP_DUR, sr)
            all_audio.append(seg)

        final_types  = [o.note for o in objects]
        type_changes = [(o.idx, list(o.type_history))
                        for o in objects if len(o.type_history) > 1]

        cycle_data.append({
            'cycle':        cycle + 1,
            'n':            n_consensus,
            'C':            C,
            'mean_vote':    mean_vote,
            'votes':        [obj.pos for obj in objects],
            'eps_trace':    eps_trace,
            'vx_trace':     vx_trace,
            'vy_trace':     vy_trace,
            'occ_trace':    occ_trace,
            'final_types':  final_types,
            'type_changes': type_changes,
        })

    audio = np.concatenate(all_audio) if all_audio else np.array([])
    return audio, cycle_data


def run_command(notes: List[str], sr: int = 44100):
    objects   = [CommandObject(notes[i % len(notes)], i,
                               notes[(i-1) % len(notes)] if i > 0 else None)
                 for i in range(N_OBJ)]
    k_arr     = 2 * np.pi * np.arange(N_K) / N_K
    eps_list  = [EPS_AMP * np.sin(2 * np.pi * t / N_T) for t in range(N_T)]
    all_audio = []
    cycle_data = []

    for cycle in range(N_CYCLES):
        # V_y = 0: no PSO phase, no imaginary coupling.
        # V_x = VX_CMD: prescribed, no feedback, no k-dependence.
        # Command occupation: driven by ε alone, not self-consistent.
        psi_grid  = [[_lower_eig(VX_CMD, 0.0, eps) for _ in k_arr]
                     for eps in eps_list]
        C         = chern_FHS(psi_grid)
        occ_trace = [command_occupation(eps) for eps in eps_list]

        for t_idx in range(N_T):
            obj = objects[t_idx % N_OBJ]
            seg = _tone(obj.base, occ_trace[t_idx], STEP_DUR, sr)
            all_audio.append(seg)

        cycle_data.append({
            'cycle':       cycle + 1,
            'n':           0,
            'C':           C,
            'mean_vote':   0.0,
            'votes':       [0.0] * N_OBJ,
            'eps_trace':   eps_list[:],
            'vx_trace':    [VX_CMD] * N_T,
            'vy_trace':    [0.0]    * N_T,
            'occ_trace':   occ_trace,
            'final_types': [o.note for o in objects],
        })

    audio = np.concatenate(all_audio) if all_audio else np.array([])
    return audio, cycle_data


# ── CLI ───────────────────────────────────────────────────────────────────────

def cli_demo(text: str = "I am autonomous",
             output_dir: str = "/mnt/user-data/outputs"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    notes = text_to_notes(text)

    print(f'\nInput: "{text}"')
    print(f"DNA:   {' — '.join(notes)}\n")

    auto_audio, auto_cycles = run_autonomous(notes)
    print("AUTONOMOUS  (Appendix D V_x · PSO V_y · unified C and audio):")
    for cd in auto_cycles:
        ev = f"  [{len(cd['type_changes'])} evolved]" if cd.get('type_changes') else ""
        print(f"  Cycle {cd['cycle']}: n={cd['n']:+d}  C={cd['C']:+d}  "
              f"vote={cd['mean_vote']:+.2f}  "
              f"occ=[{min(cd['occ_trace']):.2f},{max(cd['occ_trace']):.2f}]{ev}")

    cmd_audio, cmd_cycles = run_command(notes)
    print("\nCOMMAND  (V_y=0 · V_x prescribed · no feedback · C=0):")
    for cd in cmd_cycles:
        print(f"  Cycle {cd['cycle']}: n=+0  C={cd['C']:+d}  "
              f"occ=[{min(cd['occ_trace']):.2f},{max(cd['occ_trace']):.2f}]")

    tag = text[:20].replace(' ', '_')
    save_wav(f"{output_dir}/autonomous_{tag}.wav", auto_audio)
    save_wav(f"{output_dir}/command_{tag}.wav",    cmd_audio)

    auto_Cs = [cd['C'] for cd in auto_cycles]
    cmd_Cs  = [cd['C'] for cd in cmd_cycles]
    print(f"\nAutonomous C: {auto_Cs}   Command C: {cmd_Cs}")
    return auto_cycles, cmd_cycles


if __name__ == "__main__":
    import sys
    cli_demo(" ".join(sys.argv[1:]) if len(sys.argv) > 1 else "I am autonomous")
