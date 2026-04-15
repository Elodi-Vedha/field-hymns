import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io, wave as wv, sys, os

sys.path.insert(0, os.path.dirname(__file__))
from engine import (
    text_to_notes, run_autonomous, run_command,
    N_OBJ, N_T, N_CYCLES, MAX_WIND, NOTES, BASE_FREQ,
    INTERVALS, EPS_AMP, VX_CMD, STEP_DUR
)

st.set_page_config(
    page_title="Field Hymns · Autonomous Inheritance",
    page_icon="◈",
    layout="wide"
)

st.markdown("""
<style>
.block-container { padding-top: 1.6rem; }
.cbox { border-radius: 8px; padding: 1.2rem .8rem; text-align: center; }
.cv   { font-size: 2.6rem; font-weight: 800; font-family: monospace; }
.pos  { color: #7ecfb0; }
.neg  { color: #e8a070; }
.zero { color: #666; }
.lbl  { font-size: .68rem; letter-spacing: .12em; text-transform: uppercase;
         color: #555; margin-bottom: .25rem; }
</style>
""", unsafe_allow_html=True)

st.title("Field Hymns · Autonomous Inheritance in Solresol")
st.markdown("""
**H(k, t) = V_x(k,t) σ_x + V_y(k,t) σ_y + ε(t) σ_z**

**V_x**: Self-consistent feedback.  
`V = tanh(λ·(⟨d†d⟩ − 0.5))` where `⟨d†d⟩ = Fermi-Dirac(ε_k + λ·V)`.  
V responds to occupation. Occupation responds to V. They seek the fixed point.

**V_y**: Particle Swarm Optimisation phase accumulation as simulated coupling.  
`V_y(k_i, t) = A · sin(n · k_i − n · 2πt/T)`.  
n is negotiated by the autonomous swarm. Command: n = 0, V_y = 0.

The Chern number is computed from the same (V_x, V_y) eigenstates that determine occupation at each step, which determines the chord interval.  
**The topology, feedback, and audio are the same computation.**

Command occupation drifts with ε. Without self-consistency, there can be no fixed point.  
Autonomous occupation is moderated by feedback. 

Topology emerges from the distributed optimisation process coupled to the system’s state.
""")
st.divider()

col_in, col_btn = st.columns([4, 1])
with col_in:
    text = st.text_input("Text input", value="YOUR TEXT HERE.")
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("▶  Run", use_container_width=True, type="primary")

if text:
    notes = text_to_notes(text)
    dur = STEP_DUR * N_T * N_CYCLES
    st.caption(f"Solresol DNA: `{' — '.join(notes)}`  ·  "
               f"{N_OBJ} objects · {N_T} steps/cycle · {N_CYCLES} cycles · "
               f"~{dur:.0f}s audio")

if run and text:
    notes = text_to_notes(text)

    with st.spinner("Autonomous swarm negotiating…"):
        auto_audio, auto_cycles = run_autonomous(notes)
    with st.spinner("Command system executing…"):
        cmd_audio, cmd_cycles = run_command(notes)

    # ── Chern numbers ──────────────────────────────────────────────────────

    st.markdown("### Chern Numbers")
    cols = st.columns(N_CYCLES + 1)
    for i, cd in enumerate(auto_cycles):
        with cols[i]:
            C   = cd['C']
            cls = "pos" if C > 0 else ("neg" if C < 0 else "zero")
            st.markdown(f"""
            <div class="cbox" style="background:#0d1a14;border:1px solid #2a5a3a">
              <div class="lbl">Autonomous · Cycle {cd['cycle']}</div>
              <div class="cv {cls}">C = {C:+d}</div>
              <div style="color:#444;font-size:.76rem;margin-top:.25rem">
                n = {cd['n']:+d} · vote = {cd['mean_vote']:+.2f}
              </div>
            </div>""", unsafe_allow_html=True)
    with cols[N_CYCLES]:
        st.markdown("""
        <div class="cbox" style="background:#1a0d0d;border:1px solid #5a2a2a">
          <div class="lbl">Command · All Cycles</div>
          <div class="cv zero">C = 0</div>
          <div style="color:#444;font-size:.76rem;margin-top:.25rem">
            n = 0 · V_y = 0 · no negotiation
          </div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── (V_x, V_y, ε) parameter space ─────────────────────────────────────

    st.markdown("### Parameter Space")

    fig, axes = plt.subplots(1, N_CYCLES + 1, figsize=(14, 4), facecolor="#0a0a0f")

    for cd, ax in zip(auto_cycles, axes[:N_CYCLES]):
        ax.set_facecolor("#0f0f18")
        vx = cd['vx_trace']
        vy = cd['vy_trace']
        ep = cd['eps_trace']
        oc = cd['occ_trace']

        sc = ax.scatter(vx, ep, c=oc, cmap='RdYlGn',
                        vmin=0, vmax=1, s=18, alpha=0.85, zorder=3)
        ax.plot(vx, ep, color='#888', lw=0.6, alpha=0.4, zorder=2)
        ax.scatter([0], [0], color='#ff5555', s=55, zorder=5,
                   label='degeneracy')
        ax.set_xlabel("V_x (feedback)", color='#777', fontsize=8)
        ax.set_ylabel("ε", color='#777', fontsize=8)
        ax.set_title(f"Autonomous Cycle {cd['cycle']}  C={cd['C']:+d}  n={cd['n']:+d}",
                     color='#e8e8e8', fontsize=9)
        ax.legend(fontsize=7, facecolor='#111', edgecolor='#333',
                  labelcolor='#999')
        ax.tick_params(colors='#555', labelsize=7)
        for sp in ax.spines.values(): sp.set_color('#2a2a3a')

    ax = axes[N_CYCLES]
    ax.set_facecolor("#0f0f18")
    last_cmd = cmd_cycles[-1]
    oc_cmd = last_cmd['occ_trace']
    ax.scatter(last_cmd['vx_trace'], last_cmd['eps_trace'],
               c=oc_cmd, cmap='RdYlGn', vmin=0, vmax=1, s=18, alpha=0.85)
    ax.plot(last_cmd['vx_trace'], last_cmd['eps_trace'],
            color='#e87070', lw=0.8, alpha=0.5)
    ax.scatter([0], [0], color='#ff5555', s=55, zorder=5)
    ax.set_xlabel("V_x (prescribed)", color='#777', fontsize=8)
    ax.set_ylabel("ε", color='#777', fontsize=8)
    ax.set_title(f"Command  C=0  (V_y=0)", color='#e8e8e8', fontsize=9)
    ax.tick_params(colors='#555', labelsize=7)
    for sp in ax.spines.values(): sp.set_color('#2a2a3a')

    plt.colorbar(sc, ax=axes, label="⟨d†d⟩", fraction=0.015)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.caption("Color = occupation ⟨d†d⟩. Green = near fixed point (0.5). "
               "Red = far from fixed point. Autonomous orbit is moderated by "
               "self-consistent feedback. Command drifts freely.")

    # ── Occupation traces ──────────────────────────────────────────────────

    st.markdown("### Occupation ⟨d†d⟩ Through Each Cycle")
    st.caption(
        "Autonomous: moderated by feedback, cannot reach extreme 0 or 1. "
        "Command: driven by ε alone, swings all the way. "
        "This difference is audible — the chord interval tracks it."
    )

    fig2, ax2 = plt.subplots(figsize=(11, 2.8), facecolor="#0a0a0f")
    ax2.set_facecolor("#0f0f18")
    colors_a = ['#7ecfb0', '#e8a070', '#6ba3e8']
    for i, cd in enumerate(auto_cycles):
        ax2.plot(range(N_T * i, N_T * (i + 1)), cd['occ_trace'],
                 color=colors_a[i], lw=1.4, alpha=0.9,
                 label=f"Auto cycle {cd['cycle']}  C={cd['C']:+d}")
    ax2.plot(range(N_T * N_CYCLES), cmd_cycles[0]['occ_trace'] * N_CYCLES,
             color='#e87070', lw=1.0, ls='--', alpha=0.7, label="Command")
    ax2.axhline(0.5, color='#fff', lw=0.6, ls=':', alpha=0.35,
                label='fixed point')
    ax2.set_ylim(0, 1); ax2.set_xlabel("Step", color='#777', fontsize=8)
    ax2.set_ylabel("⟨d†d⟩", color='#777', fontsize=8)
    ax2.legend(fontsize=7, facecolor='#111', edgecolor='#333',
               labelcolor='#ccc', loc='upper right', ncol=2)
    ax2.tick_params(colors='#555', labelsize=7)
    for sp in ax2.spines.values(): sp.set_color('#2a2a3a')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # ── Winding number votes ───────────────────────────────────────────────

    st.markdown("### Winding Number Negotiation")
    fig3, axes3 = plt.subplots(1, N_CYCLES, figsize=(12, 3.0),
                                facecolor="#0a0a0f")
    for cd, ax in zip(auto_cycles, axes3):
        ax.set_facecolor("#0f0f18")
        v     = cd['votes']
        clrs  = ['#7ecfb0' if x >= 0 else '#e8a070' for x in v]
        ax.bar(range(len(v)), v, color=clrs, alpha=0.85, width=0.65)
        ax.axhline(0, color='#444', lw=0.7)
        if cd['n'] != 0:
            ax.axhline(cd['n'], color='#eee', lw=0.8, ls='--', alpha=0.5)
        ax.set_title(f"Cycle {cd['cycle']}  n={cd['n']:+d}  C={cd['C']:+d}",
                     color='#e8e8e8', fontsize=9)
        ax.set_ylim(-(MAX_WIND + .7), MAX_WIND + .7)
        ax.tick_params(colors='#555', labelsize=7)
        ax.set_xlabel("Object", color='#666', fontsize=8)
        ax.set_ylabel("Vote", color='#666', fontsize=8)
        for sp in ax.spines.values(): sp.set_color('#2a2a3a')
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    # ── Type evolution ─────────────────────────────────────────────────────

    if any(cd.get('type_changes') for cd in auto_cycles):
        st.markdown("### Emergent Type Evolution")
        st.caption(
            "`doesNotUnderstand`: objects adopt neighbor types when "
            "neighbor fitness exceeds own. Inheritance by exchange, not declaration."
        )
        orig  = [notes[i % len(notes)] for i in range(N_OBJ)]
        final = auto_cycles[-1]['final_types']
        nc    = {n: plt.cm.Set3(i / len(NOTES)) for i, n in enumerate(NOTES)}

        fig4, ax4s = plt.subplots(1, 2, figsize=(11, 1.6), facecolor="#0a0a0f")
        for ax, types, title in zip(ax4s, [orig, final],
                                    ["Original DNA", f"Final (Cycle {N_CYCLES})"]):
            ax.set_facecolor("#0a0a0f"); ax.axis("off")
            ax.set_xlim(0, len(types)); ax.set_ylim(0, 1)
            ax.set_title(title, color="#e8e8e8", fontsize=9, pad=2)
            for j, note in enumerate(types):
                ax.add_patch(plt.Rectangle((j, 0), 1, 1,
                             color=nc[note], alpha=0.85))
                ax.text(j + .5, .5, note, ha="center", va="center",
                        fontsize=8, color="#111", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    # ── Audio ──────────────────────────────────────────────────────────────

    st.markdown("### Audio")
    st.caption(
        "Autonomous: chords from (ε, occupation). "
        "The interval changes as occupation tracks the feedback loop. "
        "Command: single notes, no chord, no fixed point."
    )

    def to_wav(audio: np.ndarray) -> bytes:
        buf = io.BytesIO()
        a   = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        with wv.open(buf, "w") as wf:
            wf.setnchannels(1); wf.setsampwidth(2)
            wf.setframerate(44100); wf.writeframes(a.tobytes())
        return buf.getvalue()

    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Autonomous** · (ε, V*) chord · Feedback + PSO phase")
        st.audio(to_wav(auto_audio), format="audio/wav")
    with cb:
        st.markdown("**Command** · single note · no feedback · C = 0")
        st.audio(to_wav(cmd_audio), format="audio/wav")

    # ── Table ──────────────────────────────────────────────────────────────

    st.divider()
    st.markdown("### Per-Cycle Detail")
    ta, tb = st.columns(2)
    with ta:
        st.markdown("**Autonomous**")
        st.dataframe(
            [{"Cycle": cd["cycle"], "n": f"{cd['n']:+d}",
              "C": f"{cd['C']:+d}", "vote": f"{cd['mean_vote']:+.2f}",
              "occ min": f"{min(cd['occ_trace']):.3f}",
              "occ max": f"{max(cd['occ_trace']):.3f}",
              "evolved": len(cd.get("type_changes", []))}
             for cd in auto_cycles],
            use_container_width=True, hide_index=True
        )
    with tb:
        st.markdown("**Command**")
        st.dataframe(
            [{"Cycle": cd["cycle"], "n": "0", "C": "0",
              "vote": "0.00",
              "occ min": f"{min(cd['occ_trace']):.3f}",
              "occ max": f"{max(cd['occ_trace']):.3f}",
              "evolved": 0}
             for cd in cmd_cycles],
            use_container_width=True, hide_index=True
        )

else:
    st.info("Enter text and press ▶ Run.")
