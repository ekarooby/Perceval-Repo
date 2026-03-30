"""
Roy D2p — Deterministic Grover Search, N=6
Matches the physical chip structure from:
  Li et al., Sci. China Phys. Mech. Astron. 66, 290311 (2023)

Chip has 5 separate sections (Figure 3 of the paper):
  (i)   Preparation  : H₆ maps |0⟩ → |s⟩  (triangle mesh — sparse)
  (ii)  Oracle 1     : O  (6 phase shifters, one = π on marked mode)
  (iii) Amplification 1 : D(φ₁)  (triangle mesh — full decomposition)
  (iv)  Oracle 2     : O  (6 phase shifters, same as Oracle 1)
  (v)   Amplification 2 : D(φ₂)  (triangle mesh — full decomposition)

MZI building block follows exactly perceval.quandela.net/docs/v0.11/circuits.html:
  ub = Circuit(2) // BS() // (0, PS(φ_a)) // BS() // (1, PS(φ_b))
  Structure: BS → PS(mode 0, φ_a) → BS → PS(mode 1, φ_b)

All unitary sections decomposed with Circuit.decomposition() which
computes the EXACT MZI angles that implement each target matrix.
This is superior to GenericInterferometer which only gives topology
with symbolic (unfitted) parameters.

INPUT:  |1,0,0,0,0,0⟩  — single photon in mode 0
OUTPUT: photon exits at marked mode with P = 1  (deterministic)
"""

import os
import webbrowser
import numpy as np
from scipy.optimize import minimize, differential_evolution
import perceval as pcvl
import matplotlib.pyplot as plt
from perceval.rendering.circuit import SymbSkin

# ════════════════════════════════════════════════════════════════
# 1. PARAMETERS
# ════════════════════════════════════════════════════════════════
N           = 6
marked_item = 3      # 0-indexed; change to 0–5
OUTPUT_DIR  = r"C:\Users\ekaroob\Documents\Perceval-Repo"

print("=" * 64)
print(f"  Roy D2p (paper structure)  |  N={N}  |  marked={marked_item}")
print("=" * 64)

# ════════════════════════════════════════════════════════════════
# 2. BUILD THE FIVE SECTION MATRICES
# ════════════════════════════════════════════════════════════════
s_vec = np.ones(N, dtype=complex) / np.sqrt(N)

# ── Section (i): Preparation H₆ ─────────────────────────────────
# Householder reflection: maps |0⟩ → |s⟩
# Sparse unitary — triangle mesh needs fewer MZIs than Clements
def make_H(N):
    e0 = np.zeros(N, dtype=complex); e0[0] = 1.0
    v  = e0 - s_vec
    v /= np.linalg.norm(v)
    return np.eye(N, dtype=complex) - 2.0 * np.outer(v, v.conj())

# ── Section (ii)/(iv): Oracle O ──────────────────────────────────
# Phase flip on marked mode w: O = I − 2|w⟩⟨w|
# 6 independent PS: π on marked mode, 0 on all others
def make_O(w, N):
    O = np.eye(N, dtype=complex)
    O[w, w] = -1.0
    return O

# ── Section (iii)/(v): Amplification D(φ) ───────────────────────
# Generalised diffusion: D(φ) = I − (1−e^{iφ})|s⟩⟨s|
# General N×N unitary — triangle mesh (Reck decomposition)
def make_D(phi):
    s = s_vec.reshape(-1, 1)
    return np.eye(N, dtype=complex) - (1.0 - np.exp(1j * phi)) * (s @ s.conj().T)

# ════════════════════════════════════════════════════════════════
# 3. FIND OPTIMAL PHASES φ₁ AND φ₂  (scipy optimisation)
# ════════════════════════════════════════════════════════════════
H_np = make_H(N)
O_np = make_O(marked_item, N)

def success_prob(phi1, phi2):
    U = make_D(phi2) @ O_np @ make_D(phi1) @ O_np @ H_np
    return float(abs(U[marked_item, 0]) ** 2)

print("\n[Step 1] Finding D2p phases via scipy...")
bounds = [(0, 2*np.pi), (0, 2*np.pi)]
de = differential_evolution(
    lambda p: -success_prob(p[0], p[1]),
    bounds, maxiter=3000, tol=1e-14, seed=42, polish=True
)
nm = minimize(
    lambda p: -success_prob(p[0], p[1]),
    de.x, method='Nelder-Mead',
    options={'xatol': 1e-14, 'fatol': 1e-14, 'maxiter': 100_000}
)
phi1_opt, phi2_opt = nm.x
print(f"  φ₁ = {phi1_opt:.8f} rad  ({np.degrees(phi1_opt):.4f}°)")
print(f"  φ₂ = {phi2_opt:.8f} rad  ({np.degrees(phi2_opt):.4f}°)")
print(f"  P(success) = {success_prob(phi1_opt, phi2_opt):.10f}")

# ════════════════════════════════════════════════════════════════
# 4. BUILD THE 5-SECTION PERCEVAL CIRCUIT  (Unitary blocks)
#    Used for SLOS simulation only
# ════════════════════════════════════════════════════════════════
print("\n[Step 2] Building 5-section Perceval circuit (for simulation)...")

H_mat  = make_H(N)
O_mat  = make_O(marked_item, N)
D1_mat = make_D(phi1_opt)
D2_mat = make_D(phi2_opt)

def unitary_block(matrix, name):
    """Wrap a numpy matrix as a single Perceval Unitary block."""
    c = pcvl.Circuit(N, name=name)
    c.add(0, pcvl.Unitary(pcvl.Matrix(matrix)))
    return c

def oracle_block(w, N, name):
    """6 independent PS: π on marked mode, 0 on all others."""
    c = pcvl.Circuit(N, name=name)
    for i in range(N):
        phase = np.pi if i == w else 0.0
        c.add(i, pcvl.PS(phase))
    return c

prep_circuit    = unitary_block(H_mat,  "H6_prep")
oracle1_circuit = oracle_block(marked_item, N, "Oracle_1")
amp1_circuit    = unitary_block(D1_mat, "Amp_D_phi1")
oracle2_circuit = oracle_block(marked_item, N, "Oracle_2")
amp2_circuit    = unitary_block(D2_mat, "Amp_D_phi2")

full_circuit = (
    pcvl.Circuit(N, name="Roy_D2p_5section")
    // prep_circuit
    // oracle1_circuit
    // amp1_circuit
    // oracle2_circuit
    // amp2_circuit
)

# ════════════════════════════════════════════════════════════════
# 5. SECTION BREAKDOWN
# ════════════════════════════════════════════════════════════════
print("\n  Section-by-section breakdown:")
print(f"  {'Section':<28} {'Expected on chip':<32} {'Components'}")
print("  " + "-"*72)
sections_info = [
    ("(i)  Preparation H₆",       "triangle mesh  (~5 MZIs)",      prep_circuit),
    ("(ii) Oracle 1",              "6 PS  (one = π)",               oracle1_circuit),
    ("(iii) Amplification D(φ₁)", "triangle mesh  (15 MZIs + PS)", amp1_circuit),
    ("(iv) Oracle 2",              "6 PS  (one = π)",               oracle2_circuit),
    ("(v)  Amplification D(φ₂)",  "triangle mesh  (15 MZIs + PS)", amp2_circuit),
]
for label, expected, circ in sections_info:
    print(f"  {label:<28} {expected:<32} {circ.ncomponents()}")

# ════════════════════════════════════════════════════════════════
# 6. SIMULATE WITH PERCEVAL SLOS
# ════════════════════════════════════════════════════════════════
print("\n[Step 3] Running Perceval SLOS simulation...")

input_state   = pcvl.BasicState([1] + [0] * (N - 1))
output_states = [pcvl.BasicState([0]*i + [1] + [0]*(N-1-i)) for i in range(N)]

processor = pcvl.Processor("SLOS", full_circuit)
analyzer  = pcvl.algorithm.Analyzer(
    processor,
    input_states=[input_state],
    output_states=output_states
)

print("\n  Input:  |1,0,0,0,0,0⟩  (photon in mode 0)")
print("  Prep section maps |0⟩ → |s⟩ = (1/√6)(|0⟩+|1⟩+|2⟩+|3⟩+|4⟩+|5⟩)")
print()
print(f"  {'Mode':<8} {'P(detect)':<16} {'Note'}")
print("  " + "-"*46)

probs = []
for i in range(N):
    p = float(analyzer.distribution[0][i].real)
    probs.append(p)
    note = "<-- TARGET  (Roy D2p, P=1)" if i == marked_item else ""
    print(f"  {i:<8} {p:<16.8f} {note}")

print(f"\n  Classical:       P = {1/N:.4f}")
print(f"  Grover (k=1):    P ≈ 0.9074")
print(f"  Roy D2p (k=2):   P = {probs[marked_item]:.8f}  ← deterministic")

# ════════════════════════════════════════════════════════════════
# 7. PLOT — probability bars + chip section diagram
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    f"Roy D2p — 5-section chip  |  N={N}, marked={marked_item}",
    fontsize=12, y=1.01
)

ax = axes[0]
colors = ['#D85A30' if i == marked_item else '#378ADD' for i in range(N)]
bars = ax.bar(range(N), probs, color=colors, edgecolor='white', width=0.6)
ax.axhline(1/N,    color='#888780', lw=1.2, ls='--', label=f'Classical 1/{N}')
ax.axhline(0.9074, color='#EF9F27', lw=1.2, ls=':',  label='Grover max (0.907)')
ax.set_xlabel("Output mode"); ax.set_ylabel("P(detect)")
ax.set_title("Perceval SLOS output")
ax.set_xticks(range(N))
ax.set_xticklabels([f"Mode {i}" for i in range(N)], fontsize=9)
ax.set_ylim(0, 1.2)
for bar, p in zip(bars, probs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{p:.3f}', ha='center', fontsize=9)
ax.legend(fontsize=9)

ax2 = axes[1]
ax2.set_xlim(0, 14); ax2.set_ylim(-0.5, 4.5); ax2.axis('off')
ax2.set_title("Physical chip — 5 sections (Li et al. 2023, Fig. 3)")
chip_sections = [
    (1.0,  '#1D9E75', '(i)\nPrep\nH₆'),
    (3.2,  '#D85A30', '(ii)\nOracle\nO'),
    (5.4,  '#7F77DD', '(iii)\nAmp\nD(φ₁)'),
    (7.6,  '#D85A30', '(iv)\nOracle\nO'),
    (9.8,  '#7F77DD', '(v)\nAmp\nD(φ₂)'),
]
y_mid = 2.0
for x, color, label in chip_sections:
    ax2.add_patch(plt.Rectangle((x-0.9, y_mid-1.1), 1.8, 2.2,
                                color=color, alpha=0.85, zorder=2))
    ax2.text(x, y_mid, label, ha='center', va='center',
             fontsize=8.5, color='white', fontweight='bold', zorder=3)
for x in [1.9, 4.1, 6.3, 8.5]:
    ax2.annotate('', xy=(x+0.3, y_mid), xytext=(x, y_mid),
                 arrowprops=dict(arrowstyle='->', color='#444441', lw=1.5))
ax2.text(-0.2, y_mid, '|1,0,0,0,0,0⟩\n(photon in\nmode 0)',
         ha='center', va='center', fontsize=8, color='#444441')
ax2.annotate('', xy=(0.1, y_mid), xytext=(-0.8, y_mid),
             arrowprops=dict(arrowstyle='->', color='#444441', lw=1.5))
ax2.text(11.8, y_mid, f'P(mode {marked_item})=1.0\n(deterministic)',
         ha='center', va='center', fontsize=8.5, color='#D85A30')
ax2.annotate('', xy=(10.9, y_mid), xytext=(10.6, y_mid),
             arrowprops=dict(arrowstyle='->', color='#444441', lw=1.5))
ax2.text(6.6, 0.2,
         f'φ₁ = {np.degrees(phi1_opt):.2f}°  |  φ₂ = {np.degrees(phi2_opt):.2f}°\n'
         f'Same φ values work for all 6 target modes',
         ha='center', fontsize=9, color='#5F5E5A',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#F1EFE8', edgecolor='#B4B2A9'))
plt.tight_layout()
plot_out = os.path.join(OUTPUT_DIR, "roy_d2p_5section.png")
plt.savefig(plot_out, dpi=150, bbox_inches='tight')
print(f"\n  Plot saved: {plot_out}")
plt.show()

# ════════════════════════════════════════════════════════════════
# 8. BUILD MZI-DECOMPOSED CIRCUIT  (for visualisation)
#
#  MZI building block — EXACTLY from circuits.html:
#    ub = Circuit(2, name="ub")
#         // BS()
#         // (0, PS(phi=Parameter("φ_a")))   ← PS on mode 0
#         // BS()
#         // (1, PS(phi=Parameter("φ_b")))   ← PS on mode 1
#
#  All three unitary sections use Circuit.decomposition() with
#  this ub block and TRIANGLE shape. This computes the EXACT
#  numerical MZI angles that implement each target matrix —
#  not just a symbolic topology.
#
#  phase_shifter_fn=pcvl.PS adds the final layer of PS to make
#  the decomposed circuit strictly equivalent to the target matrix.
# ════════════════════════════════════════════════════════════════
print("\n[Step 4] Decomposing all unitary sections into MZI circuits...")

# ── ub: the MZI building block from circuits.html ────────────────
# BS → PS(mode 0, φ_a) → BS → PS(mode 1, φ_b)
# Note: second PS is on mode 1, not mode 0 — this is the key
# difference from the BS-based notebook's MZI definition.
ub = (
    pcvl.Circuit(2, name="ub")
    // pcvl.BS()
    // (0, pcvl.PS(phi=pcvl.Parameter("φ_a")))
    // pcvl.BS()
    // (1, pcvl.PS(phi=pcvl.Parameter("φ_b")))
)

# ── Section (i): Preparation H₆ ─────────────────────────────────
# Householder unitary is sparse — triangle decomposition
# naturally exploits this and uses fewer MZIs
print("  Decomposing (i)   Preparation H₆  ...")
prep_mzi = pcvl.Circuit.decomposition(
    pcvl.Matrix(H_mat),
    ub,
    phase_shifter_fn=pcvl.PS,
    shape=pcvl.InterferometerShape.TRIANGLE
)
print(f"    → {prep_mzi.ncomponents()} components")

# ── Section (iii): Amplification D(φ₁) ──────────────────────────
print("  Decomposing (iii) Amplification D(φ₁) ...")
amp1_mzi = pcvl.Circuit.decomposition(
    pcvl.Matrix(D1_mat),
    ub,
    phase_shifter_fn=pcvl.PS,
    shape=pcvl.InterferometerShape.TRIANGLE
)
print(f"    → {amp1_mzi.ncomponents()} components")

# ── Section (v): Amplification D(φ₂) ────────────────────────────
print("  Decomposing (v)   Amplification D(φ₂) ...")
amp2_mzi = pcvl.Circuit.decomposition(
    pcvl.Matrix(D2_mat),
    ub,
    phase_shifter_fn=pcvl.PS,
    shape=pcvl.InterferometerShape.TRIANGLE
)
print(f"    → {amp2_mzi.ncomponents()} components")

# ── Chain all 5 sections ─────────────────────────────────────────
full_mzi_circuit = (
    pcvl.Circuit(N, name="Roy_D2p_MZI_v7")
    // prep_mzi
    // oracle1_circuit
    // amp1_mzi
    // oracle2_circuit
    // amp2_mzi
)
print(f"\n  Full MZI circuit: {full_mzi_circuit.ncomponents()} components total")

# ════════════════════════════════════════════════════════════════
# 9. SAVE CIRCUIT DIAGRAMS AND AUTO-OPEN IN BROWSER
# ════════════════════════════════════════════════════════════════
print("\n[Step 5] Saving circuit diagrams and opening in browser...")

compact_skin = SymbSkin(compact_display=True)

section_circuits = [
    ("i_prep",     prep_mzi,        "(i)   Preparation H₆     — triangle mesh"),
    ("ii_oracle1", oracle1_circuit, "(ii)  Oracle 1            — 6 PS"),
    ("iii_amp1",   amp1_mzi,        "(iii) Amplification D(φ₁) — triangle mesh"),
    ("iv_oracle2", oracle2_circuit, "(iv)  Oracle 2            — 6 PS"),
    ("v_amp2",     amp2_mzi,        "(v)   Amplification D(φ₂) — triangle mesh"),
]

for label, circ, description in section_circuits:
    html_path = os.path.join(OUTPUT_DIR, f"section_{label}.html")
    png_path  = os.path.join(OUTPUT_DIR, f"section_{label}.png")

    # Primary: HTML/SVG
    pcvl.pdisplay_to_file(
        circ,
        path=html_path,
        output_format=pcvl.Format.HTML,
        recursive=True,
        skin=compact_skin,
        render_size=0.6,
    )

    if os.path.isfile(html_path) and os.path.getsize(html_path) > 0:
        print(f"  ✓ {description}")
        print(f"      → {html_path}")
        webbrowser.open(f"file:///{html_path}")
    else:
        # Fallback: PNG via matplotlib
        pcvl.pdisplay_to_file(
            circ,
            path=png_path,
            output_format=pcvl.Format.MPLOT,
            recursive=True,
            skin=compact_skin,
            render_size=0.6,
        )
        if os.path.isfile(png_path) and os.path.getsize(png_path) > 0:
            print(f"  ✓ {description}  (PNG fallback)")
            print(f"      → {png_path}")
            webbrowser.open(f"file:///{png_path}")
        else:
            print(f"  ✗ {description}  — both failed, displaying on screen")
            pcvl.pdisplay(circ, recursive=True, skin=compact_skin, render_size=0.6)

print("\n" + "=" * 64)
print("  Done.")
print("=" * 64)