"""
Roy D2p — Deterministic Grover Search, N=6
Matches the physical chip structure from:
  Li et al., Sci. China Phys. Mech. Astron. 66, 290311 (2023)

Chip has 5 separate sections (Figure 3 of the paper):
  (i)   Preparation  : H₆ maps |0⟩ → |s⟩  (rectangular Clements mesh)
  (ii)  Oracle 1     : O  (6 phase shifters, one = π on marked mode)
  (iii) Amplification 1 : D(φ₁)  (rectangular Clements mesh)
  (iv)  Oracle 2     : O  (6 phase shifters, same as Oracle 1)
  (v)   Amplification 2 : D(φ₂)  (rectangular Clements mesh)

v9 — RECTANGLE mesh + EXACT fitted angles via basinhopping:
  - GenericInterferometer with InterferometerShape.RECTANGLE
    gives the correct Clements layout matching Li et al. 2023
  - MZI building block from BS-based notebook (perceval docs):
      BS(45°) → PS(P1) → BS(45°) → PS(P2)
  - basinhopping global optimiser (exactly from Perceval docs)
    fits the PS angles of each GenericInterferometer to exactly
    match H_mat, D1_mat, D2_mat — escapes local minima that
    trap simple gradient descent
  - This gives BOTH correct Clements layout AND exact angles

INPUT:  |1,0,0,0,0,0⟩  — single photon in mode 0
OUTPUT: photon exits at marked mode with P = 1  (deterministic)
"""

import os
import webbrowser
import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping
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
print(f"  Roy D2p  v9  |  N={N}  |  marked={marked_item}")
print("  RECTANGLE Clements mesh + basinhopping exact angles")
print("=" * 64)

# ════════════════════════════════════════════════════════════════
# 2. BUILD THE FIVE SECTION MATRICES
# ════════════════════════════════════════════════════════════════
s_vec = np.ones(N, dtype=complex) / np.sqrt(N)

def make_H(N):
    """Householder reflection: maps |0⟩ → |s⟩"""
    e0 = np.zeros(N, dtype=complex); e0[0] = 1.0
    v  = e0 - s_vec
    v /= np.linalg.norm(v)
    return np.eye(N, dtype=complex) - 2.0 * np.outer(v, v.conj())

def make_O(w, N):
    """Oracle: phase flip on marked mode w"""
    O = np.eye(N, dtype=complex)
    O[w, w] = -1.0
    return O

def make_D(phi):
    """Generalised diffusion: D(φ) = I − (1−e^{iφ})|s⟩⟨s|"""
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
#    Used for SLOS simulation — fast and exact
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
print(f"  {'Section':<28} {'Expected on chip':<34} {'Components'}")
print("  " + "-"*74)
for label, expected, circ in [
    ("(i)  Preparation H₆",       "rectangle mesh  (Clements)",    prep_circuit),
    ("(ii) Oracle 1",              "6 PS  (one = π)",               oracle1_circuit),
    ("(iii) Amplification D(φ₁)", "rectangle mesh  (Clements)",    amp1_circuit),
    ("(iv) Oracle 2",              "6 PS  (one = π)",               oracle2_circuit),
    ("(v)  Amplification D(φ₂)",  "rectangle mesh  (Clements)",    amp2_circuit),
]:
    print(f"  {label:<28} {expected:<34} {circ.ncomponents()}")

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
# 7. PLOT
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    f"Roy D2p v9 — Clements rectangle mesh  |  N={N}, marked={marked_item}",
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
         f'φ₁ = {np.degrees(phi1_opt):.2f}°  |  φ₂ = {np.degrees(phi2_opt):.2f}°',
         ha='center', fontsize=9, color='#5F5E5A',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#F1EFE8', edgecolor='#B4B2A9'))
plt.tight_layout()
plot_out = os.path.join(OUTPUT_DIR, "roy_d2p_v9_5section.png")
plt.savefig(plot_out, dpi=150, bbox_inches='tight')
print(f"\n  Plot saved: {plot_out}")
plt.show()

# ════════════════════════════════════════════════════════════════
# 8. BUILD MZI-DECOMPOSED CIRCUIT — RECTANGLE + EXACT ANGLES
#
#  Strategy — exactly from Perceval BS-based notebook:
#    1. Build GenericInterferometer RECTANGLE with standard MZI:
#          BS(45°) → PS(P1) → BS(45°) → PS(P2)
#    2. Get symbolic parameters with c.get_parameters()
#    3. Define infidelity(c, U_target, params, values):
#          sets values → computes unitary → returns 1 - fidelity
#    4. Use basinhopping (global search) to minimise infidelity
#       — this escapes local minima that trap L-BFGS-B alone
#    5. Lock best angles into circuit permanently
#
#  Result: RECTANGLE layout (Li et al. 2023) + exact angles
# ════════════════════════════════════════════════════════════════
print("\n[Step 4] Building RECTANGLE Clements circuits with exact angles")
print("         (basinhopping global optimisation — may take a few minutes)")

# ── Infidelity function — from BS-based notebook exactly ─────────
def infidelity(c, U_target, params, params_value):
    """
    Compute 1 - fidelity between circuit unitary and target matrix.
      F = |Tr(U_dag @ U_target)|² / (N * Tr(U_dag @ U))
    Returns 1 - |F|   (0 = perfect match)
    """
    for idx, p in enumerate(params_value):
        params[idx].set_value(p)
    U     = c.compute_unitary(use_symbolic=False)
    U_dag = np.transpose(np.conjugate(U))
    f     = abs(np.trace(U_dag @ U_target)) ** 2 / (c.m * np.trace(U_dag @ U))
    return 1 - abs(f)

# ── MZI building block — from BS-based notebook exactly ──────────
# BS(45°) → PS(P1) → BS(45°) → PS(P2)
# Both BSs fixed at perfect 50:50 (theta=45°)
# Only the two PS phases P1, P2 are variable
def make_mzi(P1, P2):
    return (
        pcvl.Circuit(2, name="mzi")
        .add((0, 1), pcvl.BS(theta=(45)/180*np.pi))
        .add(0,      pcvl.PS(P1))
        .add((0, 1), pcvl.BS(theta=(45)/180*np.pi))
        .add(0,      pcvl.PS(P2))
    )

# ── Fit a GenericInterferometer to a target matrix ───────────────
def fit_rectangle_circuit(U_target, prefix, n_try=10, n_iter=50):
    """
    Build a RECTANGLE GenericInterferometer and fit its PS angles
    to implement U_target via basinhopping global optimisation,
    exactly as used in the Perceval BS-based notebook.

    Parameters
    ----------
    U_target : np.ndarray — target N×N unitary matrix
    prefix   : str        — unique parameter name prefix
    n_try    : int        — number of independent basinhopping runs
    n_iter   : int        — basinhopping iterations per run
                            (200 is a good default; increase if WARN)

    Returns
    -------
    c        : fitted Perceval Circuit with locked-in angles
    best_infid : final infidelity (target < 1e-6)
    """
    # Build the RECTANGLE GenericInterferometer
    # depth=N with MZI block gives full Clements mesh for N modes
    c = pcvl.GenericInterferometer(
        N,
        fun_gen=lambda idx: make_mzi(
            pcvl.P(f"{prefix}_m{idx}"),
            pcvl.P(f"{prefix}_n{idx}")
        ),
        shape=pcvl.InterferometerShape.RECTANGLE,
        depth=N,
        phase_shifter_fun_gen=lambda idx: pcvl.PS(
            phi=pcvl.P(f"{prefix}_r{idx}")
        )
    )

    params   = c.get_parameters()
    n_params = len(params)
    print(f"    Circuit has {n_params} free parameters")

    best_infid  = np.inf
    best_params = None

    for attempt in range(n_try):
        # Random initial point — same strategy as BS-based notebook
        x0 = np.random.uniform(0, 2*np.pi, n_params)

        # basinhopping: global search that escapes local minima
        # stepsize=0.5, niter=n_iter — from the Perceval notebook
        # L-BFGS-B as local minimiser inside each basin hop
        res = basinhopping(
            lambda x: infidelity(c, U_target, params, x),
            x0,
            stepsize=0.5,
            niter=n_iter,
            minimizer_kwargs={
                'method': 'L-BFGS-B',
                'bounds': [(0, 2*np.pi)] * n_params,
                'options': {
                    'maxiter': 1000,
                    'ftol':    1e-15,
                    'gtol':    1e-12
                }
            }
        )

        print(f"    attempt {attempt+1}/{n_try}  "
              f"infidelity = {res.fun:.2e}")

        if res.fun < best_infid:
            best_infid  = res.fun
            best_params = res.x

        # Early exit if already converged
        if best_infid < 1e-10:
            print(f"    converged at attempt {attempt+1} ✓")
            break

    # Lock the best found angles permanently into the circuit
    for idx, p in enumerate(params):
        p.set_value(best_params[idx])

    return c, best_infid

# ── Fit each unitary section ─────────────────────────────────────
np.random.seed(42)   # reproducibility

print("\n  Fitting (i)  Preparation H₆  → RECTANGLE mesh...")
prep_mzi,  infid_prep  = fit_rectangle_circuit(
    H_mat,  prefix="prep", n_try=10, n_iter=50
)
print(f"    → final infidelity = {infid_prep:.2e}  "
      f"({'PASS ✓' if infid_prep < 1e-6 else 'WARN — increase n_iter to 500'})")

print("\n  Fitting (iii) Amplification D(φ₁) → RECTANGLE mesh...")
amp1_mzi,  infid_amp1  = fit_rectangle_circuit(
    D1_mat, prefix="amp1", n_try=10, n_iter=50
)
print(f"    → final infidelity = {infid_amp1:.2e}  "
      f"({'PASS ✓' if infid_amp1 < 1e-6 else 'WARN — increase n_iter to 500'})")

print("\n  Fitting (v)   Amplification D(φ₂) → RECTANGLE mesh...")
amp2_mzi,  infid_amp2  = fit_rectangle_circuit(
    D2_mat, prefix="amp2", n_try=10, n_iter=50
)
print(f"    → final infidelity = {infid_amp2:.2e}  "
      f"({'PASS ✓' if infid_amp2 < 1e-6 else 'WARN — increase n_iter to 500'})")

# ── Chain all 5 sections ─────────────────────────────────────────
full_mzi_circuit = (
    pcvl.Circuit(N, name="Roy_D2p_MZI_v9")
    // prep_mzi
    // oracle1_circuit
    // amp1_mzi
    // oracle2_circuit
    // amp2_mzi
)
print(f"\n  Full MZI circuit: {full_mzi_circuit.ncomponents()} components total")

# ── Fitting summary ───────────────────────────────────────────────
print("\n  Fitting summary:")
print(f"  {'Section':<28} {'Infidelity':<16} {'Status'}")
print("  " + "-"*56)
for label, infid in [
    ("(i)  Preparation H₆",       infid_prep),
    ("(iii) Amplification D(φ₁)", infid_amp1),
    ("(v)  Amplification D(φ₂)",  infid_amp2),
]:
    status = "PASS ✓" if infid < 1e-6 else "WARN ✗  increase n_iter to 500"
    print(f"  {label:<28} {infid:<16.2e} {status}")

# ════════════════════════════════════════════════════════════════
# 9. SAVE CIRCUIT DIAGRAMS AND AUTO-OPEN IN BROWSER
# ════════════════════════════════════════════════════════════════
print("\n[Step 5] Saving circuit diagrams and opening in browser...")

compact_skin = SymbSkin(compact_display=True)

section_circuits = [
    ("i_prep",     prep_mzi,        "(i)   Preparation H₆     — rectangle mesh (fitted)"),
    ("ii_oracle1", oracle1_circuit, "(ii)  Oracle 1            — 6 PS"),
    ("iii_amp1",   amp1_mzi,        "(iii) Amplification D(φ₁) — rectangle mesh (fitted)"),
    ("iv_oracle2", oracle2_circuit, "(iv)  Oracle 2            — 6 PS"),
    ("v_amp2",     amp2_mzi,        "(v)   Amplification D(φ₂) — rectangle mesh (fitted)"),
]

for label, circ, description in section_circuits:
    html_path = os.path.join(OUTPUT_DIR, f"v9_section_{label}.html")
    png_path  = os.path.join(OUTPUT_DIR, f"v9_section_{label}.png")

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
            pcvl.pdisplay(circ, recursive=True,
                          skin=compact_skin, render_size=0.6)

print("\n" + "=" * 64)
print("  Done.")
print("=" * 64)