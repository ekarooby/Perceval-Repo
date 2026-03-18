"""
Roy D2p — Deterministic Grover Search on N=6 items
using multiport path encoding (1 photon, 6 modes) in Perceval.

Reference: T. Roy, L. Jiang, D.I. Schuster,
           "Deterministic Grover search with a restricted oracle",
           Phys. Rev. Research 4, L022013 (2022)

Algorithm (D2p) for k=2 oracle calls:
   |ψ_final⟩ = D(φ₂) · O · D(φ₁) · O · H₆ |0⟩

where:
  H₆   — maps photon in mode 0 → uniform superposition |s⟩=(1/√6)∑|i⟩
  O    — standard oracle: phase flip (−1) on marked mode w
  D(φ) — generalized diffusion: I − (1−e^{iφ})|s⟩⟨s|
  φ₁,φ₂ — numerically optimized so that P(find w) = 1.0 exactly

For N=6, k=1 is provably impossible (requires cos φ = −5/4 > 1),
so k=2 is the minimum needed — exactly oracle-diffusion-oracle-diffusion.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
import perceval as pcvl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ════════════════════════════════════════════════════════════════
# 1. PARAMETERS — change marked_item to search for a different item
# ════════════════════════════════════════════════════════════════
N           = 6      # number of database items = number of waveguide modes
marked_item = 3      # item to search for (0-indexed: 0 … N-1)

print("=" * 60)
print(f"  Roy D2p Deterministic Grover  |  N={N}, marked={marked_item}")
print("=" * 60)

# ════════════════════════════════════════════════════════════════
# 2. MATRIX BUILDING BLOCKS
# ════════════════════════════════════════════════════════════════

# Uniform superposition vector |s⟩ = (1/√N) * ones
s_vec = np.ones(N, dtype=complex) / np.sqrt(N)

def oracle_matrix(w: int, N: int) -> np.ndarray:
    """
    Standard Grover oracle: phase flip (−1) on mode w.
    O = I − 2|w⟩⟨w|
    This is FIXED — Roy D2p requires no modification to the oracle.
    """
    O = np.eye(N, dtype=complex)
    O[w, w] = -1.0
    return O

def diffusion_matrix(phi: float) -> np.ndarray:
    """
    Generalized D2p diffusion with phase phi.
    D(phi) = I − (1 − e^{i*phi}) |s⟩⟨s|

    Special cases:
      phi = pi  →  D(pi) = I − 2|s⟩⟨s|  =  −(standard Grover diffusion)
      phi = 0   →  D(0)  = I             =  identity (no diffusion)
    """
    s = s_vec.reshape(-1, 1)
    D = np.eye(N, dtype=complex) - (1.0 - np.exp(1j * phi)) * (s @ s.conj().T)
    return D

def init_unitary(N: int) -> np.ndarray:
    """
    Initialization unitary H_N: maps |0⟩ → |s⟩ (equal superposition).
    Built as a Householder reflection that rotates e_0 onto |s⟩.
    First column = (1/√N, …, 1/√N)^T  ←  action on the input photon.
    """
    e0 = np.zeros(N, dtype=complex)
    e0[0] = 1.0
    v = e0 - s_vec
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-12:                      # e0 already equals s (N=1 edge case)
        return np.eye(N, dtype=complex)
    v = v / norm_v
    H = np.eye(N, dtype=complex) - 2.0 * np.outer(v, v.conj())
    return H

# ════════════════════════════════════════════════════════════════
# 3. D2p CIRCUIT UNITARY
#    U_total = D(φ₂) · O · D(φ₁) · O · H_N
#    Photon input: mode 0  →  action is first column of U_total
# ════════════════════════════════════════════════════════════════

H_N = init_unitary(N)
O   = oracle_matrix(marked_item, N)

def build_U_total(phi1: float, phi2: float) -> np.ndarray:
    D1 = diffusion_matrix(phi1)
    D2 = diffusion_matrix(phi2)
    return D2 @ O @ D1 @ O @ H_N

def success_prob(phi1: float, phi2: float) -> float:
    """P(photon exits at marked mode) for given phases."""
    U = build_U_total(phi1, phi2)
    psi_out = U[:, 0]          # first column = output state for input |0⟩
    return float(abs(psi_out[marked_item]) ** 2)

# ════════════════════════════════════════════════════════════════
# 4. FIND OPTIMAL PHASES φ₁, φ₂  (maximize success probability)
# ════════════════════════════════════════════════════════════════
print("\n[Step 1] Finding D2p phases φ₁, φ₂ ...")

def objective(params):
    return -success_prob(params[0], params[1])

# Global search using differential evolution (robust to local minima)
bounds = [(0, 2 * np.pi), (0, 2 * np.pi)]
de_result = differential_evolution(
    objective, bounds,
    maxiter=3000, tol=1e-14, seed=42,
    workers=1, polish=True
)

# Refine with Nelder-Mead from the best global point
nm_result = minimize(
    objective, de_result.x,
    method='Nelder-Mead',
    options={'xatol': 1e-14, 'fatol': 1e-14, 'maxiter': 100_000}
)

phi1_opt, phi2_opt = nm_result.x
p_success = success_prob(phi1_opt, phi2_opt)

print(f"  φ₁ = {phi1_opt:.8f} rad  ({np.degrees(phi1_opt):.4f}°)")
print(f"  φ₂ = {phi2_opt:.8f} rad  ({np.degrees(phi2_opt):.4f}°)")
print(f"  P(success) = {p_success:.10f}")

if p_success < 0.9999:
    print("  WARNING: success probability < 99.99% — trying finer search ...")
    # fallback: grid + local
    best_p, best_params = 0.0, (np.pi, np.pi)
    for a in np.linspace(0, 2*np.pi, 60):
        for b in np.linspace(0, 2*np.pi, 60):
            p = success_prob(a, b)
            if p > best_p:
                best_p, best_params = p, (a, b)
    r2 = minimize(objective, best_params, method='Nelder-Mead',
                  options={'xatol': 1e-14, 'fatol': 1e-14, 'maxiter': 100_000})
    if -r2.fun > p_success:
        phi1_opt, phi2_opt = r2.x
        p_success = -r2.fun
        print(f"  Refined φ₁={phi1_opt:.6f}, φ₂={phi2_opt:.6f}, P={p_success:.10f}")

# ════════════════════════════════════════════════════════════════
# 5. BUILD & VERIFY TOTAL UNITARY
# ════════════════════════════════════════════════════════════════
print("\n[Step 2] Building total unitary U_total ...")

U_total = build_U_total(phi1_opt, phi2_opt)

# Check unitarity: U · U† should equal I
unitarity_error = np.linalg.norm(U_total @ U_total.conj().T - np.eye(N))
print(f"  Unitarity error ‖U·U† − I‖ = {unitarity_error:.2e}  (should be ~1e-14)")

# Show the output amplitude on each mode
psi_out = U_total[:, 0]
print("\n  Output amplitudes (input: photon in mode 0):")
for i in range(N):
    mark = " ← TARGET" if i == marked_item else ""
    print(f"    mode {i}:  |amp|² = {abs(psi_out[i])**2:.8f}{mark}")

# ════════════════════════════════════════════════════════════════
# 6. PERCEVAL CIRCUIT
#    Use pcvl.Unitary to embed U_total as a single component.
#    Photon enters mode 0 → exits mode marked_item with P ≈ 1.
# ════════════════════════════════════════════════════════════════
print("\n[Step 3] Building Perceval circuit ...")

# Method A: Unitary block (exact, no decomposition error)
unitary_component = pcvl.Unitary(pcvl.Matrix(U_total))
circuit = pcvl.Circuit(N, name="Roy_D2p") // unitary_component

# Method B (alternative): decompose into BS + PS mesh (uncomment to use)
# circuit = pcvl.Circuit.decomposition(
#     pcvl.Matrix(U_total),
#     pcvl.BS(),
#     phase_shifter_fn=pcvl.PS,
#     shape="triangle"        # Reck decomposition; use "rectangle" for Clements
# )

print(f"  Circuit: {N} modes, 1 photon input, Unitary block embedding")

# ════════════════════════════════════════════════════════════════
# 7. SIMULATION WITH PERCEVAL
# ════════════════════════════════════════════════════════════════
print("\n[Step 4] Running Perceval SLOS simulation ...")

input_state   = pcvl.BasicState([1] + [0] * (N - 1))   # photon in mode 0
output_states = [
    pcvl.BasicState([0]*i + [1] + [0]*(N-1-i))
    for i in range(N)
]

processor = pcvl.Processor("SLOS", circuit)
analyzer  = pcvl.algorithm.Analyzer(
    processor,
    input_states=[input_state],
    output_states=output_states
)

print("\n  Results — output probabilities:")
print(f"  {'Mode':<8} {'Item':<8} {'P(detect)':<14} {'Note'}")
print("  " + "-" * 46)

probabilities = []
for i in range(N):
    p = float(analyzer.distribution[0][i].real)
    probabilities.append(p)
    note = "← TARGET (Roy D2p: P=1)" if i == marked_item else ""
    print(f"  {i:<8} {i:<8} {p:<14.8f} {note}")

print(f"\n  Classical random guess:  P = {1/N:.4f}  ({100/N:.1f}%)")
print(f"  Standard Grover (k=1):   P ≈ 0.9074  (90.7%)")
print(f"  Roy D2p (k=2):           P = {probabilities[marked_item]:.8f}  ({100*probabilities[marked_item]:.4f}%)")

# ════════════════════════════════════════════════════════════════
# 8. PLOT
# ════════════════════════════════════════════════════════════════
print("\n[Step 5] Generating plot ...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    f"Roy D2p Deterministic Grover Search  |  N={N} items, marked = item {marked_item}",
    fontsize=13, fontweight='500', y=1.01
)

# ── Left: probability bar chart ──────────────────────────────────
ax = axes[0]
colors = ['#E85D30' if i == marked_item else '#3B8BD4' for i in range(N)]
bars = ax.bar(range(N), probabilities, color=colors,
              edgecolor='white', linewidth=0.8, width=0.6)
ax.axhline(y=1/N, color='#888780', linestyle='--', linewidth=1.2,
           label=f'Classical random (1/{N} = {1/N:.2f})')
ax.axhline(y=0.9074, color='#EF9F27', linestyle=':', linewidth=1.2,
           label='Standard Grover max (0.907)')
ax.set_xlabel("Database item (mode index)", fontsize=11)
ax.set_ylabel("Detection probability", fontsize=11)
ax.set_title("Output probabilities", fontsize=11)
ax.set_xticks(range(N))
ax.set_xticklabels([f"Item {i}" for i in range(N)], fontsize=9)
ax.set_ylim(0, 1.15)
ax.legend(fontsize=9)

for bar, p in zip(bars, probabilities):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f'{p:.4f}', ha='center', va='bottom', fontsize=9)

patch_target  = mpatches.Patch(color='#E85D30', label=f'Marked item ({marked_item})')
patch_other   = mpatches.Patch(color='#3B8BD4', label='Unmarked items')
ax.legend(handles=[patch_target, patch_other,
                   plt.Line2D([0],[0], color='#888780', lw=1.2, ls='--',
                              label=f'Classical random (1/{N})'),
                   plt.Line2D([0],[0], color='#EF9F27', lw=1.2, ls=':',
                              label='Standard Grover max')],
          fontsize=9, loc='upper right')

# ── Right: circuit structure diagram ─────────────────────────────
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(-0.5, 4.5)
ax2.axis('off')
ax2.set_title("D2p circuit structure", fontsize=11)

stages = [
    (1.0,  '#5DCAA5', 'H₆\n(init)'),
    (3.0,  '#D85A30', 'Oracle O\n(−1 on w)'),
    (5.0,  '#7F77DD', f'Diffusion\nD(φ₁={np.degrees(phi1_opt):.1f}°)'),
    (7.0,  '#D85A30', 'Oracle O\n(−1 on w)'),
    (9.0,  '#7F77DD', f'Diffusion\nD(φ₂={np.degrees(phi2_opt):.1f}°)'),
]

y_mid = 2.0
for x, color, label in stages:
    ax2.add_patch(plt.Rectangle((x-0.7, y_mid-0.6), 1.4, 1.2,
                                color=color, alpha=0.85, zorder=2))
    ax2.text(x, y_mid, label, ha='center', va='center',
             fontsize=8.5, color='white', fontweight='bold', zorder=3)
    if x < 9.0:
        ax2.annotate('', xy=(x+0.9, y_mid), xytext=(x+0.7+0.3, y_mid),
                     arrowprops=dict(arrowstyle='->', color='#444441', lw=1.5))

ax2.text(5.0, 0.3,
         f'Input: |1,0,0,0,0,0⟩  →  Output: |{",".join("1" if i==marked_item else "0" for i in range(N))}⟩\n'
         f'P(success) = {p_success:.8f}  (deterministic)',
         ha='center', va='center', fontsize=9,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#F1EFE8', edgecolor='#B4B2A9'))

ax2.text(5.0, 3.8,
         f'φ₁ = {phi1_opt:.5f} rad  |  φ₂ = {phi2_opt:.5f} rad\n'
         f'N={N}  ·  k=2 oracle calls  ·  Roy D2p 2022',
         ha='center', va='center', fontsize=9, color='#444441')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/roy_d2p_result.png', dpi=150, bbox_inches='tight')
print("  Saved: roy_d2p_result.png")
plt.show()

print("\n" + "=" * 60)
print("  Done. Roy D2p deterministic search complete.")
print("=" * 60)
