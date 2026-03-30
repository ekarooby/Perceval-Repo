import numpy as np
import matplotlib.pyplot as plt
import perceval as pcvl
import perceval.components as comp
from pyqsp import angle_sequence
from pyqsp.poly import PolyTaylorSeries

N_approx = 100
polydeg = 15

def step_surrogate(u):
    return (2.0 / np.pi) * np.arctan(N_approx * np.pi * u)

def step_true(x):
    return np.where(x >= 0, 1.0, -1.0)

print(f"Computing phases for polydeg={polydeg}...")

poly = PolyTaylorSeries().taylor_series(
    func=step_surrogate,
    degree=polydeg,
    max_scale=0.9,
    chebyshev_basis=True,
    cheb_samples=4 * polydeg
)

phiset, red_phiset, parity = angle_sequence.QuantumSignalProcessingPhases(
    poly,
    method="sym_qsp",
    chebyshev_basis=True
)

L = len(phiset) - 1
print(f"  Parity : {parity}")
print(f"  L      : {L}")
print(f"  Phases : {phiset}")

# ============================================================
# VERIFIED CORRECT convention from v4:
# PS_mat(phi) = [[e^{i*phi}, 0],[0, 1]]
# Signal: pcvl.Unitary(Rx)
# Output: 2*Im(psi[0])
# MSE = 0.00000000 confirmed
# ============================================================

def PS_mat(phi):
    return np.array([
        [np.exp(1j * phi), 0],
        [0,                1]
    ], dtype=complex)

def Rx_signal(x):
    sx = np.sqrt(max(1 - x**2, 0))
    return np.array([
        [x,       1j * sx],
        [1j * sx, x      ]
    ], dtype=complex)

def classical_qsp_ps(phiset, u, L):
    W = PS_mat(phiset[0])
    for j in range(1, L + 1):
        W = PS_mat(phiset[j]) @ Rx_signal(u) @ W
    psi = W @ np.array([1.0, 0.0])
    return 2.0 * np.imag(psi[0])

def build_qsp_pic(phiset, u_val, L):
    sx = np.sqrt(max(1 - u_val**2, 0))
    Rx = np.array([
        [u_val,   1j * sx],
        [1j * sx, u_val  ]
    ], dtype=complex)
    circuit = pcvl.Circuit(2, name=f"QSP_L{L}")
    circuit.add(0, comp.PS(float(phiset[0])))
    for j in range(1, L + 1):
        circuit.add((0, 1), pcvl.Unitary(pcvl.Matrix(Rx)))
        circuit.add(0, comp.PS(float(phiset[j])))
    return circuit

def get_pic_output(phiset, u_val, L):
    circuit = build_qsp_pic(phiset, u_val, L)
    U = circuit.compute_unitary()
    psi_out = U @ np.array([1.0, 0.0])
    return 2.0 * np.imag(psi_out[0])

# ============================================================
# Sanity check
# ============================================================

u_test = 0.5
c_val = classical_qsp_ps(phiset, u_test, L)
p_val = get_pic_output(phiset, u_test, L)

print("\n========== Sanity Check at u=0.5 ==========")
print(f"  Classical PS : {c_val:.6f}")
print(f"  PIC          : {p_val:.6f}")
print(f"  Difference   : {abs(c_val - p_val):.2e}  must be ~0")
print("============================================")

# ============================================================
# Full sweep
# ============================================================

n_points = 300
u_grid = np.linspace(-1.0, 1.0, n_points)
x_grid = u_grid * np.pi

f_classical = np.zeros(n_points)
f_pic       = np.zeros(n_points)

print(f"\nSweeping {n_points} points...")
for i, u in enumerate(u_grid):
    f_classical[i] = classical_qsp_ps(phiset, u, L)
    f_pic[i]       = get_pic_output(phiset, u, L)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{n_points} done")

print("Sweep complete.")

# ============================================================
# MSE
# ============================================================

step_vals    = step_true(x_grid)
mse_classic  = np.mean((f_classical - step_vals) ** 2)
mse_pic      = np.mean((f_pic       - step_vals) ** 2)
mse_residual = np.mean((f_pic - f_classical)     ** 2)

print("\n========== MSE Report ==========")
print(f"  MSE Classical vs Target : {mse_classic:.6f}")
print(f"  MSE PIC vs Target       : {mse_pic:.6f}")
print(f"  MSE PIC vs Classical    : {mse_residual:.2e}")
print("  PIC vs Classical must be ~0")
print("=================================")

# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Photonic QSP on PIC  STEP function  L={L}", fontsize=13)

ax = axes[0]
ax.plot(x_grid, step_vals,   'k-', lw=2,   label="Target STEP")
ax.plot(x_grid, f_classical, 'r-', lw=1.5, label="Classical QSP")
ax.plot(x_grid, f_pic,       'b.', ms=3,   label="Photonic PIC")
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-1.3, 1.3])
ax.set_xticks([-np.pi, 0, np.pi])
ax.set_xticklabels(["-pi", "0", "pi"], fontsize=11)
ax.set_xlabel("Input signal x", fontsize=11)
ax.set_ylabel("f(x)", fontsize=11)
ax.set_title(f"L={L}  MSE={mse_classic:.4f}", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
diff = f_pic - f_classical
ax2.plot(x_grid, diff, 'purple', lw=1.5, label=f"PIC minus Classical  MSE={mse_residual:.2e}")
ax2.axhline(0, color='k', lw=0.8, linestyle='--')
ax2.fill_between(x_grid, diff, alpha=0.2, color='purple')
ax2.set_xlim([-np.pi, np.pi])
ax2.set_xticks([-np.pi, 0, np.pi])
ax2.set_xticklabels(["-pi", "0", "pi"], fontsize=11)
ax2.set_xlabel("Input signal x", fontsize=11)
ax2.set_ylabel("Residual", fontsize=11)
ax2.set_title("PIC vs Classical residual  must be ~0", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pic_qsp_step_L15_v4_restored.png", dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: pic_qsp_step_L15_v4_restored.png")

# ============================================================
# PIC resource count
# ============================================================

circ_full = build_qsp_pic(phiset, 0.5, L)
n_ps  = sum(1 for _, c in circ_full._components if isinstance(c, comp.PS))
n_uni = sum(1 for _, c in circ_full._components if isinstance(c, pcvl.Unitary))

print("\n========== PIC Resource Count ==========")
print(f"  QSP layers L         : {L}")
print(f"  Phase shifters PS    : {n_ps}   (= L+1 = {L+1})")
print(f"  Signal unitaries     : {n_uni}  (= L   = {L})")
print(f"  Total components     : {n_ps + n_uni}")
print(f"  Waveguide modes      : 2  dual-rail qubit")
print("=========================================")
