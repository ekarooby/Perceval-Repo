import numpy as np
import matplotlib.pyplot as plt
import perceval as pcvl
import perceval.components as comp

phiset = np.array([
    -0.05263717,  0.05904224, -0.06758093,  0.07972595,
    -0.09871604,  0.13298893, -0.21285675,  0.56621395,
     0.56621395, -0.21285675,  0.13298893, -0.09871604,
     0.07972595, -0.06758093,  0.05904224, -0.05263717
])

L = 15

# ============================================================
# Verified conventions:
#   Perceval PS(phi) = [[e^{i*phi}, 0], [0, 1]] on mode 0
#   Signal unitary Rx_signal(u) injected via pcvl.Unitary
#   Output: f(u) = 2 * Im(psi[0])  where psi = U|0>
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

def classical_qsp(phiset, u, L):
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
    circuit = pcvl.Circuit(2, name=f"QSP_STEP_L{L}")
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
# Sweep
# ============================================================

n_points = 300
u_grid = np.linspace(-1.0, 1.0, n_points)
x_grid = u_grid * np.pi

f_classical = np.zeros(n_points)
f_pic       = np.zeros(n_points)

print(f"Sweeping {n_points} points...")
for i, u in enumerate(u_grid):
    f_classical[i] = classical_qsp(phiset, u, L)
    f_pic[i]       = get_pic_output(phiset, u, L)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{n_points} done")

print("Sweep complete.")

# ============================================================
# MSE report
# ============================================================

step_true     = np.where(x_grid >= 0, 1.0, -1.0)
mse_classical = np.mean((f_classical - step_true) ** 2)
mse_pic       = np.mean((f_pic       - step_true) ** 2)
mse_residual  = np.mean((f_pic - f_classical)     ** 2)

print("\n========== MSE Report ==========")
print(f"  MSE Classical vs Target : {mse_classical:.6f}")
print(f"  MSE PIC vs Target       : {mse_pic:.6f}")
print(f"  MSE PIC vs Classical    : {mse_residual:.10f}")
print("=================================")

# ============================================================
# Final plot matching paper Fig. 2 style
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Photonic QSP on PIC MZI mesh  STEP function L=15", fontsize=13)

ax = axes[0]
ax.plot(x_grid, step_true,   'k-', lw=2,   label="Target STEP")
ax.plot(x_grid, f_classical, 'r-', lw=1.5, label="Classical QSP (paper Fig.2)")
ax.plot(x_grid, f_pic,       'b.', ms=3,   label="Photonic PIC output")
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-1.3, 1.3])
ax.set_xticks([-np.pi, 0, np.pi])
ax.set_xticklabels(["-pi", "0", "pi"], fontsize=12)
ax.set_xlabel("Input signal x", fontsize=12)
ax.set_ylabel("Function value f(x)", fontsize=12)
ax.set_title(f"L = {L} layers", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
diff = f_pic - f_classical
ax2.plot(x_grid, diff, 'purple', lw=1.5, label="PIC minus Classical")
ax2.axhline(0, color='k', lw=0.8, linestyle='--')
ax2.fill_between(x_grid, diff, alpha=0.2, color='purple')
ax2.set_xlim([-np.pi, np.pi])
ax2.set_xticks([-np.pi, 0, np.pi])
ax2.set_xticklabels(["-pi", "0", "pi"], fontsize=12)
ax2.set_xlabel("Input signal x", fontsize=12)
ax2.set_ylabel("Residual", fontsize=12)
ax2.set_title("PIC vs Classical residual (should be zero)", fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pic_qsp_step_L15_FINAL.png", dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: pic_qsp_step_L15_FINAL.png")

# ============================================================
# PIC resource count for paper
# ============================================================

circ_sample = build_qsp_pic(phiset, 0.0, L)
n_ps  = sum(1 for _, c in circ_sample._components
            if isinstance(c, comp.PS))
n_uni = sum(1 for _, c in circ_sample._components
            if isinstance(c, pcvl.Unitary))

print("\n========== PIC Resource Count (for paper) ==========")
print(f"  QSP layers L            : {L}")
print(f"  Phase shifters PS       : {n_ps}   (= L+1 = {L+1})")
print(f"  Signal unitaries MZI    : {n_uni}  (= L   = {L})")
print(f"  Total components        : {n_ps + n_uni}")
print(f"  Waveguide modes         : 2  (dual-rail qubit)")
print("=====================================================")
