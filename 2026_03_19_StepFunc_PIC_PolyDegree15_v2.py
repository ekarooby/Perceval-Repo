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
parity = 1

# ============================================================
# Step 1: verify Perceval BS.H matrix convention
# ============================================================

print("Verifying Perceval BS.H matrix convention:")
circ_test = pcvl.Circuit(2)
circ_test.add((0, 1), comp.BS.H(theta=np.pi/2))
U_test = circ_test.compute_unitary()
print(f"  BS.H(pi/2) unitary:")
print(f"  {U_test}")
print(f"  Expected Rx_signal(0): [[0, i],[i, 0]]")
print()

# ============================================================
# Matrix definitions
# ============================================================

def Rz_mat(phi):
    return np.array([
        [np.exp(1j * phi / 2), 0],
        [0, np.exp(-1j * phi / 2)]
    ])

def Rx_signal(x):
    sx = np.sqrt(max(1 - x**2, 0))
    return np.array([
        [x,       1j * sx],
        [1j * sx, x      ]
    ])

def classical_qsp(phiset, u, L):
    W = Rz_mat(phiset[0])
    for j in range(1, L + 1):
        W = Rz_mat(phiset[j]) @ Rx_signal(u) @ W
    psi = W @ np.array([1.0, 0.0])
    return 2.0 * np.imag(psi[0])

# ============================================================
# Build PIC using ONLY numpy matrix products
# but structured as Perceval components where possible.
# For signal unitary: use pcvl.Unitary() to inject
# the exact Rx_signal matrix into the circuit.
# ============================================================

def build_qsp_pic(phiset, u_val, L):
    """
    Build QSP circuit using:
      PS(phi_j)         for Rz gates (correct in Perceval)
      pcvl.Unitary(Rx)  for signal unitary (exact matrix injection)
    """
    circuit = pcvl.Circuit(2, name=f"QSP_STEP_L{L}")

    # Rx_signal matrix for this u_val
    sx = np.sqrt(max(1 - u_val**2, 0))
    Rx = np.array([
        [u_val,   1j * sx],
        [1j * sx, u_val  ]
    ])

    # Layer 0: Rz(phi_0) only
    circuit.add(0, comp.PS(float(phiset[0])))

    # Layers j = 1 to L
    for j in range(1, L + 1):
        # Signal unitary: inject Rx_signal exactly
        circuit.add((0, 1), pcvl.Unitary(pcvl.Matrix(Rx)))
        # Rz(phi_j)
        circuit.add(0, comp.PS(float(phiset[j])))

    return circuit

def get_pic_output(phiset, u_val, L):
    circuit = build_qsp_pic(phiset, u_val, L)
    U = circuit.compute_unitary()
    input_state = np.array([1.0, 0.0])
    psi_out = U @ input_state
    return 2.0 * np.imag(psi_out[0])

# ============================================================
# Sanity check at single point before full sweep
# ============================================================

u_test = 0.5
c_val = classical_qsp(phiset, u_test, L)
p_val = get_pic_output(phiset, u_test, L)
print("Sanity check at u=0.5:")
print(f"  Classical : {c_val:.6f}")
print(f"  PIC       : {p_val:.6f}")
print(f"  Difference: {abs(c_val - p_val):.2e}")
print("  Should be near zero.")
print()

# ============================================================
# Full sweep
# ============================================================

n_points = 300
u_grid = np.linspace(-1.0, 1.0, n_points)
x_grid = u_grid * np.pi

f_classical = np.zeros(n_points)
f_pic = np.zeros(n_points)

print(f"Sweeping {n_points} points...")
for i, u in enumerate(u_grid):
    f_classical[i] = classical_qsp(phiset, u, L)
    f_pic[i] = get_pic_output(phiset, u, L)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{n_points} done")

print("Sweep complete.")

# ============================================================
# MSE report
# ============================================================

step_true = np.where(x_grid >= 0, 1.0, -1.0)

mse_classical = np.mean((f_classical - step_true) ** 2)
mse_pic       = np.mean((f_pic       - step_true) ** 2)
mse_residual  = np.mean((f_pic - f_classical)     ** 2)

print("\n========== MSE Report ==========")
print(f"  MSE Classical vs Target : {mse_classical:.6f}")
print(f"  MSE PIC vs Target       : {mse_pic:.6f}")
print(f"  MSE PIC vs Classical    : {mse_residual:.8f}")
print("=================================")

# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Photonic QSP on PIC MZI mesh  STEP function L=15", fontsize=13)

ax = axes[0]
ax.plot(x_grid, step_true,   'k-', lw=2,   label="Target STEP")
ax.plot(x_grid, f_classical, 'r-', lw=1.5, label="Classical QSP (verified)")
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
ax2.set_title("PIC vs Classical residual should be near zero", fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pic_qsp_step_L15_v3.png", dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: pic_qsp_step_L15_v3.png")

# ============================================================
# PIC resource count
# ============================================================

circ_sample = build_qsp_pic(phiset, 0.0, L)
n_ps  = sum(1 for _, c in circ_sample._components if isinstance(c, comp.PS))
n_uni = sum(1 for _, c in circ_sample._components if isinstance(c, pcvl.Unitary))

print("\n========== PIC Resource Count ==========")
print(f"  QSP layers L            : {L}")
print(f"  Phase shifters PS       : {n_ps}")
print(f"  Signal unitaries MZI    : {n_uni}")
print(f"  Total components        : {n_ps + n_uni}")
print(f"  Waveguide modes         : 2  dual-rail qubit")
print("=========================================")