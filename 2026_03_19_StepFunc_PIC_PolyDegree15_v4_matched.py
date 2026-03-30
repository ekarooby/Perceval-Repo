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
# Verify Perceval PS convention exactly
# PS(phi) in Perceval = [[e^{i*phi}, 0], [0, 1]]
# on mode 0 of a 2-mode circuit
# ============================================================

circ_ps_test = pcvl.Circuit(2)
circ_ps_test.add(0, comp.PS(np.pi))
U_ps_test = circ_ps_test.compute_unitary()
print("Perceval PS(pi) full 2x2 matrix:")
print(U_ps_test)
print()

# So Perceval PS(phi) acts as [[e^{i*phi}, 0], [0, 1]]
# We must redefine our Rz to match this

def PS_mat(phi):
    """Matches Perceval PS(phi) convention exactly."""
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

# ============================================================
# Redefine classical QSP using PS_mat instead of Rz_mat
# ============================================================

def classical_qsp_ps(phiset, u, L):
    """
    Classical QSP using Perceval PS convention.
    W = PS(L) @ Rx @ PS(L-1) @ Rx @ ... @ Rx @ PS(0)
    f(x) = 2 * Im(psi[0])
    """
    W = PS_mat(phiset[0])
    for j in range(1, L + 1):
        W = PS_mat(phiset[j]) @ Rx_signal(u) @ W
    psi = W @ np.array([1.0, 0.0])
    return 2.0 * np.imag(psi[0])

# ============================================================
# Build Perceval PIC using PS and Unitary
# ============================================================

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
# Sanity check with corrected convention
# ============================================================

u_test = 0.5
c_old = -999  # placeholder
try:
    from classical_prev import classical_qsp
    c_old = classical_qsp(phiset, u_test, L)
except:
    pass

c_new = classical_qsp_ps(phiset, u_test, L)
p_val = get_pic_output(phiset, u_test, L)

print("========== Sanity Check at u=0.5 ==========")
print(f"  Classical (PS convention) : {c_new:.6f}")
print(f"  Perceval PIC              : {p_val:.6f}")
print(f"  Difference                : {abs(c_new - p_val):.2e}")
print("  Both must match.")
print("============================================")
print()

# ============================================================
# Also try output as Re(psi[0]) in case Im is wrong
# ============================================================

def get_pic_output_re(phiset, u_val, L):
    circuit = build_qsp_pic(phiset, u_val, L)
    U = circuit.compute_unitary()
    psi_out = U @ np.array([1.0, 0.0])
    return 2.0 * np.real(psi_out[0])

def get_pic_output_z(phiset, u_val, L):
    circuit = build_qsp_pic(phiset, u_val, L)
    U = circuit.compute_unitary()
    psi_out = U @ np.array([1.0, 0.0])
    p0 = abs(psi_out[0])**2
    p1 = abs(psi_out[1])**2
    return p0 - p1

print("Also checking Re and Z variants at u=0.5:")
print(f"  2*Im(psi[0])   : {get_pic_output(phiset, u_test, L):.6f}")
print(f"  2*Re(psi[0])   : {get_pic_output_re(phiset, u_test, L):.6f}")
print(f"  p0 - p1 (Z)    : {get_pic_output_z(phiset, u_test, L):.6f}")
print(f"  Classical target: {c_new:.6f}")
print("  Which one matches classical?")
print()

# ============================================================
# Full sweep
# ============================================================

n_points = 300
u_grid = np.linspace(-1.0, 1.0, n_points)
x_grid = u_grid * np.pi

f_classical = np.zeros(n_points)
f_pic_im    = np.zeros(n_points)
f_pic_re    = np.zeros(n_points)
f_pic_z     = np.zeros(n_points)

print(f"Sweeping {n_points} points...")
for i, u in enumerate(u_grid):
    f_classical[i] = classical_qsp_ps(phiset, u, L)
    f_pic_im[i]    = get_pic_output(phiset, u, L)
    f_pic_re[i]    = get_pic_output_re(phiset, u, L)
    f_pic_z[i]     = get_pic_output_z(phiset, u, L)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{n_points} done")

print("Sweep complete.")

step_true = np.where(x_grid >= 0, 1.0, -1.0)

mse_im = np.mean((f_pic_im - f_classical)**2)
mse_re = np.mean((f_pic_re - f_classical)**2)
mse_z  = np.mean((f_pic_z  - f_classical)**2)

print("\n========== MSE vs Classical ==========")
print(f"  2*Im(psi[0]) : {mse_im:.8f}")
print(f"  2*Re(psi[0]) : {mse_re:.8f}")
print(f"  p0-p1 (Z)    : {mse_z:.8f}")
print("  Smallest MSE tells us correct output formula.")
print("=======================================")

# ============================================================
# Plot all three variants
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Photonic QSP STEP L=15 output convention check", fontsize=13)

ax = axes[0]
ax.plot(x_grid, step_true,   'k-', lw=2,   label="Target STEP")
ax.plot(x_grid, f_classical, 'r-', lw=1.5, label="Classical QSP")
ax.plot(x_grid, f_pic_im,    'b.', ms=2,   label="PIC 2*Im(psi0)")
ax.plot(x_grid, f_pic_re,    'g.', ms=2,   label="PIC 2*Re(psi0)")
ax.plot(x_grid, f_pic_z,     'm.', ms=2,   label="PIC p0-p1")
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-1.5, 1.5])
ax.set_xticks([-np.pi, 0, np.pi])
ax.set_xticklabels(["-pi", "0", "pi"], fontsize=12)
ax.set_xlabel("Input signal x", fontsize=12)
ax.set_ylabel("Function value f(x)", fontsize=12)
ax.set_title("Which output formula matches classical?", fontsize=12)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(x_grid, f_pic_im - f_classical, 'b-', lw=1, label="Im residual")
ax2.plot(x_grid, f_pic_re - f_classical, 'g-', lw=1, label="Re residual")
ax2.plot(x_grid, f_pic_z  - f_classical, 'm-', lw=1, label="Z residual")
ax2.axhline(0, color='k', lw=0.8, linestyle='--')
ax2.set_xlim([-np.pi, np.pi])
ax2.set_xticks([-np.pi, 0, np.pi])
ax2.set_xticklabels(["-pi", "0", "pi"], fontsize=12)
ax2.set_xlabel("Input signal x", fontsize=12)
ax2.set_ylabel("Residual", fontsize=12)
ax2.set_title("Residuals vs Classical", fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pic_qsp_output_convention_check.png", dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: pic_qsp_output_convention_check.png")