
#python# ============================================================
# PHOTONIC QSP ON PIC - KEY CONVENTIONS AND FINDINGS
# Date: 2026-03-20
# ============================================================
#
# GOAL:
#   Implement QSP on a Perceval PIC to simulate STEP function
#   with directly measurable output Z = p0-p1
#
# QSP CIRCUIT (paper Bu et al. 2025, Eq. 1):
#   W(x) = A(t0,p0) * prod_{j=1}^{L} [Rz(x) * A(tj,pj)]
#   A(theta,phi) = Ry(theta) * Rz(phi)
#   Signal unitary: Rz(x) where x in [-pi, pi]
#   Output: <Z> = p0-p1  (directly measurable with detectors)
#
# PHASE COMPUTATION:
#   Uses scipy gradient optimization (NOT pyqsp, NOT paddle_quantum)
#   pyqsp sym_qsp gives Im(psi[0]) output -- NOT measurable
#   paddle_quantum (paper's method) has dependency conflicts
#   Optimization minimizes MSE between Z=p0-p1 and step_surrogate
#   step_surrogate(x) = (2/pi) * arctan(100*x)  (paper Eq. B9)
#
# VERIFIED PERCEVAL MAPPINGS (confirmed by diagnostic):
#   Ry(theta) --> BS.Ry(theta)
#   Rz(phi)   --> PS(-phi/2) mode0 + PS(+phi/2) mode1
#   Rz(x)     --> PS(-x/2)   mode0 + PS(+x/2)   mode1
#
# CRITICAL: Gate order in Perceval is LEFT TO RIGHT
#   A(theta,phi) = Ry(theta) * Rz(phi)
#   In Perceval: add Rz FIRST, then Ry
#   circuit.add(0, PS(-phi/2))      # Rz first
#   circuit.add(1, PS(+phi/2))
#   circuit.add((0,1), BS.Ry(theta)) # then Ry
#
# MEASUREMENT:
#   Output Z = p0-p1 = |psi[0]|^2 - |psi[1]|^2
#   p0 = photon count at mode 0
#   p1 = photon count at mode 1
#   This is directly measurable with single photon detectors
#   NO phase-sensitive measurement needed
#
# RESULTS (L=15):
#   MSE PIC vs Classical : 2.68e-30 (machine precision zero)
#   MSE PIC vs true STEP : 0.0405
#   Comparable to paper's L=15 result (MSE ~ 0.06)
#
# FAILED APPROACHES (do not repeat):
#   pyqsp sym_qsp + Im(psi[0])  : correct shape but NOT measurable
#   pyqsp + trigonometric remap  : MSE too large (>0.46)
#   paddle_quantum               : dependency conflicts with perceval
#   BS.H for Ry                  : wrong matrix
#   BS.Rx for Ry                 : wrong matrix
#   Ry then Rz order in Perceval : wrong result (MSE=0.75)
#
# ENVIRONMENT:
#   perceval_env (conda)
#   perceval-quandela, pyqsp, scipy, numpy, matplotlib
#   paddle_quantum NOT needed
# ============================================================


import numpy as np
import matplotlib.pyplot as plt
import perceval as pcvl
import perceval.components as comp

# ============================================================
# Load optimized angles
# ============================================================

theta_opt = np.load("theta_step_opt.npy")
phi_opt   = np.load("phi_step_opt.npy")
L         = len(theta_opt) - 1

print(f"Loaded QSP angles: L={L}")

N_approx = 100

def step_surrogate(x):
    return (2.0 / np.pi) * np.arctan(N_approx * x)

def step_true(x):
    return np.where(x >= 0, 1.0, -1.0)

# ============================================================
# Verified Perceval mappings:
#   Ry(theta) --> BS.Ry(theta)
#   Rz(phi)   --> PS(-phi/2) mode0 + PS(+phi/2) mode1
#   Output    --> Z = p0-p1
# ============================================================

def Ry_mat(theta):
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s],[s, c]], dtype=complex)

def Rz_mat(phi):
    return np.array([
        [np.exp(-1j*phi/2), 0              ],
        [0,                 np.exp(1j*phi/2)]
    ], dtype=complex)

def A_mat(theta, phi):
    return Ry_mat(theta) @ Rz_mat(phi)

def classical_qsp(theta_arr, phi_arr, x_val, L):
    W = A_mat(theta_arr[0], phi_arr[0])
    for j in range(1, L + 1):
        W = A_mat(theta_arr[j], phi_arr[j]) @ Rz_mat(x_val) @ W
    psi = W @ np.array([1.0, 0.0])
    return abs(psi[0])**2 - abs(psi[1])**2

def build_qsp_pic(theta_arr, phi_arr, x_val, L):
    circuit = pcvl.Circuit(2, name=f"QSP_STEP_L{L}")

    # A(theta_0, phi_0) = Ry(theta_0) * Rz(phi_0)
    # Perceval applies LEFT to RIGHT so add Rz first then Ry
    circuit.add(0,      comp.PS(float(-phi_arr[0] / 2)))   # Rz first
    circuit.add(1,      comp.PS(float( phi_arr[0] / 2)))
    circuit.add((0, 1), comp.BS.Ry(theta=float(theta_arr[0])))  # then Ry

    for j in range(1, L + 1):
        # Rz(x): signal unitary
        circuit.add(0, comp.PS(float(-x_val / 2)))
        circuit.add(1, comp.PS(float( x_val / 2)))
        # A(theta_j, phi_j) = Ry * Rz: add Rz first then Ry
        circuit.add(0,      comp.PS(float(-phi_arr[j] / 2)))
        circuit.add(1,      comp.PS(float( phi_arr[j] / 2)))
        circuit.add((0, 1), comp.BS.Ry(theta=float(theta_arr[j])))

    return circuit

def get_pic_z(theta_arr, phi_arr, x_val, L):
    circuit = build_qsp_pic(theta_arr, phi_arr, x_val, L)
    U       = np.array(circuit.compute_unitary())
    psi     = U @ np.array([1.0, 0.0])
    return abs(psi[0])**2 - abs(psi[1])**2

# ============================================================
# Sanity check
# ============================================================

print("\n========== Sanity Check ==========")
for x_test in [-1.0, -0.5, 0.0, 0.5, 1.0]:
    c_val = classical_qsp(theta_opt, phi_opt, x_test, L)
    p_val = get_pic_z(theta_opt, phi_opt, x_test, L)
    f_ref = step_surrogate(x_test)
    print(f"  x={x_test:+.1f}: surrogate={f_ref:.4f}  "
          f"classical={c_val:.4f}  PIC={p_val:.4f}  "
          f"diff={abs(c_val-p_val):.2e}")
print("===================================")

# ============================================================
# Full sweep
# ============================================================

theta_grid  = np.linspace(-np.pi, np.pi, 300)
f_classical = np.zeros(len(theta_grid))
f_pic       = np.zeros(len(theta_grid))
f_surrogate = np.array([step_surrogate(x) for x in theta_grid])
f_true      = step_true(theta_grid)

print(f"\nSweeping {len(theta_grid)} points...")
for i, x in enumerate(theta_grid):
    f_classical[i] = classical_qsp(theta_opt, phi_opt, x, L)
    f_pic[i]       = get_pic_z(theta_opt, phi_opt, x, L)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(theta_grid)} done")

print("Sweep complete.")

# ============================================================
# MSE report
# ============================================================

mse_pic_classic = np.mean((f_pic - f_classical) ** 2)
mse_pic_surr    = np.mean((f_pic - f_surrogate) ** 2)
mse_pic_true    = np.mean((f_pic - f_true)      ** 2)

print("\n========== MSE Report ==========")
print(f"  MSE PIC vs Classical : {mse_pic_classic:.2e}  must be ~0")
print(f"  MSE PIC vs surrogate : {mse_pic_surr:.4f}")
print(f"  MSE PIC vs true STEP : {mse_pic_true:.4f}")
print(f"  Output Z=p0-p1 is directly measurable with detectors")
print("=================================")

# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    f"Photonic QSP on PIC  STEP L={L}  Z=p0-p1  measurable",
    fontsize=13
)

ax = axes[0]
ax.plot(theta_grid, f_true,      'k-',  lw=2,   label="True STEP")
ax.plot(theta_grid, f_surrogate, 'g--', lw=2,   label="arctan surrogate")
ax.plot(theta_grid, f_classical, 'r-',  lw=1.5, label="Classical Z=p0-p1")
ax.plot(theta_grid, f_pic,       'b.',  ms=3,
        label=f"PIC Z=p0-p1  MSE={mse_pic_true:.4f}")
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-1.3, 1.3])
ax.set_xticks([-np.pi, 0, np.pi])
ax.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"], fontsize=12)
ax.set_xlabel(r"$x$", fontsize=12)
ax.set_ylabel("Function value", fontsize=12)
ax.set_title(f"L={L}  blue must sit on red", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
diff = f_pic - f_classical
ax2.plot(theta_grid, diff, 'purple', lw=1.5,
         label=f"PIC minus Classical  MSE={mse_pic_classic:.2e}")
ax2.axhline(0, color='k', lw=0.8, linestyle='--')
ax2.fill_between(theta_grid, diff, alpha=0.2, color='purple')
ax2.set_xlim([-np.pi, np.pi])
ax2.set_xticks([-np.pi, 0, np.pi])
ax2.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"], fontsize=12)
ax2.set_xlabel(r"$x$", fontsize=12)
ax2.set_ylabel("Residual", fontsize=12)
ax2.set_title("PIC vs Classical  must be ~0", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pic_qsp_step_measurable_FINAL.png", dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: pic_qsp_step_measurable_FINAL.png")

# ============================================================
# PIC resource count
# ============================================================

circ_full = build_qsp_pic(theta_opt, phi_opt, 0.5, L)
n_ps = sum(1 for _, c in circ_full._components if isinstance(c, comp.PS))
n_bs = sum(1 for _, c in circ_full._components if isinstance(c, comp.BS))

print("\n========== PIC Resource Count for Quandela ==========")
print(f"  QSP layers L        : {L}")
print(f"  Phase shifters PS   : {n_ps}")
print(f"  Beam splitters BS   : {n_bs}")
print(f"  Total components    : {n_ps + n_bs}")
print(f"  Waveguide modes     : 2  dual-rail qubit")
print(f"  Output measurement  : Z = p0-p1 photon counting")
print("=====================================================")

# ============================================================
# Draw circuit for L=3 - grab MplotCanvas directly
# ============================================================

import matplotlib
import matplotlib.pyplot as plt

print("\nDrawing circuit for L=3...")
circ_draw = build_qsp_pic(theta_opt[:4], phi_opt[:4], 0.5, L=3) # theta_opt[:4] — slices only the first 4 elements of the array (indices 0,1,2,3), 
# which gives exactly L+1=4 angles for L=3 layers
#L=3 — explicitly tells build_qsp_pic to build only 3 layers
# To draw the full L=15 circuit, change this line to:
#circ_draw = build_qsp_pic(theta_opt, phi_opt, 0.5, L=15)
# The rule is: if you want L layers, pass theta_opt[:L+1] and phi_opt[:L+1]



# Get the canvas object directly
canvas = pcvl.pdisplay(circ_draw, output_format=pcvl.Format.MPLOT)

print(f"Canvas type : {type(canvas)}")
print(f"Canvas dir  : {[x for x in dir(canvas) if not x.startswith('_')]}")

# Try to get the figure from the canvas
if hasattr(canvas, 'fig'):
    fig = canvas.fig
    print(f"Found canvas.fig: {fig}")
elif hasattr(canvas, 'figure'):
    fig = canvas.figure
    print(f"Found canvas.figure: {fig}")
elif hasattr(canvas, 'get_figure'):
    fig = canvas.get_figure()
    print(f"Found canvas.get_figure(): {fig}")
else:
    print("No fig attribute found — listing all attributes:")
    for attr in dir(canvas):
        if not attr.startswith('_'):
            try:
                val = getattr(canvas, attr)
                print(f"  {attr} = {val}")
            except:
                print(f"  {attr} = <error>")