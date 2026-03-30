# ============================================================
# QSP STEP FUNCTION - LOCAL SLOS SIMULATION VERSION
# ============================================================
#
# GOAL:
#   Simulate the QSP STEP function circuit locally using
#   Perceval's SLOS backend, which fires photons one by one
#   and counts detections -- mimicking a real experiment.
#
# DIFFERENCE FROM CIRCUIT CODE:
#   Circuit code: uses compute_unitary() -- exact matrix math
#   This code:    uses SLOS sampler -- statistical photon counting
#
# NOTE ON N_SHOTS:
#   N_SHOTS = 100000 is sufficient. Do not increase it.
#   MSE SLOS vs Perceval analytic = ~0 --> sampling noise negligible
#   Increasing N_SHOTS will NOT improve STEP approximation quality.
#   To improve STEP approximation quality:
#     --> increase L (more QSP layers), NOT N_SHOTS
#
# NOTE ON X POINTS:
#   x_values = 100 points used for local SLOS simulation
#   --> gives smooth curve, fast locally, free to run
#   For real QPU run, reduce to 30 points to save QPU credits
#   MSE vs true STEP depends on number of x points:
#     30 points  --> MSE ~ 0.0775
#     100 points --> MSE ~ 0.0476
#   This is NOT because the circuit improved -- it is because
#   more x points gives a more representative average over [-pi, pi]
#   The true circuit approximation error is ~0.0405 (from circuit code)
#
# NAMING CONVENTIONS:
#   f_classical          : pure numpy matrix math, no Perceval (circuit code)
#   f_perceval_analytic  : Perceval compute_unitary(), exact, no sampling
#   z_slos               : Perceval SLOS sampler, statistical photon counts
#   z_experimental       : reserved for future real QPU hardware results
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import perceval as pcvl
import perceval.components as comp
from perceval.algorithm import Sampler

# ============================================================
# Step 1: Load optimized QSP angles
# theta_opt : 16 Ry rotation angles  (fixed, never change with x)
# phi_opt   : 16 Rz rotation angles  (fixed, never change with x)
# These were found by scipy L-BFGS-B optimizer
# ============================================================

#theta_opt = np.load("theta_step_opt.npy")
#phi_opt   = np.load("phi_step_opt.npy")

theta_nlft = np.load("theta_step_nlft.npy")
phi_nlft   = np.load("phi_step_nlft.npy")

L         = len(theta_nlft) - 1
print(f"Loaded QSP angles: L={L}")

N_approx = 100

def step_surrogate(x):
    """Smooth arctan approximation of STEP function (paper Eq. B9)."""
    return (2.0 / np.pi) * np.arctan(N_approx * x)

def step_true(x):
    """Ideal STEP function: -1 for x<0, +1 for x>=0."""
    return np.where(x >= 0, 1.0, -1.0)

def build_qsp_pic(theta_arr, phi_arr, x_val, L):
    """
    Build the Perceval QSP circuit for a specific x value.
    Structure per layer:
      Rz(phi_j) : PS(-phi_j/2) mode0 + PS(+phi_j/2) mode1  -- fixed
      Ry(theta_j): BS.Ry(theta_j)                           -- fixed
      Rz(x)     : PS(-x/2) mode0 + PS(+x/2) mode1          -- changes with x
    Perceval applies gates LEFT TO RIGHT so Rz is added before Ry.
    """
    circuit = pcvl.Circuit(2, name=f"QSP_STEP_L{L}")

    # Initial block A(theta_0, phi_0) = Ry(theta_0) * Rz(phi_0)
    circuit.add(0,      comp.PS(float(-phi_arr[0] / 2)))   # Rz first
    circuit.add(1,      comp.PS(float( phi_arr[0] / 2)))
    circuit.add((0, 1), comp.BS.Ry(theta=float(theta_arr[0])))  # then Ry

    for j in range(1, L + 1):
        # Signal unitary Rz(x) -- only part that changes with x
        circuit.add(0, comp.PS(float(-x_val / 2)))
        circuit.add(1, comp.PS(float( x_val / 2)))
        # Fixed block A(theta_j, phi_j) = Ry(theta_j) * Rz(phi_j)
        circuit.add(0,      comp.PS(float(-phi_arr[j] / 2)))
        circuit.add(1,      comp.PS(float( phi_arr[j] / 2)))
        circuit.add((0, 1), comp.BS.Ry(theta=float(theta_arr[j])))

    return circuit

# ============================================================
# Step 2: Experiment settings
#
# N_SHOTS : number of photons fired per x value
#           100000 is sufficient -- see note at top of file
# x_values: x points to sweep over [-pi, pi]
#           30 points is enough to see the STEP shape clearly
# ============================================================

N_SHOTS  = 100000
x_values = np.linspace(-np.pi, np.pi, 100)

# ============================================================
# Step 3: Run SLOS local simulation
#
# For each x value:
#   1. Build circuit with that x baked into signal PS phases
#   2. Run SLOS sampler -- fires N_SHOTS photons through circuit
#   3. Count detections in mode 0 and mode 1
#   4. Compute p0 = count_mode0 / total
#              p1 = count_mode1 / total
#              Z  = p0 - p1
#
# SLOS = Scalable Linear Optical Simulator
# Local backend -- no token, no internet, runs instantly
# Models probabilistic photon detection like real hardware
# ============================================================

z_slos  = np.zeros(len(x_values))   # Z = p0-p1 from SLOS sampling
p0_slos = np.zeros(len(x_values))   # p0 from SLOS sampling
p1_slos = np.zeros(len(x_values))   # p1 from SLOS sampling

print(f"\nStarting SLOS sweep:")
print(f"  {len(x_values)} x values, {N_SHOTS} shots each")
print(f"  Total photons fired: {len(x_values) * N_SHOTS:,}")
print("=" * 55)

for i, x_val in enumerate(x_values):

    # Build circuit for this x value
    circuit = build_qsp_pic(theta_nlft, phi_nlft, x_val, L)

    # Set up local Processor with SLOS backend
    # No token or internet connection needed
    local_proc = pcvl.Processor("SLOS", circuit)

    # Input state: 1 photon in mode 0, 0 photons in mode 1
    local_proc.with_input(pcvl.BasicState([1, 0]))

    # Filter: keep only events with at least 1 detected photon
    local_proc.min_detected_photons_filter(1)

    # Run sampler -- fires N_SHOTS photons and returns counts
    sampler = Sampler(local_proc)
    results = sampler.sample_count(N_SHOTS)

    # Convert BSCount object to regular dict for easy access
    counts  = dict(results['results'])

    # Extract photon counts per mode
    # BasicState([1,0]) = photon detected in mode 0
    # BasicState([0,1]) = photon detected in mode 1
    count_mode0 = counts.get(pcvl.BasicState([1, 0]), 0)
    count_mode1 = counts.get(pcvl.BasicState([0, 1]), 0)
    total       = count_mode0 + count_mode1

    # Compute probabilities and Z = p0 - p1
    if total > 0:
        p0 = count_mode0 / total
        p1 = count_mode1 / total
        z  = p0 - p1
    else:
        p0, p1, z = 0.0, 0.0, 0.0
        print(f"  WARNING: no counts detected at x={x_val:.3f}")

    z_slos[i]  = z
    p0_slos[i] = p0
    p1_slos[i] = p1

    print(f"  [{i+1:2d}/{len(x_values)}] x={x_val:+.3f}  "
          f"mode0={count_mode0}  mode1={count_mode1}  "
          f"p0={p0:.3f}  p1={p1:.3f}  Z={z:+.3f}")

print("=" * 55)
print("Sweep complete.")

# ============================================================
# Step 4: Save SLOS results to disk
# Saved separately from future QPU results to avoid confusion
# ============================================================

np.save("x_values_slos.npy", x_values)
np.save("z_slos.npy",        z_slos)
np.save("p0_slos.npy",       p0_slos)
np.save("p1_slos.npy",       p1_slos)
print("\nSLOS results saved to disk.")

# ============================================================
# Step 5: Compute reference curves for comparison
#
# f_perceval_analytic : Perceval circuit + compute_unitary()
#                       exact Z via linear algebra, no sampling
#                       used to verify SLOS sampling is accurate
#
# f_surrogate         : arctan target we optimized against
# f_true              : ideal +-1 STEP function
# ============================================================

# Fine grid for smooth reference curves in plot
x_fine      = np.linspace(-np.pi, np.pi, 300)
f_surrogate = np.array([step_surrogate(x) for x in x_fine])
f_true      = step_true(x_fine)

# Perceval analytic Z at same x points as SLOS sweep
# Uses compute_unitary() -- exact linear algebra, no randomness
f_perceval_analytic = np.zeros(len(x_values))
for i, x_val in enumerate(x_values):
    circuit = build_qsp_pic(theta_nlft, phi_nlft, x_val, L)
    U       = np.array(circuit.compute_unitary())
    psi     = U @ np.array([1.0, 0.0])
    f_perceval_analytic[i] = abs(psi[0])**2 - abs(psi[1])**2

# ============================================================
# Step 6: MSE report
#
# mse_slos_vs_analytic : SLOS vs Perceval analytic
#                        measures pure sampling noise
#                        should be ~0 with N_SHOTS=100000
#
# mse_slos_vs_step     : SLOS vs true STEP
#                        overall approximation quality
#                        = QSP approximation error + sampling noise
# ============================================================

mse_slos_vs_analytic = np.mean((z_slos - f_perceval_analytic)**2)
mse_slos_vs_step     = np.mean((z_slos - step_true(x_values))**2)

print(f"\n========== MSE Report ==========")
print(f"  MSE SLOS vs Perceval analytic : {mse_slos_vs_analytic:.4f}  must be ~0")
print(f"  MSE SLOS vs true STEP         : {mse_slos_vs_step:.4f}")
print(f"=================================")

# ============================================================
# Step 7: Plot results
#
# Left panel:  SLOS sampled Z vs Perceval analytic vs true STEP
#              blue dots should sit on red line
# Right panel: residual SLOS minus Perceval analytic
#              shows sampling noise -- should be close to zero
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    f"QSP STEP Function - SLOS Local Simulation  L={L}  N_shots={N_SHOTS}",
    fontsize=13
)

# Left panel
ax = axes[0]
ax.plot(x_fine,   f_true,              'k-',  lw=2,
        label="True STEP")
ax.plot(x_fine,   f_surrogate,         'g--', lw=1.5,
        label="arctan surrogate")
ax.plot(x_values, f_perceval_analytic, 'r-',  lw=1.5,
        label="Perceval analytic Z=p0-p1")
ax.plot(x_values, z_slos,              'b.',  ms=8,
        label=f"SLOS sampled Z=p0-p1  "
              f"MSE vs true STEP={mse_slos_vs_step:.4f}")
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-1.3, 1.3])
ax.set_xticks([-np.pi, 0, np.pi])
ax.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"], fontsize=12)
ax.set_xlabel(r"$x$", fontsize=12)
ax.set_ylabel("Z = p0 - p1", fontsize=12)
ax.set_title(f"L={L}  blue dots must sit on red line", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right panel: residual between SLOS and Perceval analytic
# Shows pure sampling noise -- should be close to zero
ax2 = axes[1]
diff = z_slos - f_perceval_analytic
ax2.plot(x_values, diff, 'purple', lw=1.5, marker='.', ms=6,
         label=f"SLOS minus Perceval analytic  "
               f"MSE vs Perceval analytic={mse_slos_vs_analytic:.4f}")
ax2.axhline(0, color='k', lw=0.8, linestyle='--')
ax2.fill_between(x_values, diff, alpha=0.2, color='purple')
ax2.set_xlim([-np.pi, np.pi])
ax2.set_xticks([-np.pi, 0, np.pi])
ax2.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"], fontsize=12)
ax2.set_xlabel(r"$x$", fontsize=12)
ax2.set_ylabel("Residual", fontsize=12)
ax2.set_title("SLOS vs Perceval analytic residual\n"
              "(shows sampling noise -- should be ~0)", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("qsp_step_slos_simulation.png", dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: qsp_step_slos_simulation.png")