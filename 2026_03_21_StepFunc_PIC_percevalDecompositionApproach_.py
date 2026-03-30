import numpy as np
import perceval as pcvl
import perceval.components as comp

def get_U(circuit):
    return np.array(circuit.compute_unitary())

def diff_up_to_global_phase(A, B):
    """Correct global phase distance."""
    phases = np.linspace(0, 2*np.pi, 360)
    return np.min([np.linalg.norm(A - np.exp(1j*p)*B) for p in phases])

BS50 = (1/np.sqrt(2)) * np.array([[1, 1j],[1j, 1]])

# ============================================================
# Step 1: Print MZI matrix for several phi and compare
# with ALL three BS conventions to find which one matches
# ============================================================
print("=== For each phi, find which BS convention MZI matches ===\n")

for phi in [np.pi/4, np.pi/2, np.pi, 3*np.pi/4]:
    # MZI matrix
    P    = np.diag([1.0, np.exp(1j*phi)])
    Umzi = BS50 @ P @ BS50

    print(f"phi = {phi/np.pi:.3f}*pi")
    print(f"  MZI matrix: {np.round(Umzi, 4)}")

    for conv, theta in [('Rx', phi), ('Ry', phi), ('H', phi)]:
        c = pcvl.Circuit(2)
        getattr(comp.BS, conv)(theta=float(theta))
        c.add((0,1), getattr(comp.BS, conv)(theta=float(theta)))
        U = get_U(c)
        d = diff_up_to_global_phase(Umzi, U)
        print(f"  vs BS.{conv}(theta=phi={phi/np.pi:.3f}pi): diff={d:.2e}")

    # Also try BS.Ry with different theta mapping
    for theta_test in [phi, np.pi-phi, np.pi+phi, 2*np.pi-phi]:
        c = pcvl.Circuit(2)
        c.add((0,1), comp.BS.Ry(theta=float(theta_test)))
        U = get_U(c)
        d = diff_up_to_global_phase(Umzi, U)
        if d < 0.01:
            print(f"  MATCH: BS.Ry(theta={theta_test/np.pi:.4f}pi) diff={d:.2e}")
    print()

# ============================================================
# Step 2: Brute force - for MZI(phi=pi/2), find matching BS
# ============================================================
print("=== Brute force: MZI(phi=pi/2) vs BS.Rx/Ry/H at all theta ===\n")
phi_test = np.pi/2
P    = np.diag([1.0, np.exp(1j*phi_test)])
Umzi = BS50 @ P @ BS50
print(f"MZI(phi=pi/2) = {np.round(Umzi,4)}")

best = {'Rx': (999,0), 'Ry': (999,0), 'H': (999,0)}
for theta in np.linspace(0, 2*np.pi, 10000):
    for conv in ['Rx', 'Ry', 'H']:
        c = pcvl.Circuit(2)
        c.add((0,1), getattr(comp.BS, conv)(theta=float(theta)))
        U = get_U(c)
        d = diff_up_to_global_phase(Umzi, U)
        if d < best[conv][0]:
            best[conv] = (d, theta)

for conv, (d, theta) in best.items():
    print(f"  Best BS.{conv}(theta={theta/np.pi:.4f}pi): diff={d:.2e}")

# ============================================================
# Step 3: Direct answer - what is MZI(phi) equivalent to?
# ============================================================
print("\n=== Direct structure analysis ===")
print("MZI(phi) = BS50 @ diag(1, e^{iphi}) @ BS50")
print("         = e^{i*phi/2} * BS50 @ diag(e^{-iphi/2}, e^{+iphi/2}) @ BS50")
print("         = e^{i*phi/2} * BS50 @ Rz(phi) @ BS50")
print()
print("So MZI(phi) = e^{i*phi/2} * (BS50 @ Rz(phi) @ BS50)")
print()
print("Checking if BS50 @ Rz(phi) @ BS50 = BS.Rx(phi):")

def Rz_mat(phi):
    return np.array([[np.exp(-1j*phi/2), 0],
                     [0, np.exp(1j*phi/2)]], dtype=complex)

for phi in [np.pi/4, np.pi/2, np.pi]:
    sandwich = BS50 @ Rz_mat(phi) @ BS50
    c = pcvl.Circuit(2)
    c.add((0,1), comp.BS.Rx(theta=float(phi)))
    U_rx = get_U(c)
    d = diff_up_to_global_phase(sandwich, U_rx)
    print(f"  phi={phi/np.pi:.3f}pi: BS50@Rz(phi)@BS50 vs BS.Rx(phi): diff={d:.2e}  "
          f"{'MATCH' if d<1e-6 else 'no match'}")
    print(f"    BS50@Rz@BS50 = {np.round(sandwich,4)}")
    print(f"    BS.Rx        = {np.round(U_rx,4)}")