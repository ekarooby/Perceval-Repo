# ============================================================
# DIAGNOSTIC: First check what BS() default actually is,
# then find if MZI can match BS.Rx(theta) at all
# ============================================================

import numpy as np
import perceval as pcvl
import perceval.components as comp

def get_U(circuit):
    return np.array(circuit.compute_unitary())

# Step 1: Check what BS() default actually is
print("=== BS() default matrix ===")
c = pcvl.Circuit(2)
c.add((0,1), comp.BS())
U_default = get_U(c)
print(np.round(U_default, 6))

print("\n=== BS.Rx(pi/2) matrix ===")
c2 = pcvl.Circuit(2)
c2.add((0,1), comp.BS.Rx(theta=np.pi/2))
U_rx = get_U(c2)
print(np.round(U_rx, 6))

print("\n=== Are BS() and BS.Rx(pi/2) the same? ===")
print(f"  diff = {np.linalg.norm(U_default - U_rx):.2e}")

# Step 2: Check MZI with PS on mode 0 vs mode 1
print("\n=== MZI with PS on mode 0 ===")
for phi in [0, np.pi/2, np.pi, 3*np.pi/2]:
    c = pcvl.Circuit(2)
    c.add((0,1), comp.BS())
    c.add(0,     comp.PS(float(phi)))
    c.add((0,1), comp.BS())
    U = get_U(c)
    print(f"  phi={phi/np.pi:.2f}*pi: U=\n{np.round(U,4)}")

print("\n=== MZI with PS on mode 1 ===")
for phi in [0, np.pi/2, np.pi, 3*np.pi/2]:
    c = pcvl.Circuit(2)
    c.add((0,1), comp.BS())
    c.add(1,     comp.PS(float(phi)))
    c.add((0,1), comp.BS())
    U = get_U(c)
    print(f"  phi={phi/np.pi:.2f}*pi: U=\n{np.round(U,4)}")

# Step 3: What does BS.Rx(theta) look like for several theta?
print("\n=== BS.Rx(theta) for several theta ===")
for theta in [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]:
    c = pcvl.Circuit(2)
    c.add((0,1), comp.BS.Rx(theta=float(theta)))
    U = get_U(c)
    print(f"  theta={theta/np.pi:.2f}*pi: U=\n{np.round(U,4)}")

# Step 4: Analytically compute MZI matrix
# MZI(phi, mode1) = BS50 @ diag(1, e^{i*phi}) @ BS50
# where BS50 = BS.Rx(pi/2) = (1/sqrt(2))*[[1,i],[i,1]]
print("\n=== Analytical MZI(phi) = BS50 @ diag(1,e^{iphi}) @ BS50 ===")
def BS50():
    return (1/np.sqrt(2)) * np.array([[1, 1j],[1j, 1]])

for phi in [0, np.pi/4, np.pi/2, np.pi]:
    P = np.diag([1, np.exp(1j*phi)])
    U_analytical = BS50() @ P @ BS50()
    print(f"  phi={phi/np.pi:.2f}*pi: U=\n{np.round(U_analytical,4)}")
    # Compare with Perceval MZI
    c = pcvl.Circuit(2)
    c.add((0,1), comp.BS())
    c.add(1,     comp.PS(float(phi)))
    c.add((0,1), comp.BS())
    U_perceval = get_U(c)
    print(f"  Perceval MZI: U=\n{np.round(U_perceval,4)}")
    print(f"  diff analytical vs perceval: {np.linalg.norm(U_analytical-U_perceval):.2e}\n")