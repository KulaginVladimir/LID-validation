import festim as F
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

def TDS(n1, E_p1, n2, E_p2, n3, E_p3):
    """Runs the simulation with parameters p that represent:

    Args:
        n1 (float): concentration of trap 1, at. fr.
        E_p1 (float): detrapping barrier from trap 1, eV
        n2 (float): concentration of trap 2, at. fr.
        E_p2 (float): detrapping barrier from trap 2, eV

    Returns:
        F.DerivedQuantities: the derived quantities of the simulation
    """
    w_atom_density = 6.31e28  # atom/m3
    D0_W = 1.93e-7/np.sqrt(2)
    Ed_W = 0.2

    # Define Simulation object
    model = F.Simulation()

    # Define a simple mesh
    vertices = np.concatenate([
        np.linspace(0, 1e-6, num=500)])
    
    model.mesh = F.MeshFromVertices(vertices)

    # Define material properties
    tungsten = F.Material(
        id=1,
        D_0=D0_W,
        E_D=Ed_W,
        S_0=1.87e24,
        E_S=1.04,
        borders=[0, 1e-6]
    )

    model.materials = F.Materials([tungsten])

    # Define traps
    trap_1 = F.Trap(
        k_0=D0_W / (1.1e-10**2 * 6 * w_atom_density),
        E_k=Ed_W,
        p_0=1e13,
        E_p=E_p1,
        density=n1 * w_atom_density,
        materials=tungsten,
    )

    trap_2 = F.Trap(
        k_0=D0_W / (1.1e-10**2 * 6 * w_atom_density),
        E_k=Ed_W,
        p_0=1e13,
        E_p=E_p2,
        density=n2 * w_atom_density,
        materials=tungsten,
    )

    trap_3 = F.Trap(
        k_0=D0_W / (1.1e-10**2 * 6 * w_atom_density),
        E_k=Ed_W,
        p_0=1e13,
        E_p=E_p3,
        density=n3 * w_atom_density,
        materials=tungsten,
    )

    model.traps = [trap_1, trap_2, trap_3]

    # Set initial conditions
    model.initial_conditions = [
        F.InitialCondition(field="1", value=n1 * w_atom_density),
        F.InitialCondition(field="2", value=n2 * w_atom_density),
        F.InitialCondition(field="3", value=n3 * w_atom_density),
    ]

    # Set boundary conditions
    model.boundary_conditions = [
        F.DirichletBC(surfaces=[1], value=0, field=0)
    ]

    # Define the material temperature evolution
    ramp = 0.5  # K/s
    model.T = 300 + ramp * F.t

    # Define the simulation settings
    model.dt = F.Stepsize(
        initial_value=0.1,
        stepsize_change_ratio=1.1,
        max_stepsize=10,
        dt_min=1e-6,
    )

    model.settings = F.Settings(
        absolute_tolerance=1e5,
        relative_tolerance=1e-10,
        final_time=1500,
        maximum_iterations=50,
    )

    # Define the exports
    derived_quantities = F.DerivedQuantities(
        [
            F.SurfaceFlux(surface=1, field="solute"),
            F.SurfaceFlux(surface=2, field="solute"),
            F.AverageVolume(field="T", volume=1),
            F.TotalVolume(field="1", volume=1),
        ],
        show_units=True
    )

    XDMF = [
        F.XDMFExport(field="retention", filename="./res/ret.xdmf", checkpoint=False,),
        F.XDMFExport(field="solute", filename="./res/mob.xdmf", checkpoint=False,),
        F.XDMFExport(field="1", filename="./res/1.xdmf", checkpoint=False,),
    ]
    model.exports = [derived_quantities] + XDMF
    model.initialise()
    model.run()

    return derived_quantities
    

ref = np.genfromtxt("./TDS_S160724.csv", delimiter=",", skip_header=1)

initial_guess = [0.01836434, 1.26448174, 0.02948491, 1.08813931, 0.01870261, 1.50186434]

# Process the obtained results
res = TDS(*initial_guess)

T = res.filter(fields="T").data

# Visualise
tr1 = np.array(res.filter(fields="1").data)
time = np.array(res.t)

dtr1 = -np.diff(tr1)/np.diff(time)

plt.plot(T, -np.array(res.filter(fields="solute", surfaces=1).data), label="left")
plt.plot(T, -np.array(res.filter(fields="solute", surfaces=2).data), label="right")

plt.plot(T[1:], dtr1, label="1", ls='dashed')

plt.plot(ref[:, 0], ref[:, 1], linewidth=2, label="Reference")
plt.legend()

plt.ylabel(r"Desorption flux (m$^{-2}$ s$^{-1}$)")
plt.xlabel(r"Temperature (K)")
plt.legend()
plt.show()