import festim as F
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sympy as sp


def info(i, p):
    """
    Print information during the fitting procedure
    """
    print("-" * 40)
    print(f"i = {i}")
    print("New simulation.")
    print(f"Point is: {p}")


def TDS(n1, E_p1, n2, E_p2, n3, E_p3, n4, E_p4, S0, E_S):
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
    D0_W = 4.1e-7 / np.sqrt(2)
    Ed_W = 0.39
    D0_Cu = 6.6e-7 / np.sqrt(2)
    Ed_Cu = 0.39

    # Define Simulation object
    model = F.Simulation()

    # Define a simple mesh
    vertices = np.concatenate(
        [np.linspace(0, 1e-6, num=400), np.linspace(1e-6, 3e-3 + 1e-6, num=500)]
    )

    model.mesh = F.MeshFromVertices(vertices)

    # Define material properties
    tungsten = F.Material(id=1, D_0=D0_W, E_D=Ed_W, S_0=S0, E_S=E_S, borders=[0, 1e-6])
    copper = F.Material(
        id=2, S_0=1, E_S=0, D_0=D0_Cu, E_D=Ed_Cu, borders=[1e-6, 3e-3 + 1e-6]
    )

    model.materials = [tungsten, copper]

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

    trap_4 = F.Trap(
        k_0=D0_W / (1.1e-10**2 * 6 * w_atom_density),
        E_k=Ed_W,
        p_0=1e13,
        E_p=E_p4,
        density=n4 * w_atom_density,
        materials=tungsten,
    )

    model.traps = [trap_1, trap_2, trap_3, trap_4]

    # Set initial conditions
    model.initial_conditions = [
        F.InitialCondition(
            field="1",
            value=n1 * w_atom_density * sp.Piecewise((1, F.x < 1e-6), (0, True)),
        ),
        F.InitialCondition(
            field="2",
            value=n2 * w_atom_density * sp.Piecewise((1, F.x < 1e-6), (0, True)),
        ),
        F.InitialCondition(
            field="3",
            value=n3 * w_atom_density * sp.Piecewise((1, F.x < 1e-6), (0, True)),
        ),
        F.InitialCondition(
            field="4",
            value=n4 * w_atom_density * sp.Piecewise((1, F.x < 1e-6), (0, True)),
        ),
    ]

    # Set boundary conditions
    model.boundary_conditions = [F.DirichletBC(surfaces=[1, 2], value=0, field=0)]

    # Define the material temperature evolution
    ramp = 0.5  # K/s
    model.T = F.Temperature(value=300 + ramp * (F.t))

    # Define the simulation settings
    model.dt = F.Stepsize(
        initial_value=0.1,
        stepsize_change_ratio=1.1,
        max_stepsize=10,
        dt_min=1e-6,
    )

    model.settings = F.Settings(
        absolute_tolerance=1e10,
        relative_tolerance=1e-10,
        final_time=1600,
        maximum_iterations=50,
        chemical_pot=True,
        traps_element_type="DG",
    )

    # Define the exports
    derived_quantities = F.DerivedQuantities(
        [
            F.SurfaceFlux(surface=1, field="solute"),
            F.SurfaceFlux(surface=2, field="solute"),
            F.AverageVolume(field="T", volume=1),
            F.TotalVolume(field="1", volume=1),
            F.TotalVolume(field="2", volume=1),
            F.TotalVolume(field="3", volume=1),
            F.TotalVolume(field="4", volume=1),
        ],
        show_units=True,
    )

    model.exports = [derived_quantities]
    model.initialise()
    model.run()

    return derived_quantities


def error_function(prm):
    """
    Compute average absolute error between simulation and reference
    """
    global i

    i += 1
    info(i, prm)

    # Filter the results if a negative value is found
    if any([e < 0 for e in prm]):
        return 1e30

    # Get the simulation result
    res = TDS(*prm)

    T = np.array(res.filter(fields="T").data)
    flux = -np.array(res.filter(fields="solute", surfaces=1).data) - np.array(
        res.filter(fields="solute", surfaces=2).data
    )

    interp_tds = interp1d(T, flux, fill_value="extrapolate")

    # Compute the mean absolute error between sim and ref
    err = np.abs(interp_tds(ref[:, 0]) - ref[:, 1]).mean()

    print(f"Average absolute error is : {err:.2e}")
    return err


ref = np.genfromtxt("./TDS_S160724.csv", delimiter=",", skip_header=1)

i = 0  # initialise counter

# Set the tolerances
fatol = 1e18
xatol = 1e-5

initial_guess = [
    0.00473236,
    1.63076048,
    0.01953951,
    1.05599858,
    0.0079067,
    1.21484961,
    0.00779637,
    1.43648592,
    0.01537556,
    -0.01013876,
]

# Minimise the error function


# Process the obtained results
res = TDS(*initial_guess)

T = res.filter(fields="T").data

tr1 = np.array(res.filter(fields="1").data)
tr2 = np.array(res.filter(fields="2").data)
tr3 = np.array(res.filter(fields="3").data)
tr4 = np.array(res.filter(fields="4").data)
time = np.array(res.t)

dtr1 = -np.diff(tr1) / np.diff(time)
dtr2 = -np.diff(tr2) / np.diff(time)
dtr3 = -np.diff(tr3) / np.diff(time)
dtr4 = -np.diff(tr4) / np.diff(time)

plt.scatter(
    ref[:, 0],
    ref[:, 1],
    marker="o",
    s=10,
    alpha=0.2,
    label="Эксперимент",
    color="tab:blue",
)
plt.plot(
    T,
    -np.array(res.filter(fields="solute", surfaces=1).data),
    label="FESTIM",
    color="tab:red",
    lw=2.5,
)
# plt.plot(T, -np.array(res.filter(fields="solute", surfaces=2).data), label="right")

plt.plot(T[1:], dtr1, label="Ловушка 1", ls="dashed", lw=1)
plt.fill_between(T[1:], np.zeros_like(dtr1), dtr1, alpha=0.15, color="grey")
(l1,) = plt.plot(T[1:], dtr2, label="Ловушка 2", ls="dashed", lw=1)
plt.fill_between(T[1:], np.zeros_like(dtr1), dtr2, alpha=0.15, color="grey")
plt.plot(T[1:], dtr3, label="Ловушка 3", ls="dashed", lw=1)
plt.fill_between(T[1:], np.zeros_like(dtr1), dtr3, alpha=0.15, color="grey")
plt.plot(T[1:], dtr4, label="Ловушка 4", ls="dashed", lw=1)
plt.fill_between(T[1:], np.zeros_like(dtr1), dtr4, alpha=0.15, color="grey")


plt.ylabel(r"Поток D, м$^{-2}$ м$^{-1}$")
plt.xlabel(r"Температура, K")

plt.xlim(300, 1000)
plt.xticks([i for i in range(300, 1100, 100)])
plt.ylim(0, 6e18)
plt.legend()

plt.savefig("TDS_sol.png", dpi=500, bbox_inches=0, pad_inches=0.0)
plt.show()
