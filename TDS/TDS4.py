import festim as F
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


def TDS(n1, E_p1, n2, E_p2, n3, E_p3, n4, E_p4):
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
    D0_W = 1.93e-7 / np.sqrt(2)
    Ed_W = 0.2

    # Define Simulation object
    model = F.Simulation()

    # Define a simple mesh
    vertices = np.concatenate([np.linspace(0, 0.5e-6, num=500)])

    model.mesh = F.MeshFromVertices(vertices)

    # Define material properties
    tungsten = F.Material(
        id=1,
        D_0=D0_W,
        E_D=Ed_W,
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
        F.InitialCondition(field="1", value=n1 * w_atom_density),
        F.InitialCondition(field="2", value=n2 * w_atom_density),
        F.InitialCondition(field="3", value=n3 * w_atom_density),
        F.InitialCondition(field="4", value=n4 * w_atom_density),
    ]

    # Set boundary conditions
    model.boundary_conditions = [F.DirichletBC(surfaces=[1], value=0, field=0)]

    # Define the material temperature evolution
    ramp = 0.5  # K/s
    start_tds_time = 200
    T_value = 300 + ramp * F.t
    model.T = F.Temperature(value=T_value)

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


ref = np.genfromtxt("./TDS_S160724.csv", delimiter=",", skip_header=1)

initial_guess = [
    0.00348121,
    1.74028978,
    0.01613249,
    1.06772256,
    0.0086282,
    1.25484157,
    0.00993691,
    1.48390902,
]

# initial_guess = [0.00654021, 1.80704673, 0.0305249,  1.0858882,  0.01988605, 1.27684133, 0.02048084, 1.53107981]
initial_guess = [
    0.00656028,
    1.80624193,
    0.03050205,
    1.08579679,
    0.01988972,
    1.27651477,
    0.02048154,
    1.53088963,
]

initial_guess = [
    0.01021946,
    1.71838558,
    0.02734196,
    1.07831671,
    0.02016937,
    1.24589215,
    0.01926433,
    1.4854398,
]

initial_guess = [
    0.00600191,
    1.81760592,
    0.02894656,
    1.1081157,
    0.01790,
    1.27906163,
    0.01967725,
    1.53602897,
]
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

params = {
    "font.size": 12,
}
plt.rcParams.update(params)

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

plt.savefig("TDS_fitted.png", dpi=500, bbox_inches="tight", pad_inches=0.02)
plt.show()
