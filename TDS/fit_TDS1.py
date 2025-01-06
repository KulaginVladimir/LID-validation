import festim as F
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def info(i, p):
    """
    Print information during the fitting procedure
    """
    print("-" * 40)
    print(f"i = {i}")
    print("New simulation.")
    print(f"Point is: {p}")


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
    vertices = np.linspace(0, 1e-6, num=500)
    
    model.mesh = F.MeshFromVertices(vertices)

    # Define material properties
    tungsten = F.Material(
        id=1,
        D_0=D0_W,
        E_D=Ed_W,
    )

    model.materials = [tungsten]

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
        F.DirichletBC(surfaces=1, value=0, field=0)
    ]

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
        final_time=1400,
        maximum_iterations=50,
    )

    # Define the exports
    derived_quantities = F.DerivedQuantities(
        [
            F.HydrogenFlux(surface=1),
            F.HydrogenFlux(surface=2),
            F.AverageVolume(field="T", volume=1),
            F.TotalVolume(field="1", volume=1),
            F.TotalVolume(field="2", volume=1),
            F.TotalVolume(field="3", volume=1)
        ],
        show_units=True
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
    flux = -np.array(res.filter(fields="solute", surfaces=1).data) - np.array(res.filter(fields="solute", surfaces=2).data)

    interp_tds = interp1d(T, flux, fill_value="extrapolate")

    # Compute the mean absolute error between sim and ref
    err = np.abs(interp_tds(ref[:, 0]) - ref[:, 1]).mean()

    if i % 25 == 0:
        tr1 = np.array(res.filter(fields="1").data)
        tr2 = np.array(res.filter(fields="2").data)
        tr3 = np.array(res.filter(fields="3").data)
        time = np.array(res.t)

        dtr1 = -np.diff(tr1)/np.diff(time)
        dtr2 = -np.diff(tr2)/np.diff(time)
        dtr3 = -np.diff(tr3)/np.diff(time)

        plt.plot(T, -np.array(res.filter(fields="solute", surfaces=1).data), label="left")
        plt.plot(T, -np.array(res.filter(fields="solute", surfaces=2).data), label="right")

        plt.plot(T[1:], dtr1, label="1", ls='dashed')
        plt.plot(T[1:], dtr2, label="2", ls='dashed')
        plt.plot(T[1:], dtr3, label="3", ls='dashed')

        plt.plot(ref[:, 0], ref[:, 1], linewidth=2, label="Reference")
        plt.legend()
        plt.show()
    print(f"Average absolute error is : {err:.2e}")
    return err
    

ref = np.genfromtxt("./TDS_S160724.csv", delimiter=",", skip_header=1)

i = 0  # initialise counter

# Set the tolerances
fatol = 1e18
xatol = 1e-3

initial_guess = [0.00984528, 1.55595644, 0.01711844, 1.07135345, 0.01034346, 1.2971446]

# Minimise the error function
res = minimize(
    error_function,
    np.array(initial_guess),
    method="Nelder-Mead",
    options={"disp": True, "fatol": fatol, "xatol": xatol},
)

# Process the obtained results
predicted_data = TDS(*res.x)

T = predicted_data.filter(fields="T").data

flux = -(np.array(predicted_data.filter(fields="solute", surfaces=1).data) + np.array(predicted_data.filter(fields="solute", surfaces=2).data))

# Visualise
plt.plot(ref[:, 0], ref[:, 1], linewidth=2, label="Reference")
plt.plot(T, flux, linewidth=2, label="Optimised")

plt.ylabel(r"Desorption flux (m$^{-2}$ s$^{-1}$)")
plt.xlabel(r"Temperature (K)")
plt.legend()
plt.show()