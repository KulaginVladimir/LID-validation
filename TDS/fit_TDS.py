import festim as F
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import random

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
    D0 = 1.93e-7/np.sqrt(2)
    Ed = 0.2

    # Define Simulation object
    model = F.Simulation()

    # Define a simple mesh
    vertices = np.linspace(0, 1e-6, num=1000)
    model.mesh = F.MeshFromVertices(vertices)

    # Define material properties
    tungsten = F.Material(
        id=1,
        D_0=D0,
        E_D=Ed,
    )
    model.materials = tungsten

    # Define traps
    trap_1 = F.Trap(
        k_0=D0 / (1.1e-10**2 * 6 * w_atom_density),
        E_k=Ed,
        p_0=1e13,
        E_p=E_p1,
        density=n1 * w_atom_density,
        materials=tungsten,
    )

    trap_2 = F.Trap(
        k_0=D0 / (1.1e-10**2 * 6 * w_atom_density),
        E_k=Ed,
        p_0=1e13,
        E_p=E_p2,
        density=n2 * w_atom_density,
        materials=tungsten,
    )

    trap_3 = F.Trap(
        k_0=D0 / (1.1e-10**2 * 6 * w_atom_density),
        E_k=Ed,
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
        final_time=1600,
        maximum_iterations=50,
    )

    # Define the exports
    derived_quantities = F.DerivedQuantities(
        [
            F.HydrogenFlux(surface=1),
            F.HydrogenFlux(surface=2),
            F.AverageVolume(field="T", volume=1),
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

    # Get the simulation result
    try: 
        res = TDS(*prm)

        T = np.array(res.filter(fields="T").data)
        flux = -np.array(res.filter(fields="solute", surfaces=1).data)

        # Plot the intermediate TDS spectra
        #if ITERATION == 0:
        #    plt.plot(T, flux, color="tab:red", lw=2, label="Initial guess")
        #else:
        #    plt.plot(T, flux, color="tab:grey", lw=0.5)

        interp_tds = interp1d(T, flux, fill_value="extrapolate")

        # Compute the mean absolute error between sim and ref
        err = np.abs(interp_tds(ref[:, 0]) - ref[:, 1]).mean()

        print(f"Average absolute error is : {err:.2e}")
        return err
    except:
        return 1e30
    

ref = np.genfromtxt("./TDS_S160724.csv", delimiter=",", skip_header=1)

dim_n1 = Real(low=1e-5, high=1e-1, name="trap_conc1", prior="log-uniform")
dim_Edt1 = Real(low=0.5, high=2.1, name="Edt1")
dim_n2 = Real(low=1e-5, high=1e-1, name="trap_conc2", prior="log-uniform")
dim_Edt2 = Real(low=0.5, high=2.1, name="Edt2")
dim_n3 = Real(low=1e-5, high=1e-1, name="trap_conc3", prior="log-uniform")
dim_Edt3 = Real(low=0.5, high=2.1, name="Edt3")

dimensions = [
    dim_n1,
    dim_Edt1,
    dim_n2,
    dim_Edt2,
    dim_n3,
    dim_Edt3,
]

default_parameters = [0.002617564252438404, 1.234176427898428, 0.021317600436734342, 1.0564506829439768, 0.0065940178872959205, 1.6789668121378427]

@use_named_args(dimensions=dimensions)
def fitness(trap_conc1, Edt1, trap_conc2, Edt2, trap_conc3, Edt3):
    config = [trap_conc1, Edt1, trap_conc2, Edt2, trap_conc3, Edt3]
    global ITERATION

    info(ITERATION, config)

    # possibility to change where we save
    error = error_function(config)

    if np.isnan(error):
        error = 1e30

    ITERATION += 1
    return error


ITERATION = 0
n_calls = 250

res = gp_minimize(
    func=fitness,
    dimensions=dimensions,
    acq_func="EI",  # Expected Improvement.
    n_calls=n_calls,
    x0=default_parameters,
    random_state=random.randint(1,10000),
)

print(res.x)

# Process the obtained results
predicted_data = TDS(*res.x)

T = predicted_data.filter(fields="T").data

flux_left = -np.array(predicted_data.filter(fields="solute", surfaces=1).data)

# Visualise
plt.plot(ref[:, 0], ref[:, 1], linewidth=2, label="Reference")
plt.plot(T, flux_left, linewidth=2, label="Optimised")

plt.ylabel(r"Desorption flux (m$^{-2}$ s$^{-1}$)")
plt.xlabel(r"Temperature (K)")
plt.legend()
plt.show()