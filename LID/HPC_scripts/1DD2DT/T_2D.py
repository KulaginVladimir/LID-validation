import festim as F
import numpy as np
import sympy as sp
import properties
import sys

N = float(sys.argv[1])
N_tot = float(sys.argv[2])
duration = sys.argv[3]

if duration == "1ms":
    E_max = 1.003
    final_time = 1e-1
    t_max = 1e-3
    max_stepsize = lambda t: 2e-5 if t < 2e-3 else 1e-3
elif duration == "250us":
    E_max = 0.351
    final_time = 1e-2
    t_max = 1.7e-4
    max_stepsize = lambda t: 5e-6 if t < 5e-4 else 5e-4

E0 = E_max * (N - 1) / (N_tot - 1)


def pulse(r, t):
    FWHM = 1e-3
    sigma_r = FWHM / 2 / np.sqrt(2 * np.log(2))

    time_profile = {
        "1ms": [
            2.00546391e-04,
            1.01122930e-03,
            3.38692875e-07,
            3.23973615e-02,
            1.97812839e-04,
            # 1.08923e-03,
            910e-6,
        ],
        "250us": [
            1.15673171e-04,
            2.19229191e-04,
            3.71988035e-06,
            1.89739539e-02,
            1.82511663e-05,
            # 1.78315e-04,
            # 1.2665e-4,
            1.5e-4,
        ],
    }
    t1, t2, dt1, delta, dt2, norm = time_profile[duration]

    f1 = lambda t: 1 / (1 + sp.exp(-(t - 0.5 * t1) / dt1))
    f2 = lambda t: 1 - delta * (t - t1) / (t2 - t1)
    f3 = lambda t: sp.exp(-(t - t2) / dt2)

    return (
        E0
        * sp.exp(-(r**2) / 2 / sigma_r**2)
        / 2
        / np.pi
        / sigma_r**2
        * (1 / norm)
        * sp.Piecewise(
            (f1(t), t <= t1),
            (f1(t1) * f2(t), (t > t1) & (t <= t2)),
            (0, True),  # the trailing edge is not accounted for
        )
    )


def rad(T, _):
    return -5.670374419e-8 * (T**4 - 300**4)


# Define Simulation object
model = F.Simulation(log_level=40)

model.mesh = F.MeshFromXDMF(
    volume_file="./mesh/mesh.xdmf", boundary_file="./mesh/mf.xdmf", type="cylindrical"
)

# Define material properties
tungsten = F.Material(
    id=2,
    D_0=0,
    E_D=0,
    rho=properties.rho_W,
    thermal_cond=properties.thermal_cond_function_W,
    heat_capacity=properties.heat_capacity_function_W,
    Q=properties.heat_of_transport_function_W,
)
copper = F.Material(
    id=1,
    D_0=0,
    E_D=0,
    rho=1,
    heat_capacity=properties.rhoCp_Cu,
    thermal_cond=properties.thermal_cond_Cu,
    Q=0,
)

model.materials = F.Materials([tungsten, copper])


# Set boundary conditions
model.boundary_conditions = [
    F.FluxBC(surfaces=6, value=pulse(F.x, F.t), field="T"),
    F.CustomFlux(surfaces=[3, 6], field="T", function=rad),
]

# Define the material temperature evolution
model.T = F.HeatTransferProblem(
    initial_condition=300,
    absolute_tolerance=1e-1,
    relative_tolerance=1e-4,
    maximum_iterations=50,
)

# Define the simulation settings
model.dt = F.Stepsize(
    initial_value=1e-7,
    stepsize_change_ratio=1.1,
    max_stepsize=max_stepsize,
    dt_min=1e-8,
)

model.settings = F.Settings(
    absolute_tolerance=1e12,
    relative_tolerance=1e-8,
    final_time=final_time,
)

# Define the exports
points_list = []
rs = np.linspace(0, 1.5e-3, 62, endpoint=True)

for r in rs:
    points_list.append(F.PointValue(field="T", x=[r, 6e-3 + 1e-6]))

derived_quantities = F.DerivedQuantities(
    points_list, show_units=True, filename=f"../T_{duration}/Trz_E{E0:.3f}.csv"
)

model.exports = [derived_quantities]  # + XDMF
model.initialise()
model.run()
