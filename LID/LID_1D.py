import festim as F
import numpy as np
import sympy as sp
import properties

w_atom_density = 6.31e28  # atom/m3
n1 = 0.02894656 * w_atom_density
E_p1 = 1.1081157
n2 = 0.01790 * w_atom_density
E_p2 = 1.27906163
n3 = 0.01967725 * w_atom_density
E_p3 = 1.53602897
n4 = 0.00600191 * w_atom_density
E_p4 = 1.81760592

D0_W = 1.93e-7 / np.sqrt(2)
Ed_W = 0.2


def rad(T, _):
    return -5.670374419e-8 * (T**4 - 300**4)


def run_1D(E0, r, duration, output_folder):

    def pulse(t, r):
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
                1.78315e-04,
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
            * sp.Piecewise(
                (f1(t), t <= t1),
                (f1(t1) * f2(t), (t > t1) & (t <= t2)),
                (f1(t1) * f2(t2) * f3(t), True),
            )
            / norm
        )

    if duration == "1ms":
        final_time = 1e-1
        t_max = 1e-3
        max_stepsize = lambda t: 5e-6 if t < 1e-3 else 0.5e-3
    elif duration == "250us":
        final_time = 5e-3
        t_max = 1.7e-4
        max_stepsize = (lambda t: 1e-5 if t < 2.5e-4 else 5e-4,)

    export_times = [t_max, final_time]

    # Define Simulation object
    model = F.Simulation(log_level=40)

    # Define a simple mesh
    vertices = np.concatenate(
        [
            np.linspace(0, 1e-6, num=1000),
            np.linspace(1e-6, 1e-4, num=500),
            np.linspace(1e-4, 6e-3 + 1e-6, num=500),
        ]
    )

    model.mesh = F.MeshFromVertices(vertices)

    # Define material properties
    tungsten = F.Material(
        id=1,
        D_0=D0_W,
        E_D=Ed_W,
        rho=properties.rho_W,
        thermal_cond=properties.thermal_cond_function_W,
        heat_capacity=properties.heat_capacity_function_W,
        Q=properties.heat_of_transport_function_W,
        borders=[0, 1e-6],
    )
    copper = F.Material(
        id=2,
        D_0=0,
        E_D=0,
        rho=1,
        heat_capacity=properties.rhoCp_Cu,
        thermal_cond=properties.thermal_cond_Cu,
        Q=0,
        borders=[1e-6, 6e-3 + 1e-6],
    )

    model.materials = F.Materials([tungsten, copper])

    # Define traps

    model.traps = [
        F.Trap(
            k_0=D0_W / (1.1e-10**2 * 6 * w_atom_density),
            E_k=Ed_W,
            p_0=1e13,
            E_p=E_p1,
            density=n1,
            materials=model.materials[0],
        ),
        F.Trap(
            k_0=D0_W / (1.1e-10**2 * 6 * w_atom_density),
            E_k=Ed_W,
            p_0=1e13,
            E_p=E_p2,
            density=n2,
            materials=model.materials[0],
        ),
        F.Trap(
            k_0=D0_W / (1.1e-10**2 * 6 * w_atom_density),
            E_k=Ed_W,
            p_0=1e13,
            E_p=E_p3,
            density=n3,
            materials=model.materials[0],
        ),
        F.Trap(
            k_0=D0_W / (1.1e-10**2 * 6 * w_atom_density),
            E_k=Ed_W,
            p_0=1e13,
            E_p=E_p4,
            density=n4,
            materials=model.materials[0],
        ),
    ]

    # Set initial conditions
    model.initial_conditions = [
        F.InitialCondition(
            field="1", value=n1 * sp.Piecewise((1, F.x < 1e-6), (0, True))
        ),
        F.InitialCondition(
            field="2", value=n2 * sp.Piecewise((1, F.x < 1e-6), (0, True))
        ),
        F.InitialCondition(
            field="3", value=n3 * sp.Piecewise((1, F.x < 1e-6), (0, True))
        ),
        F.InitialCondition(
            field="4", value=n4 * sp.Piecewise((1, F.x < 1e-6), (0, True))
        ),
    ]

    # Set boundary conditions
    model.boundary_conditions = [
        F.DirichletBC(surfaces=[1, 2], value=0, field="solute"),
        F.FluxBC(surfaces=1, value=pulse(F.t, r), field="T"),
        F.CustomFlux(surfaces=1, field="T", function=rad),
        F.CustomFlux(surfaces=2, field="T", function=rad),
    ]

    # Define the material temperature evolution
    model.T = F.HeatTransferProblem(
        initial_condition=300,
        absolute_tolerance=1.0,
        relative_tolerance=1e-3,
        maximum_iterations=50,
    )

    # Define the simulation settings
    model.dt = F.Stepsize(
        initial_value=1e-6,
        stepsize_change_ratio=1.1,
        max_stepsize=max_stepsize,
        dt_min=1e-8,
        milestones=export_times,
    )

    model.settings = F.Settings(
        absolute_tolerance=1e12,
        relative_tolerance=1e-9,
        final_time=final_time,
        soret=True,
        traps_element_type="DG",
    )

    # Define the exports
    derived_quantities = F.DerivedQuantities(
        [
            F.HydrogenFlux(surface=1),
            F.TotalVolume(field="retention", volume=1),
        ],
    )

    TXT = [
        F.TXTExport(
            field="retention",
            filename=output_folder + f"retention_{duration}_E{E0:.3f}_r{r:.2e}.txt",
            times=export_times,
        ),
        F.TXTExport(
            field="T",
            filename=output_folder + f"T_{duration}_E{E0:.3f}_r{r:.2e}.txt",
            times=export_times,
        ),
    ]

    model.exports = [derived_quantities] + TXT
    model.initialise()
    model.run()

    return derived_quantities
