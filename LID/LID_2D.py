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

n1 = 0.02894656 * w_atom_density
E_p1 = 1.1081157
n2 = 0.01790 * w_atom_density
E_p2 = 1.27906163
n3 = 0.01967725 * w_atom_density
E_p3 = 1.53602897
n4 = 0.00600191 * w_atom_density
E_p4 = 1.81760592


def rad(T, _):
    return -5.670374419e-8 * (T**4 - 300**4)


def run_2D(E0, duration, output_folder):

    if duration == "1ms":
        final_time = 1e-1
        t_max = 1e-3
        max_stepsize = lambda t: 2.5e-5 if t < 2e-3 else 1e-3
    elif duration == "250us":
        final_time = 1e-2
        t_max = 1.7e-4
        max_stepsize = lambda t: 2e-6 if t < 4e-4 else 5e-4

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
                (f1(t), t <= t1), (f1(t1) * f2(t), (t > t1) & (t <= t2)), (0, True)
            )
        )

    # Define Simulation object
    model = F.Simulation(log_level=40)

    model.mesh = F.MeshFromXDMF(
        volume_file="./mesh_LID2D/mesh.xdmf",
        boundary_file="./mesh_LID2D/mf.xdmf",
        type="cylindrical",
    )

    # Define material properties
    tungsten = F.Material(
        id=2,
        D_0=D0_W,
        E_D=Ed_W,
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

    n1 = 0.02894656 * w_atom_density
    E_p1 = 1.1081157
    n2 = 0.01790 * w_atom_density
    E_p2 = 1.27906163
    n3 = 0.01967725 * w_atom_density
    E_p3 = 1.53602897
    n4 = 0.00600191 * w_atom_density
    E_p4 = 1.81760592

    trap_1 = F.Trap(
        k_0=D0_W / (1.1e-10**2 * 6 * w_atom_density),
        E_k=Ed_W,
        p_0=1e13,
        E_p=E_p1,
        density=n1,
        materials=tungsten,
    )

    trap_2 = F.Trap(
        k_0=D0_W / (1.1e-10**2 * 6 * w_atom_density),
        E_k=Ed_W,
        p_0=1e13,
        E_p=E_p2,
        density=n2,
        materials=tungsten,
    )

    trap_3 = F.Trap(
        k_0=D0_W / (1.1e-10**2 * 6 * w_atom_density),
        E_k=Ed_W,
        p_0=1e13,
        E_p=E_p3,
        density=n3,
        materials=tungsten,
    )

    trap_4 = F.Trap(
        k_0=D0_W / (1.1e-10**2 * 6 * w_atom_density),
        E_k=Ed_W,
        p_0=1e13,
        E_p=E_p4,
        density=n4,
        materials=tungsten,
    )

    model.traps = [trap_1, trap_2, trap_3, trap_4]

    model.initial_conditions = [
        F.InitialCondition(
            field="1", value=n1 * sp.Piecewise((1, F.y > 6e-3), (0, True))
        ),
        F.InitialCondition(
            field="2", value=n2 * sp.Piecewise((1, F.y > 6e-3), (0, True))
        ),
        F.InitialCondition(
            field="3", value=n3 * sp.Piecewise((1, F.y > 6e-3), (0, True))
        ),
        F.InitialCondition(
            field="4", value=n4 * sp.Piecewise((1, F.y > 6e-3), (0, True))
        ),
    ]

    # Set boundary conditions
    model.boundary_conditions = [
        F.FluxBC(surfaces=6, value=pulse(F.x, F.t), field="T"),
        F.CustomFlux(surfaces=[3, 6], field="T", function=rad),
        F.DirichletBC(surfaces=6, value=0, field=0),
    ]

    # Define the material temperature evolution
    model.T = F.HeatTransferProblem(
        initial_condition=300,
        absolute_tolerance=1e-1,
        relative_tolerance=1e-4,
        maximum_iterations=30,
        linear_solver="mumps",
    )

    # Define the simulation settings
    model.dt = F.Stepsize(
        initial_value=5e-7,
        stepsize_change_ratio=1.1,
        max_stepsize=max_stepsize,
        dt_min=1e-8,
    )

    model.settings = F.Settings(
        absolute_tolerance=1e10,
        relative_tolerance=1e-10,
        final_time=final_time,
        soret=True,
        maximum_iterations=30,
        traps_element_type="DG",
        linear_solver="mumps",
    )

    # Define the exports
    derived_quantities = F.DerivedQuantities(
        [
            F.SurfaceFluxCylindrical(field="solute", surface=6),
        ],
        show_units=True,
    )

    XDMF = [
        F.XDMFExport(
            field="T",
            filename=output_folder + f"fields_{duration}_E{E0:.3f}/T.xdmf",
            checkpoint=True,
            label="T",
            mode="last",
        ),
        F.XDMFExport(
            field="retention",
            filename=output_folder + f"fields_{duration}_E{E0:.3f}/retention.xdmf",
            checkpoint=True,
            label="retention",
            mode="last",
        ),
    ]

    model.exports = [derived_quantities] + XDMF
    model.initialise()
    model.run()

    return derived_quantities
