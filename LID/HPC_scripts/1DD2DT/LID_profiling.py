import festim as F
import numpy as np
import sympy as sp
import properties
import sys
import fenics as f
from scipy.interpolate import interp1d


class InterpolatedExpression(f.UserExpression):
    def __init__(self, f):
        super().__init__()
        self.f = f
        self.t = 0

    def eval(self, value, x):
        value[0] = self.f(self.t)


class DirichletBCFromData(F.DirichletBC):
    def __init__(self, surfaces, f, field):
        value = InterpolatedExpression(f)
        super().__init__(surfaces, value, field)

    # override the create_expression method
    def create_expression(self, T):
        self.expression = self.value


N = float(sys.argv[1])
N_tot = float(sys.argv[2])
duration = sys.argv[3]

if duration == "1ms":
    E_max = 1.003
    final_time = 1e-1
    t_max = 1e-3
    max_stepsize = lambda t: 2.5e-5 if t < 2e-3 else 1e-3
elif duration == "250us":
    E_max = 0.351
    final_time = 1e-2
    t_max = 1.7e-4
    max_stepsize = lambda t: 2.5e-6 if t < 5e-4 else 5e-4


E0 = E_max * (N - 1) / (N_tot - 1)
export_times = [t_max, final_time]

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


def run(r, T_int, export_times):
    # Define Simulation object
    model = F.Simulation(log_level=40)

    # Define a simple mesh
    vertices = np.concatenate(
        [
            np.linspace(0, 1.1e-6, num=1000),
            np.linspace(1.1e-6, 1e-4, num=500),
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
        F.DirichletBC(surfaces=1, value=0, field="solute"),
        DirichletBCFromData(surfaces=1, field="T", f=T_int),
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
        relative_tolerance=1e-8,
        final_time=final_time,
        soret=True,
        # chemical_pot=True,
        traps_element_type="DG",
    )

    TXT = [
        F.TXTExport(
            field="retention",
            filename=f"../results_{duration}/retention_{duration}_E{E0:.3f}_r{r:.3e}.txt",
            times=export_times,
            write_at_last=True,
        ),
        F.TXTExport(
            field="T",
            filename=f"../results_{duration}/T_{duration}_E{E0:.3f}_r{r:.3e}.txt",
            times=export_times,
            write_at_last=True,
        ),
    ]

    # Define the exports
    derived_quantities = F.DerivedQuantities(
        [
            F.HydrogenFlux(surface=1),
            F.TotalVolume(field="retention", volume=1),
        ],
        show_units=True,
    )
    model.exports = [derived_quantities]

    if E0 == 1.003:
        model.exports += TXT

    model.initialise()
    model.run()
    return derived_quantities


rs = np.linspace(0, 1.5e-3, 62, endpoint=True)

# rets_in = np.zeros_like(rs)
rets_fin = np.zeros_like(rs)

T_surf = np.loadtxt(
    f"/mnt/pool/6/vvkulagin/FESTIM/LID_validation/T_2D/T_{duration}/Trz_E{E0:.3f}.csv",  # path to the file produced by T_2D.py
    skiprows=1,
    delimiter=",",
)

fluxes = np.zeros_like(rs)

for i, r in enumerate(rs):
    print(f"Iteration {i}: r={r/1e-3:.3e} mm")

    T_int = interp1d(T_surf[:, 0], T_surf[:, i + 1])

    data = run(r, T_int, export_times)

    flux = -np.array(data[0].data)
    t = np.array(data.t)

    fluxes[i] = np.trapz(flux, x=t)
    retention = np.array(data[1].data)

    rets_fin[i] = retention[-1]

# initial_retention = np.trapz(2*np.pi*rs*rets_in, x=rs)
final_retention = np.trapz(2 * np.pi * rs * rets_fin, x=rs)

des = np.trapz(2 * np.pi * rs * fluxes, x=rs)

export = np.column_stack([rs.transpose(), rets_fin.transpose()])
header = f"E={E0:.3f},Desorbed={des},ExportTimes={export_times}"

np.savetxt(
    f"../results_{duration}/profiles_{duration}_E{E0:.3f}.txt",
    export,
    header=header,
    delimiter=",",
    comments="",
)
