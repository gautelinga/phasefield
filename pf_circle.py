import numpy as np
import dolfin as df
import mshr
from mpi4py import MPI
import math


def make_circle_mesh(R, res=40.):
    domain = mshr.Circle(df.Point(0., 0.), 1.)
    mesh = mshr.generate_mesh(domain, res)
    mesh.coordinates()[:] *= R
    return mesh


def dfdphi(phi, c, lamda):
    return (1.-phi**2)*(phi-c*lamda)


def mag(x):
    return df.sqrt(df.dot(x, x))


def comp_corr(p, P, V):
    grad_p = df.project(df.grad(p), V)
    mag_grad_p = mag(grad_p)
    kappa = df.div(grad_p/mag_grad_p)
    return df.project(kappa*mag_grad_p, P)


class Around(df.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


class PhaseFieldEquation(df.NonlinearProblem):
    def __init__(self, a, L):
        df.NonlinearProblem.__init__(self)
        self.L = L
        self.a = a

    def F(self, b, x):
        df.assemble(self.L, tensor=b)

    def J(self, A, x):
        df.assemble(self.L, tensor=A)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Form compiler options
    df.parameters["form_compiler"]["optimize"] = True
    df.parameters["form_compiler"]["cpp_optimize"] = True
    df.set_log_level(df.WARNING)

    ffc_options = {"optimize": True,
                   "eliminate_zeros": True,
                   "precompute_basis_const": True,
                   "precompute_ip_const": True}

    res = 150.
    R = 200.
    R_0 = 6.
    c_inf = 1.0
    Pe_PF = 1.
    # Da = 100.
    T = 5000.
    # lamda = Pe_PF / (5./3. + math.sqrt(2)/Da)
    lamda = 3.*Pe_PF/5.

    mesh = make_circle_mesh(R, res)
    mesh.init()

    h5f_mesh = df.HDF5File(mesh.mpi_comm(), "circle_mesh.h5", "w")
    h5f_mesh.write(mesh, "mesh")
    h5f_mesh.close()

    P = df.FunctionSpace(mesh, "CG", 1)
    V = df.VectorFunctionSpace(mesh, "CG", 1)

    domains = df.CellFunction("size_t", mesh)
    boundaries = df.FacetFunction("size_t", mesh)

    around = Around()
    boundaries.set_all(0)
    around.mark(boundaries, 1)

    bcs_phi = df.DirichletBC(P, df.Constant(1.), boundaries, 1)
    bcs_c = df.DirichletBC(P, df.Constant(c_inf), boundaries, 1)

    dx = df.Measure('dx', domain=mesh, subdomain_data=domains)
    ds = df.Measure('ds', domain=mesh, subdomain_data=boundaries)

    num_cells_in = np.array(float(mesh.num_entities(2)))
    num_cells = np.zeros(1)
    comm.Allreduce(num_cells_in, num_cells, op=MPI.SUM)

    dt = 0.9*np.sqrt(np.pi*R**2/num_cells[0])
    nt = int(T/dt)
    if rank == 0:
        print "dt =", dt
        print "nt =", nt

    phi_init = df.Expression("tanh((sqrt(x[0]*x[0]+x[1]*x[1]) - "
                             "R_0 + a*sin(n*atan2(x[0], x[1])))/delta)",
                             R_0=R_0,
                             delta=1.,
                             a=.0, n=6., degree=1)

    phi_init = df.interpolate(phi_init, P)

    # Define nonlinear variational problem
    dphi = df.TrialFunction(P)  #
    phi = df.Function(P)  # current solution
    phi_0 = df.Function(P)  # solution from previous step
    q = df.TestFunction(P)
    kappa = df.Function(P)  # curvature, based on previous
    kappa.vector()[:] = 0.

    phi.interpolate(phi_init)
    phi_0.interpolate(phi_init)

    c_test = df.TestFunction(P)
    c_trial = df.TrialFunction(P)
    c_0 = df.Function(P)
    c_0.vector()[:] = c_inf * 0.5*(phi_init.vector()[:]+1.)
    c = df.Function(P)
    c.interpolate(c_0)

    # df.plot(c, interactive=True)

    F = (Pe_PF*q*(phi - phi_0)/dt*dx
         + df.dot(df.grad(phi), df.grad(q))*dx
         - q*(dfdphi(phi, lamda, c_0))*dx)

    J = df.derivative(F, phi, dphi)

    F_c = c_test*(c_trial-c_0)/dt*dx + \
        df.dot(df.grad(c_trial), df.grad(c_test))*dx
    a, L = df.lhs(F_c), df.rhs(F_c)

    # df.solve(a == L, c, bcs_c)
    # df.plot(c, interactive=True)

    c_prob = df.LinearVariationalProblem(a, L, c, bcs_c)
    c_solver = df.LinearVariationalSolver(c_prob)

    # problem = PhaseFieldEquation(F, J)
    # solver = df.NewtonSolver()
    # solver.parameters["linear_solver"] = "lu"
    # solver.parameters["convergence_criterion"] = "incremental"
    # solver.parameters["relative_tolerance"] = 1e-6

    progress = df.Progress("Time-steppin'")
    # df.set_log_level(df.PROGRESS)

    field_names = ["phi", "c", "corr"]

    xf = dict()
    for field in field_names:
        xf[field] = df.XDMFFile(mesh.mpi_comm(), field + ".xdmf")
        xf[field].parameters["rewrite_function_mesh"] = False
        xf[field].parameters["flush_output"] = True

    phi_corr = df.Function(P)

    t = 0.
    for it in xrange(nt):
        if rank == 0:
            print "Step", it
        t += dt

        c_0.vector()[:] = c.vector()
        phi_0.vector()[:] = phi.vector()

        df.solve(F == 0, phi, bcs_phi, J=J,
                 form_compiler_parameters=ffc_options)
        phi_corr.assign(comp_corr(phi, P, V))

        phi.vector()[:] -= phi_corr.vector()[:]*dt

        c_solver.solve()
        c.vector()[:] += (phi.vector()[:] - phi_0.vector()[:])/dt

        if it % 10 == 0:
            xf["phi"].write(phi, t)
            xf["c"].write(c, t)
            xf["corr"].write(phi_corr, t)

        progress.update(t/T)

        # df.plot(phi, title="phi", interactive=True)


if __name__ == "__main__":
    main()
