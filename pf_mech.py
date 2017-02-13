import dolfin as df
import numpy as np
import math
from mpi4py import MPI


class PeriodicBoundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return bool(x[0] < df.DOLFIN_EPS and
                    x[0] > -df.DOLFIN_EPS and on_boundary)

    def map(self, x, y):
        y[0] = x[0] - 1.0
        y[1] = x[1]


class TopBoundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return bool(x[1] > (1.0 - df.DOLFIN_EPS)
                    and on_boundary)


class BottomBoundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return bool(x[1] < df.DOLFIN_EPS
                    and on_boundary)


def mask_f(phi):
    return 0.5*(1.+phi)
    # return 0.25*(2. - 3.*phi + phi**3)


def mask_s(phi):
    return 0.5*(1.-phi)


def dmask_f(phi):
    return 0.5
    # return -0.75*(1. - phi**2)


def dmask_s(phi):
    return -0.5


def mask_trans_f(phi, eps):
    return 2.757*0.5*mask_f(phi)*((1.-phi)/eps)**2
    # return 2.757*(1.+phi)**2*mask(phi)/(4.*eps**2)


def mag(u):
    return df.sqrt(df.dot(u, u))


def sigma_s(v, E, nu):
    mu = E/(2.*(1.+nu))
    lmbda = E*nu/((1.+nu)*(1.-2.*nu))
    return 2.0*mu*df.sym(df.grad(v)) + \
        lmbda*df.tr(df.sym(df.grad(v)))*df.Identity(len(v))


def sigma_f(v, p, mu):
    return - p * df.Identity(len(v)) + 2*mu*df.sym(df.grad(v))


def main():
    eps = 0.01
    mu = 1.
    u_top = df.Constant((1., 0.))
    p_0 = 1.
    u_s_btm = df.Constant((0., 0.))
    E = 1.0e2  # Young's modulus
    nu = 0.3  # Poisson's ratio
    N = 100

    ffc_options = {"optimize": True,
                   "cpp_optimize": True,
                   "eliminate_zeros": True,
                   "precompute_basis_const": True,
                   "precompute_ip_const": True}
    df.parameters['form_compiler']['cpp_optimize'] = True
    df.parameters['form_compiler']['optimize'] = True
    
    mesh = df.RectangleMesh(df.Point(0., 0.),
                            df.Point(1., 1.),
                            N, N)
    pbc = PeriodicBoundary()
    top = TopBoundary()
    btm = BottomBoundary()

    boundaries = df.FacetFunction("size_t", mesh)
    boundaries.set_all(0)
    top.mark(boundaries, 1)
    btm.mark(boundaries, 2)

    P2 = df.VectorElement('P', df.triangle, 2)
    P1 = df.FiniteElement('P', df.triangle, 1)
    TH = P2 * P1
    W = df.FunctionSpace(mesh, TH, constrained_domain=pbc)
    V = df.FunctionSpace(mesh, P2, constrained_domain=pbc)
    P = df.FunctionSpace(mesh, P1, constrained_domain=pbc)

    # Define variational problem
    (u, p) = df.TrialFunctions(W)
    (v, q) = df.TestFunctions(W)
    w = df.Function(W)

    f = df.Constant((0., 0.))

    # phi_expr = df.Expression("tanh((x[1]-y_0)/delta)",
    #                          y_0=0.5, delta=eps/math.sqrt(2),
    #                          degree=1)
    phi_expr = df.Expression("tanh((x[1]-y_0-A*sin(2*n*pi*x[0]))/delta)",
                             y_0=0.5, delta=eps/math.sqrt(2), A=0*0.05, n=4,
                             degree=1)
    
    phi = df.interpolate(phi_expr, P)

    a_f = (mu * mask_f(phi) * df.inner(df.nabla_grad(v),
                                       df.nabla_grad(u)) * df.dx
           + mu * dmask_f(phi) *
           df.inner(df.nabla_grad(v),
                    df.outer(u, df.nabla_grad(phi))) * df.dx
           - mask_f(phi) * p * df.div(v) * df.dx
           - dmask_f(phi) * p * df.dot(v, df.nabla_grad(phi)) * df.dx
           + mask_f(phi) * q * df.div(u) * df.dx
           + dmask_f(phi) * q * df.dot(u, df.nabla_grad(phi)) * df.dx
           + mu * mask_trans_f(phi, eps) * df.dot(v, u) * df.dx)
    L_f = mask_f(phi) * df.dot(f, v) * df.dx

    bc_u = df.DirichletBC(W.sub(0), u_top, boundaries, 1)
    bc_p = df.DirichletBC(W.sub(1), df.Constant(p_0),
                          "x[0] < DOLFIN_EPS && "
                          "x[1] > 1.-DOLFIN_EPS",
                          "pointwise")

    bcs_f = [bc_u, bc_p]

    problem_f = df.LinearVariationalProblem(
        a_f, L_f, w, bcs=bcs_f,
        form_compiler_parameters=ffc_options)
    solver_f = df.LinearVariationalSolver(problem_f)

    solver_f.solve()

    u, p = w.split(deepcopy=False)

    u_s = df.TrialFunction(V)
    v_s = df.TestFunction(V)

    a_s = (df.inner(df.grad(v_s),
                    mask_s(phi)*sigma_s(u_s, E, nu)) +
           1e-7*df.inner(df.grad(v_s), mask_f(phi)*df.grad(u_s))) * df.dx
    L_s = df.inner(f, v_s) * df.dx \
        - df.inner(sigma_f(mask_f(phi)*u, mask_f(phi)*p, mu),
                   df.grad(v_s)) * df.dx

    bc_s_btm = df.DirichletBC(V, u_s_btm, boundaries, 2)
    bc_s_top = df.DirichletBC(V, u_s_btm, boundaries, 1)
    bcs_s = [bc_s_top, bc_s_btm]

    d = df.Function(V)

    problem_s = df.LinearVariationalProblem(
        a_s, L_s, d, bcs=bcs_s,
        form_compiler_parameters=ffc_options)
    solver_s = df.LinearVariationalSolver(problem_s)
    solver_s.solve()

    #df.plot(mesh)
    #df.plot(phi)
    #df.plot(u)
    #df.plot(mask_f(phi)*mag(u))
    #df.plot(mask_f(phi)*p)

    #df.plot(mask_s(phi))

    ultramask_s_phi = df.project(mask_s(phi), P)
    uma = ultramask_s_phi.vector().array()
    uma[uma < 0.01] = 0.
    ultramask_s_phi.vector()[:] = uma

    df.plot(ultramask_s_phi, title="ULTRAMASK")

    df.plot(mask_s(phi)*d)
    df.plot(mask_s(phi)*mag(d))
    df.plot(mask_s(phi)*df.tr(sigma_s(d, E, nu)))
    df.plot(ultramask_s_phi*df.inner(sigma_s(mask_s(phi)*d, E, nu),
                                     df.sym(df.nabla_grad(d))))

    df.interactive()


if __name__ == "__main__":
    main()
