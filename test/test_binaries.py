import os
import math
import pytest
import pyomo.environ as pyo
from pyomo.contrib.sensitivity_toolbox.sens import sensitivity_calculation


# TODO: This directory will be wherever we download the binaries that we
# want to test, likely just in the current working directory.
if "IDAES_DIR" in os.environ:
    IDAES_DIR = os.environ["IDAES_DIR"]
else:
    IDAES_DIR = os.path.join(os.environ["HOME"], ".idaes")
ipopts_to_test = [
    ("ipopt", os.path.join(IDAES_DIR, "bin", "ipopt")),
    ("ipopt_l1", os.path.join(IDAES_DIR, "bin", "ipopt_l1")),
    #("cyipopt", None),
]
ipopt_options_to_test = [
    ("default", {}),
    ("mumps", {"print_user_options": "yes", "linear_solver": "mumps"}),
    ("ma27", {"print_user_options": "yes", "linear_solver": "ma27"}),
    ("ma57", {"print_user_options": "yes", "linear_solver": "ma57"}),
    ("ma57_metis", {"print_user_options": "yes", "linear_solver": "ma57", "ma57_pivot_order": 4}),
]
sensitivity_solvers = [
    ("ipopt", "k_aug", "dot_sens"),
    ("ipopt_sens", "ipopt_sens", None),
    ("ipopt_sens_l1", "ipopt_sense_l1", None),
]


def _test_ipopt_with_options(name, exe, options):
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1,2], initialize=1.5)
    m.con = pyo.Constraint(expr=m.x[1]*m.x[2] == 0.5)
    m.obj = pyo.Objective(expr=m.x[1]**2 + 2*m.x[2]**2)

    if exe is None:
        solver = pyo.SolverFactory(name, options=options)
    else:
        solver = pyo.SolverFactory(name, executable=exe, options=options)

    solver.solve(m, tee=True)

    target_sol = [("x[1]", 0.840896415), ("x[2]", 0.594603557)]
    assert all(
        math.isclose(m.find_component(name).value, val, abs_tol=1e-7)
        for name, val in target_sol
    )


class TestIpopt:
    pass


# Set test functions on the test class
for solver_name, exe in ipopts_to_test:
    for option_label, options in ipopt_options_to_test:
        attr_name = f"test_{solver_name}_{option_label}"
        setattr(
            TestIpopt,
            attr_name,
            lambda self: _test_ipopt_with_options(solver_name, exe, options),
        )


class TestBonmin:

    exe = os.path.join(IDAES_DIR, "bin", "bonmin")

    def _make_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2], initialize=1.5)
        m.y = pyo.Var(domain=pyo.PositiveIntegers)
        m.con = pyo.Constraint(expr=m.x[1]*m.x[2] == m.y)
        m.obj = pyo.Objective(expr=m.x[1]**2 + 2*m.x[2]**2)
        return m

    def test_bonmin_default(self):
        m = self._make_model()
        solver = pyo.SolverFactory("bonmin", executable=exe)
        solver.solve(m, tee=True)

        assert math.isclose(m.y.value, 1.0, abs_tol=1e-7)
        assert math.isclose(m.x[1].value, 1.18920710, abs_tol=1e-7)
        assert math.isclose(m.x[2].value, 0.84089641, abs_tol=1e-7)


class TestCouenne:

    exe = os.path.join(IDAES_DIR, "bin", "couenne")

    def _make_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2], initialize=1.5)
        m.y = pyo.Var(domain=pyo.PositiveIntegers)
        m.con = pyo.Constraint(expr=m.x[1]*m.x[2] == m.y)
        m.obj = pyo.Objective(expr=(m.x[1] + 0.01)**2 + 2*(m.x[2] + 0.01)**2)
        return m

    def test_couenne_default(self):
        m = self._make_model()
        solver = pyo.SolverFactory("couenne", executable=exe)
        solver.solve(m, tee=True)

        assert math.isclose(m.y.value, 1.0, abs_tol=1e-7)
        assert math.isclose(m.x[1].value, -1.18816674, abs_tol=1e-7)
        assert math.isclose(m.x[2].value, -0.84163270, abs_tol=1e-7)


def _test_sensitivity(
    solver_name,
    solver_exe,
    sens_name,
    sens_exe,
    update_name,
    update_exe,
):
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1,2], initialize=1.5)
    m.p = pyo.Param(mutable=True, initialize=0.5)
    m.con = pyo.Constraint(expr=m.x[1]*m.x[2] == m.p)
    m.obj = pyo.Objective(expr=m.x[1]**2 + 2*m.x[2]**2)

    if exe is None:
        solver = pyo.SolverFactory(solver_name)
    else:
        solver = pyo.SolverFactory(solver_name, executable=solver_exe)

    solver.solve(m, tee=True)

    sensitivity_calculation(
        sens_name,
        m,
        [m.p],
        [0.7],
        cloneModel=False,
        tee=True,
    )


class TestSensitivity:
    pass


if __name__ == "__main__":
    #pytest.main([__file__])
    _test_sensitivity(
        "ipopt_sens",
        os.path.join(IDAES_DIR, "bin", "ipopt_sens"),
        "ipopt_sens",
        os.path.join(IDAES_DIR, "bin", "ipopt_sens"),
        None,
        None,
    )
