from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Dict, Iterable, List, Tuple

LpMinimize = 1
LpStatus = {1: "Optimal", -1: "Infeasible"}


def _as_expr(value: float | "LpAffineExpression" | "LpVariable") -> "LpAffineExpression":
    return LpAffineExpression.from_any(value)


@dataclass(eq=False)
class LpVariable:
    name: str
    lowBound: float | None = None
    upBound: float | None = None
    cat: str | None = None
    value: float = 0.0

    def __hash__(self) -> int:
        return object.__hash__(self)

    def __mul__(self, other: float) -> "LpAffineExpression":
        return LpAffineExpression({self: other}, 0.0)

    def __rmul__(self, other: float) -> "LpAffineExpression":
        return self.__mul__(other)

    def __add__(self, other: float | "LpAffineExpression") -> "LpAffineExpression":
        return _as_expr(self) + other

    def __radd__(self, other: float | "LpAffineExpression") -> "LpAffineExpression":
        return _as_expr(other) + self

    def __sub__(self, other: float | "LpAffineExpression") -> "LpAffineExpression":
        return _as_expr(self) - other

    def __rsub__(self, other: float | "LpAffineExpression") -> "LpAffineExpression":
        return _as_expr(other) - self

    def __le__(self, rhs: float | "LpAffineExpression" | "LpVariable") -> "LpConstraint":
        return _as_expr(self).__le__(rhs)

    def __ge__(self, rhs: float | "LpAffineExpression" | "LpVariable") -> "LpConstraint":
        return _as_expr(self).__ge__(rhs)

    def __eq__(self, rhs: float | "LpAffineExpression" | "LpVariable") -> "LpConstraint":
        return _as_expr(self).__eq__(rhs)


@dataclass
class LpAffineExpression:
    coeffs: Dict[LpVariable, float]
    constant: float = 0.0

    @staticmethod
    def from_any(value: float | LpVariable | "LpAffineExpression") -> "LpAffineExpression":
        if isinstance(value, LpAffineExpression):
            return value
        if isinstance(value, LpVariable):
            return LpAffineExpression({value: 1.0}, 0.0)
        return LpAffineExpression({}, float(value))

    def __add__(self, other: float | LpVariable | "LpAffineExpression") -> "LpAffineExpression":
        other_expr = LpAffineExpression.from_any(other)
        coeffs = dict(self.coeffs)
        for var, coef in other_expr.coeffs.items():
            coeffs[var] = coeffs.get(var, 0.0) + coef
        return LpAffineExpression(coeffs, self.constant + other_expr.constant)

    def __radd__(self, other: float | LpVariable | "LpAffineExpression") -> "LpAffineExpression":
        return self.__add__(other)

    def __sub__(self, other: float | LpVariable | "LpAffineExpression") -> "LpAffineExpression":
        other_expr = LpAffineExpression.from_any(other)
        coeffs = dict(self.coeffs)
        for var, coef in other_expr.coeffs.items():
            coeffs[var] = coeffs.get(var, 0.0) - coef
        return LpAffineExpression(coeffs, self.constant - other_expr.constant)

    def __rsub__(self, other: float | LpVariable | "LpAffineExpression") -> "LpAffineExpression":
        return LpAffineExpression.from_any(other).__sub__(self)

    def __mul__(self, other: float) -> "LpAffineExpression":
        coeffs = {var: coef * other for var, coef in self.coeffs.items()}
        return LpAffineExpression(coeffs, self.constant * other)

    def __rmul__(self, other: float) -> "LpAffineExpression":
        return self.__mul__(other)

    def _compare(self, rhs: float | LpVariable | "LpAffineExpression", sense: str) -> "LpConstraint":
        rhs_expr = LpAffineExpression.from_any(rhs)
        return LpConstraint(self - rhs_expr, sense, 0.0)

    def __le__(self, rhs: float | LpVariable | "LpAffineExpression") -> "LpConstraint":
        return self._compare(rhs, "<=")

    def __ge__(self, rhs: float | LpVariable | "LpAffineExpression") -> "LpConstraint":
        return self._compare(rhs, ">=")

    def __eq__(self, rhs: float | LpVariable | "LpAffineExpression") -> "LpConstraint":
        return self._compare(rhs, "==")


@dataclass
class LpConstraint:
    expr: LpAffineExpression
    sense: str
    rhs: float


class PULP_CBC_CMD:
    def __init__(self, msg: bool = False) -> None:
        self.msg = msg


class LpProblem:
    def __init__(self, name: str, sense: int) -> None:
        self.name = name
        self.sense = sense
        self.objective: LpAffineExpression | None = None
        self.constraints: List[LpConstraint] = []
        self.status = -1

    def __iadd__(self, other: LpAffineExpression | LpConstraint) -> "LpProblem":
        if isinstance(other, LpConstraint):
            self.constraints.append(other)
        else:
            self.objective = other
        return self

    def solve(self, solver: PULP_CBC_CMD | None = None) -> int:
        if self.objective is None:
            raise ValueError("Objective not defined")
        variables = list({var for constraint in self.constraints for var in constraint.expr.coeffs} | set(self.objective.coeffs))
        binaries = [v for v in variables if v.cat == "Binary"]
        continuous = [v for v in variables if v.cat != "Binary"]
        continuous_ids = {id(v) for v in continuous}
        best_obj = None
        best_solution: Dict[LpVariable, float] | None = None

        for combo in _enumerate_binary(binaries):
            fixed = {var: val for var, val in zip(binaries, combo)}
            lp_constraints = []
            for constraint in self.constraints:
                coeffs = {var: coef for var, coef in constraint.expr.coeffs.items() if id(var) in continuous_ids}
                fixed_contrib = sum(constraint.expr.coeffs.get(var, 0.0) * fixed[var] for var in fixed)
                rhs = constraint.rhs - constraint.expr.constant - fixed_contrib
                lp_constraints.append((coeffs, constraint.sense, rhs))
            lp_solution = _solve_lp(continuous, self.objective, lp_constraints, fixed)
            if lp_solution is None:
                continue
            obj_val = _evaluate_expr(self.objective, lp_solution)
            if best_obj is None or obj_val < best_obj:
                best_obj = obj_val
                best_solution = lp_solution
        if best_solution is None:
            self.status = -1
            return self.status
        for var, val in best_solution.items():
            var.value = val
        self.status = 1
        return self.status


def lpSum(values: Iterable[float | LpVariable | LpAffineExpression]) -> LpAffineExpression:
    expr = LpAffineExpression({}, 0.0)
    for val in values:
        expr = expr + LpAffineExpression.from_any(val)
    return expr


def value(var: LpVariable) -> float:
    return var.value


def _enumerate_binary(binaries: List[LpVariable]) -> Iterable[Tuple[int, ...]]:
    if not binaries:
        return [()]
    return itertools.product([0, 1], repeat=len(binaries))


def _evaluate_expr(expr: LpAffineExpression, solution: Dict[LpVariable, float]) -> float:
    total = expr.constant
    for var, coef in expr.coeffs.items():
        total += coef * solution[var]
    return total


def _solve_lp(
    variables: List[LpVariable],
    objective: LpAffineExpression,
    constraints: List[Tuple[Dict[LpVariable, float], str, float]],
    fixed: Dict[LpVariable, float],
) -> Dict[LpVariable, float] | None:
    n = len(variables)
    if n == 0:
        return fixed

    var_index = {var: idx for idx, var in enumerate(variables)}
    matrix = []
    rhs = []

    for coeffs, sense, bound in constraints:
        row = [0.0] * n
        for var, coef in coeffs.items():
            row[var_index[var]] = coef
        if sense == "==":
            matrix.append(row)
            rhs.append(bound)
            matrix.append([-val for val in row])
            rhs.append(-bound)
        elif sense == "<=":
            matrix.append(row)
            rhs.append(bound)
        elif sense == ">=":
            matrix.append([-val for val in row])
            rhs.append(-bound)

    for var in variables:
        idx = var_index[var]
        if var.lowBound is not None:
            row = [0.0] * n
            row[idx] = -1.0
            matrix.append(row)
            rhs.append(-var.lowBound)
        if var.upBound is not None:
            row = [0.0] * n
            row[idx] = 1.0
            matrix.append(row)
            rhs.append(var.upBound)

    best_val = None
    best_solution = None
    if n > 8:
        raise ValueError("Fallback LP solver supports up to 8 continuous variables.")
    for combo in itertools.combinations(range(len(matrix)), n):
        sub_matrix = [matrix[i][:] for i in combo]
        sub_rhs = [rhs[i] for i in combo]
        solution = _solve_linear_system(sub_matrix, sub_rhs)
        if solution is None:
            continue
        feasible = True
        for row, bound in zip(matrix, rhs):
            lhs = sum(row[i] * solution[i] for i in range(n))
            if lhs - bound > 1.0e-8:
                feasible = False
                break
        if not feasible:
            continue
        sol_map = {var: solution[var_index[var]] for var in variables} | fixed
        obj_val = _evaluate_expr(objective, sol_map)
        if best_val is None or obj_val < best_val:
            best_val = obj_val
            best_solution = sol_map
    return best_solution


def _solve_linear_system(matrix: List[List[float]], rhs: List[float]) -> List[float] | None:
    n = len(rhs)
    aug = [row[:] + [rhs_val] for row, rhs_val in zip(matrix, rhs)]
    for i in range(n):
        pivot = None
        for r in range(i, n):
            if abs(aug[r][i]) > 1.0e-12:
                pivot = r
                break
        if pivot is None:
            return None
        if pivot != i:
            aug[i], aug[pivot] = aug[pivot], aug[i]
        pivot_val = aug[i][i]
        for c in range(i, n + 1):
            aug[i][c] /= pivot_val
        for r in range(n):
            if r == i:
                continue
            factor = aug[r][i]
            for c in range(i, n + 1):
                aug[r][c] -= factor * aug[i][c]
    return [aug[i][n] for i in range(n)]
