import importlib
import importlib.util
import itertools
from typing import Any, Dict, List, Tuple


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


def _solve_lp_vertex(c: List[float], constraints: List[Tuple[List[float], float]], ub: List[float]) -> Tuple[List[float], float] | None:
    n = len(c)
    all_constraints = constraints[:]
    for i in range(n):
        upper = ub[i]
        all_constraints.append(([1.0 if j == i else 0.0 for j in range(n)], upper))
        all_constraints.append(([-1.0 if j == i else 0.0 for j in range(n)], 0.0))

    best_val = None
    best_x = None
    indices = list(range(len(all_constraints)))
    for combo in itertools.combinations(indices, n):
        matrix = [all_constraints[idx][0] for idx in combo]
        rhs = [all_constraints[idx][1] for idx in combo]
        solution = _solve_linear_system(matrix, rhs)
        if solution is None:
            continue
        feasible = True
        for coeffs, bound in all_constraints:
            lhs = sum(coeffs[i] * solution[i] for i in range(n))
            if lhs - bound > 1.0e-8:
                feasible = False
                break
        if not feasible:
            continue
        obj = sum(c[i] * solution[i] for i in range(n))
        if best_val is None or obj < best_val:
            best_val = obj
            best_x = solution
    if best_x is None:
        return None
    return best_x, float(best_val)


def _fallback_milp(
    data: Dict[str, Any],
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[str, Dict[int, float]]], Dict[str, Dict[str, Dict[int, int]]]]:
    times = data["sets"]["time"]
    vehicles = data["sets"]["vehicles"]
    stations = data["sets"]["stations"]
    delta_t = data["meta"]["delta_t"]
    charging_params = data["parameters"]["charging"]
    avail = data["parameters"]["avail"]
    e_fly_or_drive = data["parameters"]["e_fly_or_drive"]
    station_params = data["parameters"]["stations"]
    prices = data["parameters"]["electricity_price"]

    y_vars = [(m, s, t) for m in vehicles for s in stations for t in times]
    if len(y_vars) > 12:
        raise ValueError("Fallback MILP supports at most 12 binary variables.")

    best_solution = None
    best_cost = None

    for y_combo in itertools.product([0, 1], repeat=len(y_vars)):
        y_map = {key: val for key, val in zip(y_vars, y_combo)}
        feasible = True
        for m, s, t in y_vars:
            if y_map[(m, s, t)] > avail[m][s][t]:
                feasible = False
                break
        if not feasible:
            continue
        for s in stations:
            cap = station_params[s]["cap_stall"]
            for t in times:
                if sum(y_map[(m, s, t)] for m in vehicles) > cap:
                    feasible = False
                    break
            if not feasible:
                break
        if not feasible:
            continue

        p_vars = [(m, s, t) for m in vehicles for s in stations for t in times]
        n = len(p_vars)
        if n == 0:
            continue
        c = [prices[s][t] * delta_t for (m, s, t) in p_vars]
        ub = [charging_params[m]["P_max"] * y_map[(m, s, t)] for (m, s, t) in p_vars]
        constraints: List[Tuple[List[float], float]] = []

        for s in stations:
            for t in times:
                coeffs = [1.0 if p_vars[i][1] == s and p_vars[i][2] == t else 0.0 for i in range(n)]
                constraints.append((coeffs, station_params[s]["P_site"][t]))

        for m in vehicles:
            E_init = charging_params[m]["E_init"]
            E_min = charging_params[m]["E_min"]
            E_max = charging_params[m]["E_max"]
            E_res = charging_params[m]["E_res"]
            eta = charging_params[m]["eta_ch"]
            for idx, t in enumerate(times):
                if idx == 0:
                    if not (E_min <= E_init <= E_max and E_init >= E_res):
                        feasible = False
                        break
                    continue
                coeffs = [0.0 for _ in range(n)]
                for i, (m_i, s_i, t_i) in enumerate(p_vars):
                    if m_i == m and times.index(t_i) < idx:
                        coeffs[i] += eta * delta_t
                energy_base = E_init - sum(e_fly_or_drive[m][times[j]] for j in range(idx))
                lower = max(E_min, E_res) - energy_base
                upper = E_max - energy_base
                constraints.append(([-c_val for c_val in coeffs], -lower))
                constraints.append((coeffs, upper))
            if not feasible:
                break
        if not feasible:
            continue

        if n > 6:
            raise ValueError("Fallback LP solver supports at most 6 charging variables.")
        result = _solve_lp_vertex(c, constraints, ub)
        if result is None:
            continue
        p_solution, cost = result
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_solution = (y_map, p_solution)

    if best_solution is None:
        raise ValueError("Fallback MILP found no feasible solution.")

    y_map, p_solution = best_solution
    E_out: Dict[str, Dict[int, float]] = {m: {} for m in vehicles}
    p_out: Dict[str, Dict[str, Dict[int, float]]] = {m: {s: {} for s in stations} for m in vehicles}
    y_out: Dict[str, Dict[str, Dict[int, int]]] = {m: {s: {} for s in stations} for m in vehicles}

    p_vars = [(m, s, t) for m in vehicles for s in stations for t in times]
    for idx, (m, s, t) in enumerate(p_vars):
        p_out[m][s][t] = p_solution[idx]
        y_out[m][s][t] = y_map[(m, s, t)]

    for m in vehicles:
        E_init = charging_params[m]["E_init"]
        E_out[m][times[0]] = E_init
        eta = charging_params[m]["eta_ch"]
        for idx, t in enumerate(times):
            if idx == len(times) - 1:
                continue
            charge = sum(eta * p_out[m][s][t] * delta_t for s in stations)
            E_out[m][times[idx + 1]] = E_out[m][t] - e_fly_or_drive[m][t] + charge
    return E_out, p_out, y_out


def solve_charging(
    data: Dict[str, Any],
) -> Tuple[
    Dict[str, Dict[int, float]],
    Dict[str, Dict[str, Dict[int, float]]],
    Dict[str, Dict[str, Dict[int, int]]],
    Dict[str, float],
]:
    if importlib.util.find_spec("pulp") is not None:
        pulp = importlib.import_module("pulp")
        times = data["sets"]["time"]
        vehicles = data["sets"]["vehicles"]
        stations = data["sets"]["stations"]
        delta_t = data["meta"]["delta_t"]
        charging_params = data["parameters"]["charging"]
        avail = data["parameters"]["avail"]
        e_fly_or_drive = data["parameters"]["e_fly_or_drive"]
        station_params = data["parameters"]["stations"]
        prices = data["parameters"]["electricity_price"]

        model = pulp.LpProblem("charging", pulp.LpMinimize)

        E = {
            m: {
                t: pulp.LpVariable(
                    f"E_{m}_{t}", lowBound=charging_params[m]["E_min"], upBound=charging_params[m]["E_max"]
                )
                for t in times
            }
            for m in vehicles
        }
        p_ch = {
            m: {
                s: {
                    t: pulp.LpVariable(f"p_{m}_{s}_{t}", lowBound=0.0, upBound=charging_params[m]["P_max"])
                    for t in times
                }
                for s in stations
            }
            for m in vehicles
        }
        y = {
            m: {
                s: {
                    t: pulp.LpVariable(f"y_{m}_{s}_{t}", lowBound=0.0, upBound=1.0, cat="Binary")
                    for t in times
                }
                for s in stations
            }
            for m in vehicles
        }

        model += pulp.lpSum(
            prices[s][t] * p_ch[m][s][t] * delta_t for m in vehicles for s in stations for t in times
        )

        for m in vehicles:
            model += E[m][times[0]] == charging_params[m]["E_init"]
            eta = charging_params[m]["eta_ch"]
            E_res = charging_params[m]["E_res"]
            for idx, t in enumerate(times):
                model += E[m][t] >= E_res
                if idx < len(times) - 1:
                    model += (
                        E[m][times[idx + 1]]
                        == E[m][t]
                        - e_fly_or_drive[m][t]
                        + pulp.lpSum(eta * p_ch[m][s][t] * delta_t for s in stations)
                    )
                for s in stations:
                    model += p_ch[m][s][t] <= charging_params[m]["P_max"] * y[m][s][t]
                    model += y[m][s][t] <= avail[m][s][t]

        for s in stations:
            cap = station_params[s]["cap_stall"]
            for t in times:
                model += pulp.lpSum(p_ch[m][s][t] for m in vehicles) <= station_params[s]["P_site"][t]
                model += pulp.lpSum(y[m][s][t] for m in vehicles) <= cap

        model.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[model.status] != "Optimal":
            raise ValueError(f"Charging optimization did not solve to optimality: {pulp.LpStatus[model.status]}")

        E_out: Dict[str, Dict[int, float]] = {m: {t: float(pulp.value(E[m][t])) for t in times} for m in vehicles}
        p_out: Dict[str, Dict[str, Dict[int, float]]] = {
            m: {s: {t: float(pulp.value(p_ch[m][s][t])) for t in times} for s in stations} for m in vehicles
        }
        y_out: Dict[str, Dict[str, Dict[int, int]]] = {
            m: {s: {t: int(round(pulp.value(y[m][s][t]))) for t in times} for s in stations} for m in vehicles
        }
    else:
        E_out, p_out, y_out = _fallback_milp(data)

    residuals = compute_charging_residuals(data, E_out, p_out, y_out)
    return E_out, p_out, y_out, residuals


def compute_charging_residuals(
    data: Dict[str, Any],
    E: Dict[str, Dict[int, float]],
    p_ch: Dict[str, Dict[str, Dict[int, float]]],
    y: Dict[str, Dict[str, Dict[int, int]]],
) -> Dict[str, float]:
    times = data["sets"]["time"]
    vehicles = data["sets"]["vehicles"]
    stations = data["sets"]["stations"]
    delta_t = data["meta"]["delta_t"]
    charging_params = data["parameters"]["charging"]
    avail = data["parameters"]["avail"]
    e_fly_or_drive = data["parameters"]["e_fly_or_drive"]
    station_params = data["parameters"]["stations"]

    residuals = {
        "C12": 0.0,
        "C13": 0.0,
        "C14": 0.0,
        "C15": 0.0,
        "C16": 0.0,
        "C17": 0.0,
    }

    for m in vehicles:
        E_min = charging_params[m]["E_min"]
        E_max = charging_params[m]["E_max"]
        E_res = charging_params[m]["E_res"]
        eta = charging_params[m]["eta_ch"]
        for idx, t in enumerate(times):
            if idx < len(times) - 1:
                lhs = E[m][times[idx + 1]]
                rhs = E[m][t] - e_fly_or_drive[m][t]
                charge_term = sum(eta * p_ch[m][s][t] * delta_t for s in stations)
                rhs += charge_term
                residuals["C12"] = max(residuals["C12"], abs(lhs - rhs))
            residuals["C13"] = max(residuals["C13"], max(E_min - E[m][t], E[m][t] - E_max, 0.0))
            residuals["C17"] = max(residuals["C17"], max(E_res - E[m][t], 0.0))
            for s in stations:
                P_max = charging_params[m]["P_max"]
                residuals["C14"] = max(
                    residuals["C14"],
                    max(0.0, p_ch[m][s][t] - P_max * y[m][s][t], y[m][s][t] - avail[m][s][t]),
                )
    for s in stations:
        cap = station_params[s]["cap_stall"]
        for t in times:
            total_power = sum(p_ch[m][s][t] for m in vehicles)
            total_y = sum(y[m][s][t] for m in vehicles)
            P_site = station_params[s]["P_site"][t]
            residuals["C15"] = max(residuals["C15"], max(0.0, total_power - P_site))
            residuals["C16"] = max(residuals["C16"], max(0.0, total_y - cap))
    return residuals
