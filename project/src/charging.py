from typing import Any, Dict, Tuple

from .assignment import aggregate_ev_energy_demand, aggregate_evtol_demand, compute_evtol_energy_demand

HAS_GUROBI = True
try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    HAS_GUROBI = False

LAST_SOLVER_USED = "unknown"


def _solve_shared_power_core(
    data: Dict[str, Any],
    times: list[int],
    e_dep: Dict[str, Dict[int, float]],
    ev_energy: Dict[str, Dict[int, float]],
) -> Tuple[
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, float],
]:
    if not HAS_GUROBI:
        return _solve_shared_power_core_heuristic(data, times, e_dep, ev_energy)

    stations = data["sets"]["stations"]
    delta_t = data["meta"]["delta_t"]
    station_params = data["parameters"]["stations"]
    prices = data["parameters"]["electricity_price"]
    storage_params = data["parameters"].get("vertiport_storage")

    if storage_params is None and e_dep:
        raise ValueError("Missing required key path: parameters.vertiport_storage")

    model = gp.Model("shared_power_inventory")
    model.setParam("OutputFlag", 0)

    B = {}
    P_vt = {}
    shed_vt = {}
    for dep in e_dep:
        if dep not in storage_params:
            raise ValueError(f"Missing required key path: parameters.vertiport_storage.{dep}")
        for t in times:
            B[dep, t] = model.addVar(
                lb=storage_params[dep]["B_min"],
                ub=storage_params[dep]["B_max"],
                name=f"B_{dep}_{t}",
            )
            P_vt[dep, t] = model.addVar(
                lb=0.0,
                ub=station_params[dep]["P_site"][t],
                name=f"P_vt_{dep}_{t}",
            )
            shed_vt[dep, t] = model.addVar(lb=0.0, name=f"shed_vt_{dep}_{t}")

    shed_ev = {}
    for s in stations:
        for t in times:
            shed_ev[s, t] = model.addVar(lb=0.0, name=f"shed_ev_{s}_{t}")

    penalty_ev = 1.0e6
    penalty_vt = 1.0e6
    model.setObjective(
        gp.quicksum(prices[dep][t] * P_vt[dep, t] * delta_t for dep in e_dep for t in times)
        + penalty_ev * gp.quicksum(shed_ev[s, t] * delta_t for s in stations for t in times)
        + penalty_vt * gp.quicksum(shed_vt[dep, t] for dep in e_dep for t in times),
        GRB.MINIMIZE,
    )

    for dep in e_dep:
        model.addConstr(B[dep, times[0]] == storage_params[dep]["B_init"], name=f"B_init_{dep}")
        eta = storage_params[dep]["eta_ch"]
        for idx, t in enumerate(times):
            if idx < len(times) - 1:
                model.addConstr(
                    B[dep, times[idx + 1]]
                    == B[dep, t] + eta * P_vt[dep, t] * delta_t - (e_dep[dep][t] - shed_vt[dep, t]),
                    name=f"B_bal_{dep}_{t}",
                )

    power_constraints = {}
    for s in stations:
        for t in times:
            p_ev_req = ev_energy.get(s, {}).get(t, 0.0) / delta_t
            p_vt_sum = gp.quicksum(P_vt[dep, t] for dep in e_dep if dep == s)
            power_constraints[s, t] = model.addConstr(
                p_vt_sum + p_ev_req - shed_ev[s, t] <= station_params[s]["P_site"][t],
                name=f"P_shared_{s}_{t}",
            )

    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise ValueError(f"Shared power LP did not solve to optimality: status={model.Status}")

    B_out = {dep: {t: float(B[dep, t].X) for t in times} for dep in e_dep}
    P_out = {dep: {t: float(P_vt[dep, t].X) for t in times} for dep in e_dep}
    shed_ev_out = {s: {t: float(shed_ev[s, t].X) for t in times} for s in stations}
    shed_vt_out = {dep: {t: float(shed_vt[dep, t].X) for t in times} for dep in e_dep}
    shadow_prices = {s: {t: float(power_constraints[s, t].Pi) for t in times} for s in stations}

    residuals = {"INV1": 0.0, "INV2": 0.0, "INV3": 0.0, "INV4": 0.0}
    for dep in e_dep:
        eta = storage_params[dep]["eta_ch"]
        for idx, t in enumerate(times):
            if idx < len(times) - 1:
                lhs = B_out[dep][times[idx + 1]]
                rhs = B_out[dep][t] + eta * P_out[dep][t] * delta_t - (e_dep[dep][t] - shed_vt_out[dep][t])
                residuals["INV1"] = max(residuals["INV1"], abs(lhs - rhs))
            residuals["INV2"] = max(
                residuals["INV2"],
                max(storage_params[dep]["B_min"] - B_out[dep][t], B_out[dep][t] - storage_params[dep]["B_max"], 0.0),
            )
    for s in stations:
        for t in times:
            p_ev_req = ev_energy.get(s, {}).get(t, 0.0) / delta_t
            p_vt_sum = sum(P_out.get(dep, {}).get(t, 0.0) for dep in e_dep if dep == s)
            residuals["INV3"] = max(
                residuals["INV3"],
                max(0.0, p_vt_sum + p_ev_req - shed_ev_out[s][t] - station_params[s]["P_site"][t]),
            )
            residuals["INV4"] = max(residuals["INV4"], max(0.0, shed_ev_out[s][t]))

    return B_out, P_out, shed_ev_out, shed_vt_out, shadow_prices, residuals


def _solve_shared_power_core_heuristic(
    data: Dict[str, Any],
    times: list[int],
    e_dep: Dict[str, Dict[int, float]],
    ev_energy: Dict[str, Dict[int, float]],
) -> Tuple[
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, float],
]:
    stations = data["sets"]["stations"]
    delta_t = data["meta"]["delta_t"]
    station_params = data["parameters"]["stations"]
    storage_params = data["parameters"].get("vertiport_storage")
    kappa = float(data.get("config", {}).get("shadow_kappa", 2.0))

    if storage_params is None and e_dep:
        raise ValueError("Missing required key path: parameters.vertiport_storage")

    B_out = {dep: {t: 0.0 for t in times} for dep in e_dep}
    P_out = {dep: {t: 0.0 for t in times} for dep in e_dep}
    shed_ev_out = {s: {t: 0.0 for t in times} for s in stations}
    shed_vt_out = {dep: {t: 0.0 for t in times} for dep in e_dep}
    shadow_prices = {s: {t: 0.0 for t in times} for s in stations}

    available_power = {s: {t: station_params[s]["P_site"][t] for t in times} for s in stations}
    for s in stations:
        for t in times:
            p_ev_req = ev_energy.get(s, {}).get(t, 0.0) / delta_t
            available = station_params[s]["P_site"][t] - p_ev_req
            if available < 0.0:
                shed_ev_out[s][t] = -available
                available = 0.0
            available_power[s][t] = available

    for s in stations:
        for t in times:
            p_ev_req = ev_energy.get(s, {}).get(t, 0.0) / delta_t
            p_vt_req = 0.0
            if s in e_dep:
                eta = storage_params[s]["eta_ch"]
                p_vt_req = max(0.0, e_dep[s][t] / (eta * delta_t)) if eta > 0 else 0.0
            slack = station_params[s]["P_site"][t] - (p_ev_req + p_vt_req)
            if slack < 0.0 and station_params[s]["P_site"][t] > 0.0:
                shadow_prices[s][t] = kappa * (-slack / station_params[s]["P_site"][t])

    for dep in e_dep:
        B_out[dep][times[0]] = storage_params[dep]["B_init"]
        eta = storage_params[dep]["eta_ch"]
        for idx, t in enumerate(times):
            available = available_power[dep][t]
            desired = max(0.0, e_dep[dep][t] / (eta * delta_t))
            P_out[dep][t] = min(available, desired)
            available_power[dep][t] -= P_out[dep][t]
            next_B = B_out[dep][t] + eta * P_out[dep][t] * delta_t - e_dep[dep][t]
            if next_B < storage_params[dep]["B_min"]:
                shortfall = storage_params[dep]["B_min"] - next_B
                shed_vt_out[dep][t] = shortfall
                next_B += shortfall
            if idx < len(times) - 1:
                B_out[dep][times[idx + 1]] = max(
                    storage_params[dep]["B_min"], min(storage_params[dep]["B_max"], next_B)
                )

    residuals = {"INV1": 0.0, "INV2": 0.0, "INV3": 0.0, "INV4": 0.0}
    for dep in e_dep:
        eta = storage_params[dep]["eta_ch"]
        for idx, t in enumerate(times):
            if idx < len(times) - 1:
                lhs = B_out[dep][times[idx + 1]]
                rhs = B_out[dep][t] + eta * P_out[dep][t] * delta_t - (e_dep[dep][t] - shed_vt_out[dep][t])
                residuals["INV1"] = max(residuals["INV1"], abs(lhs - rhs))
            residuals["INV2"] = max(
                residuals["INV2"],
                max(storage_params[dep]["B_min"] - B_out[dep][t], B_out[dep][t] - storage_params[dep]["B_max"], 0.0),
            )
    for s in stations:
        for t in times:
            p_ev_req = ev_energy.get(s, {}).get(t, 0.0) / delta_t
            p_vt_sum = sum(P_out.get(dep, {}).get(t, 0.0) for dep in e_dep if dep == s)
            residuals["INV3"] = max(
                residuals["INV3"],
                max(0.0, p_vt_sum + p_ev_req - shed_ev_out[s][t] - station_params[s]["P_site"][t]),
            )
            residuals["INV4"] = max(residuals["INV4"], max(0.0, shed_ev_out[s][t]))

    return B_out, P_out, shed_ev_out, shed_vt_out, shadow_prices, residuals


def solve_shared_power_lp(
    data: Dict[str, Any],
    itineraries: list[Dict[str, Any]],
    flows: Dict[str, Dict[str, Dict[int, float]]],
    times: list[int],
) -> Tuple[
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, float],
]:
    d_vt_route = aggregate_evtol_demand(flows, itineraries, times)
    e_dep = compute_evtol_energy_demand(d_vt_route, itineraries, times)
    ev_energy = aggregate_ev_energy_demand(itineraries, flows, times)
    return _solve_shared_power_core(data, times, e_dep, ev_energy)


def solve_shared_power_inventory_lp(
    data: Dict[str, Any],
    e_dep: Dict[str, Dict[int, float]],
    ev_energy: Dict[str, Dict[int, float]],
) -> Tuple[
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, float],
]:
    times = data["sets"]["time"]
    return _solve_shared_power_core(data, times, e_dep, ev_energy)


def solve_charging(
    data: Dict[str, Any],
    e_dep: Dict[str, Dict[int, float]] | None = None,
    d_dep: Dict[str, Dict[int, float]] | None = None,
) -> Tuple[
    Dict[str, Dict[int, float]],
    Dict[str, Dict[str, Dict[int, float]]],
    Dict[str, Dict[str, Dict[int, int]]],
    Dict[str, float],
    Dict[str, Dict[int, float]] | None,
    Dict[str, Dict[int, float]] | None,
    Dict[str, float] | None,
    Dict[str, Dict[int, float]] | None,
]:
    global LAST_SOLVER_USED

    if not HAS_GUROBI:
        return _solve_charging_pulp(data, e_dep, d_dep)

    LAST_SOLVER_USED = "gurobi"
    times = data["sets"]["time"]
    vehicles = data["sets"]["vehicles"]
    stations = data["sets"]["stations"]
    delta_t = data["meta"]["delta_t"]
    charging_params = data["parameters"]["charging"]
    avail = data["parameters"]["avail"]
    e_fly_or_drive = data["parameters"]["e_fly_or_drive"]
    station_params = data["parameters"]["stations"]
    prices = data["parameters"]["electricity_price"]
    storage_params = data["parameters"].get("vertiport_storage")

    include_vt = e_dep is not None and d_dep is not None and any(
        val > 0.0 for dep in (e_dep or {}) for val in e_dep[dep].values()
    )
    if include_vt and storage_params is None:
        raise ValueError("Missing required key path: parameters.vertiport_storage")

    model = gp.Model("charging")
    model.setParam("OutputFlag", 0)

    E = {}
    p_ch = {}
    y = {}
    for m in vehicles:
        for t in times:
            E[m, t] = model.addVar(
                lb=charging_params[m]["E_min"],
                ub=charging_params[m]["E_max"],
                name=f"E_{m}_{t}",
            )
        for s in stations:
            for t in times:
                p_ch[m, s, t] = model.addVar(lb=0.0, ub=charging_params[m]["P_max"], name=f"p_{m}_{s}_{t}")
                y[m, s, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{m}_{s}_{t}")

    B_vt = {}
    P_vt = {}
    if include_vt:
        for dep in e_dep:
            if dep not in storage_params:
                raise ValueError(f"Missing required key path: parameters.vertiport_storage.{dep}")
            for t in times:
                B_vt[dep, t] = model.addVar(
                    lb=storage_params[dep]["B_min"],
                    ub=storage_params[dep]["B_max"],
                    name=f"B_vt_{dep}_{t}",
                )
                P_vt[dep, t] = model.addVar(
                    lb=0.0,
                    ub=station_params[dep]["P_site"][t],
                    name=f"P_vt_{dep}_{t}",
                )

    model.setObjective(
        gp.quicksum(prices[s][t] * p_ch[m, s, t] * delta_t for m in vehicles for s in stations for t in times)
        + gp.quicksum(prices[dep][t] * P_vt[dep, t] * delta_t for dep in e_dep or {} for t in times),
        GRB.MINIMIZE,
    )

    for m in vehicles:
        model.addConstr(E[m, times[0]] == charging_params[m]["E_init"], name=f"E_init_{m}")
        eta = charging_params[m]["eta_ch"]
        E_res = charging_params[m]["E_res"]
        for idx, t in enumerate(times):
            model.addConstr(E[m, t] >= E_res, name=f"E_res_{m}_{t}")
            if idx < len(times) - 1:
                model.addConstr(
                    E[m, times[idx + 1]]
                    == E[m, t] - e_fly_or_drive[m][t] + gp.quicksum(eta * p_ch[m, s, t] * delta_t for s in stations),
                    name=f"E_dyn_{m}_{t}",
                )
            model.addConstr(gp.quicksum(y[m, s, t] for s in stations) <= 1, name=f"y_one_{m}_{t}")
            for s in stations:
                model.addConstr(p_ch[m, s, t] <= charging_params[m]["P_max"] * y[m, s, t], name=f"p_link_{m}_{s}_{t}")
                model.addConstr(y[m, s, t] <= avail[m][s][t], name=f"y_avail_{m}_{s}_{t}")

    for s in stations:
        cap = station_params[s]["cap_stall"]
        for t in times:
            vt_power = gp.quicksum(P_vt[dep, t] for dep in e_dep or {} if dep == s) if include_vt else 0.0
            model.addConstr(
                gp.quicksum(p_ch[m, s, t] for m in vehicles) + vt_power <= station_params[s]["P_site"][t],
                name=f"P_site_{s}_{t}",
            )
            model.addConstr(gp.quicksum(y[m, s, t] for m in vehicles) <= cap, name=f"cap_{s}_{t}")

    if include_vt:
        for dep in e_dep:
            model.addConstr(B_vt[dep, times[0]] == storage_params[dep]["B_init"], name=f"B_init_{dep}")
            eta = storage_params[dep]["eta_ch"]
            for idx, t in enumerate(times):
                if idx < len(times) - 1:
                    model.addConstr(
                        B_vt[dep, times[idx + 1]]
                        == B_vt[dep, t] + eta * P_vt[dep, t] * delta_t - e_dep[dep][t],
                        name=f"B_bal_{dep}_{t}",
                    )

    model.optimize()
    if model.Status != GRB.OPTIMAL:
        if model.Status == GRB.INFEASIBLE:
            model.computeIIS()
            model.write("charging.ilp")
        raise ValueError(f"Charging optimization did not solve to optimality: status={model.Status}")

    E_out: Dict[str, Dict[int, float]] = {m: {t: float(E[m, t].X) for t in times} for m in vehicles}
    p_out: Dict[str, Dict[str, Dict[int, float]]] = {
        m: {s: {t: float(p_ch[m, s, t].X) for t in times} for s in stations} for m in vehicles
    }
    y_out: Dict[str, Dict[str, Dict[int, int]]] = {
        m: {s: {t: int(round(y[m, s, t].X)) for t in times} for s in stations} for m in vehicles
    }

    B_out = None
    P_out = None
    inv_residuals = None
    shadow_prices = None
    if include_vt:
        B_out = {dep: {t: float(B_vt[dep, t].X) for t in times} for dep in e_dep}
        P_out = {dep: {t: float(P_vt[dep, t].X) for t in times} for dep in e_dep}
        inv_residuals = compute_inventory_residuals(data, e_dep, B_out, P_out)

    residuals = compute_charging_residuals(data, E_out, p_out, y_out, P_out)

    return E_out, p_out, y_out, residuals, B_out, P_out, inv_residuals, shadow_prices


def _solve_charging_pulp(
    data: Dict[str, Any],
    e_dep: Dict[str, Dict[int, float]] | None = None,
    d_dep: Dict[str, Dict[int, float]] | None = None,
) -> Tuple[
    Dict[str, Dict[int, float]],
    Dict[str, Dict[str, Dict[int, float]]],
    Dict[str, Dict[str, Dict[int, int]]],
    Dict[str, float],
    Dict[str, Dict[int, float]] | None,
    Dict[str, Dict[int, float]] | None,
    Dict[str, float] | None,
    Dict[str, Dict[int, float]] | None,
]:
    global LAST_SOLVER_USED
    LAST_SOLVER_USED = "pulp"

    times = data["sets"]["time"]
    vehicles = data["sets"]["vehicles"]
    stations = data["sets"]["stations"]
    delta_t = data["meta"]["delta_t"]
    charging_params = data["parameters"]["charging"]
    avail = data["parameters"]["avail"]
    e_fly_or_drive = data["parameters"]["e_fly_or_drive"]
    station_params = data["parameters"]["stations"]
    prices = data["parameters"]["electricity_price"]
    storage_params = data["parameters"].get("vertiport_storage")

    include_vt = e_dep is not None and d_dep is not None and any(
        val > 0.0 for dep in (e_dep or {}) for val in e_dep[dep].values()
    )
    if include_vt and storage_params is None:
        raise ValueError("Missing required key path: parameters.vertiport_storage")

    E_out = {m: {t: 0.0 for t in times} for m in vehicles}
    p_out = {m: {s: {t: 0.0 for t in times} for s in stations} for m in vehicles}
    y_out = {m: {s: {t: 0 for t in times} for s in stations} for m in vehicles}

    B_out = None
    P_out = None
    inv_residuals = None
    shadow_prices = None

    P_vt_remaining = {s: {t: station_params[s]["P_site"][t] for t in times} for s in stations}
    if include_vt:
        B_out = {dep: {t: 0.0 for t in times} for dep in e_dep}
        P_out = {dep: {t: 0.0 for t in times} for dep in e_dep}
        for dep in e_dep:
            B_out[dep][times[0]] = storage_params[dep]["B_init"]
            eta = storage_params[dep]["eta_ch"]
            for idx, t in enumerate(times):
                desired = max(0.0, e_dep[dep][t] / (eta * delta_t))
                P_out[dep][t] = min(P_vt_remaining[dep][t], desired)
                P_vt_remaining[dep][t] -= P_out[dep][t]
                next_B = B_out[dep][t] + eta * P_out[dep][t] * delta_t - e_dep[dep][t]
                if idx < len(times) - 1:
                    B_out[dep][times[idx + 1]] = max(storage_params[dep]["B_min"], min(storage_params[dep]["B_max"], next_B))
        inv_residuals = compute_inventory_residuals(data, e_dep, B_out, P_out)

    stall_remaining = {s: {t: station_params[s]["cap_stall"] for t in times} for s in stations}
    for m in vehicles:
        E_out[m][times[0]] = charging_params[m]["E_init"]
        eta = charging_params[m]["eta_ch"]
        for idx, t in enumerate(times):
            if idx < len(times) - 1:
                projected = E_out[m][t] - e_fly_or_drive[m][t]
                needed = max(0.0, charging_params[m]["E_res"] - projected)
                required_power = needed / (eta * delta_t) if eta > 0 else 0.0
                if required_power > 0.0:
                    candidate_stations = [
                        s for s in stations if avail[m][s][t] == 1 and stall_remaining[s][t] > 0
                    ]
                    if candidate_stations:
                        s_choice = max(candidate_stations, key=lambda s: P_vt_remaining[s][t])
                        charge = min(required_power, charging_params[m]["P_max"], P_vt_remaining[s_choice][t])
                        if charge > 0.0:
                            p_out[m][s_choice][t] = charge
                            y_out[m][s_choice][t] = 1
                            P_vt_remaining[s_choice][t] -= charge
                            stall_remaining[s_choice][t] -= 1
                            projected += eta * charge * delta_t
                E_out[m][times[idx + 1]] = projected

    residuals = compute_charging_residuals(data, E_out, p_out, y_out, P_out)

    return E_out, p_out, y_out, residuals, B_out, P_out, inv_residuals, shadow_prices


def compute_charging_residuals(
    data: Dict[str, Any],
    E: Dict[str, Dict[int, float]],
    p_ch: Dict[str, Dict[str, Dict[int, float]]],
    y: Dict[str, Dict[str, Dict[int, int]]],
    P_vt: Dict[str, Dict[int, float]] | None = None,
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
        "C18": 0.0,
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
            residuals["C18"] = max(
                residuals["C18"],
                max(0.0, sum(y[m][s][t] for s in stations) - 1.0),
            )
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
            if P_vt and s in P_vt:
                total_power += P_vt[s][t]
            total_y = sum(y[m][s][t] for m in vehicles)
            P_site = station_params[s]["P_site"][t]
            residuals["C15"] = max(residuals["C15"], max(0.0, total_power - P_site))
            residuals["C16"] = max(residuals["C16"], max(0.0, total_y - cap))
    return residuals


def compute_inventory_residuals(
    data: Dict[str, Any],
    e_dep: Dict[str, Dict[int, float]],
    B_vt: Dict[str, Dict[int, float]],
    P_vt: Dict[str, Dict[int, float]],
) -> Dict[str, float]:
    times = data["sets"]["time"]
    delta_t = data["meta"]["delta_t"]
    station_params = data["parameters"]["stations"]
    storage_params = data["parameters"]["vertiport_storage"]

    residuals = {"INV1": 0.0, "INV2": 0.0, "INV3": 0.0}
    for dep in e_dep:
        eta = storage_params[dep]["eta_ch"]
        for idx, t in enumerate(times):
            if idx < len(times) - 1:
                lhs = B_vt[dep][times[idx + 1]]
                rhs = B_vt[dep][t] + eta * P_vt[dep][t] * delta_t - e_dep[dep][t]
                residuals["INV1"] = max(residuals["INV1"], abs(lhs - rhs))
            residuals["INV2"] = max(
                residuals["INV2"],
                max(storage_params[dep]["B_min"] - B_vt[dep][t], B_vt[dep][t] - storage_params[dep]["B_max"], 0.0),
            )
            residuals["INV3"] = max(
                residuals["INV3"],
                max(0.0, P_vt[dep][t] - station_params[dep]["P_site"][t]),
            )
    return residuals
