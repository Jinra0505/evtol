from typing import Any, Dict, Tuple

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as exc:
    raise ImportError(
        "gurobipy is required for charging optimization. Please install Gurobi and set up a license."
    ) from exc


def solve_evtol_inventory(
    data: Dict[str, Any],
    e_dep: Dict[str, Dict[int, float]],
    d_dep: Dict[str, Dict[int, float]],
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[int, float]], Dict[str, float], Dict[str, Dict[int, float]]]:
    times = data["sets"]["time"]
    delta_t = data["meta"]["delta_t"]
    station_params = data["parameters"]["stations"]
    prices = data["parameters"]["electricity_price"]
    storage_params = data["parameters"].get("vertiport_storage")

    if storage_params is None:
        raise ValueError("Missing required key path: parameters.vertiport_storage")

    model = gp.Model("evtol_inventory")
    model.setParam("OutputFlag", 0)

    B = {}
    P = {}
    for dep in e_dep:
        if dep not in storage_params:
            raise ValueError(f"Missing required key path: parameters.vertiport_storage.{dep}")
        for t in times:
            B[dep, t] = model.addVar(
                lb=storage_params[dep]["B_min"],
                ub=storage_params[dep]["B_max"],
                name=f"B_{dep}_{t}",
            )
            P[dep, t] = model.addVar(
                lb=0.0,
                ub=station_params[dep]["P_site"][t],
                name=f"P_{dep}_{t}",
            )

    model.setObjective(
        gp.quicksum(prices[dep][t] * P[dep, t] * delta_t for dep in e_dep for t in times), GRB.MINIMIZE
    )

    balance_constraints = {}
    for dep in e_dep:
        model.addConstr(B[dep, times[0]] == storage_params[dep]["B_init"], name=f"B_init_{dep}")
        eta = storage_params[dep]["eta_ch"]
        for idx, t in enumerate(times):
            if idx < len(times) - 1:
                balance_constraints[dep, t] = model.addConstr(
                    B[dep, times[idx + 1]]
                    == B[dep, t] + eta * P[dep, t] * delta_t - e_dep[dep][t],
                    name=f"B_bal_{dep}_{t}",
                )

    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise ValueError(f"eVTOL inventory optimization did not solve to optimality: status={model.Status}")

    B_out = {dep: {t: float(B[dep, t].X) for t in times} for dep in e_dep}
    P_out = {dep: {t: float(P[dep, t].X) for t in times} for dep in e_dep}
    shadow_prices = {dep: {t: float(balance_constraints[dep, t].Pi) for t in times[:-1]} for dep in e_dep}

    residuals = {"INV1": 0.0, "INV2": 0.0, "INV3": 0.0, "INV4": 0.0}
    for dep in e_dep:
        eta = storage_params[dep]["eta_ch"]
        for idx, t in enumerate(times):
            if idx < len(times) - 1:
                lhs = B_out[dep][times[idx + 1]]
                rhs = B_out[dep][t] + eta * P_out[dep][t] * delta_t - e_dep[dep][t]
                residuals["INV1"] = max(residuals["INV1"], abs(lhs - rhs))
            residuals["INV2"] = max(
                residuals["INV2"],
                max(storage_params[dep]["B_min"] - B_out[dep][t], B_out[dep][t] - storage_params[dep]["B_max"], 0.0),
            )
            residuals["INV3"] = max(
                residuals["INV3"],
                max(0.0, P_out[dep][t] - station_params[dep]["P_site"][t]),
            )
    cap_pax = data["parameters"].get("vertiport_cap_pax")
    if cap_pax:
        for dep, time_map in d_dep.items():
            for t in times:
                residuals["INV4"] = max(residuals["INV4"], max(0.0, d_dep[dep][t] - cap_pax[dep][t]))

    return B_out, P_out, residuals, shadow_prices


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
    times = data["sets"]["time"]
    vehicles = data["sets"]["vehicles"]
    stations = data["sets"]["stations"]
    delta_t = data["meta"]["delta_t"]
    charging_params = data["parameters"]["charging"]
    avail = data["parameters"]["avail"]
    e_fly_or_drive = data["parameters"]["e_fly_or_drive"]
    station_params = data["parameters"]["stations"]
    prices = data["parameters"]["electricity_price"]

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

    model.setObjective(
        gp.quicksum(prices[s][t] * p_ch[m, s, t] * delta_t for m in vehicles for s in stations for t in times),
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
            model.addConstr(
                gp.quicksum(p_ch[m, s, t] for m in vehicles) <= station_params[s]["P_site"][t],
                name=f"P_site_{s}_{t}",
            )
            model.addConstr(gp.quicksum(y[m, s, t] for m in vehicles) <= cap, name=f"cap_{s}_{t}")

    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise ValueError(f"Charging optimization did not solve to optimality: status={model.Status}")

    E_out: Dict[str, Dict[int, float]] = {m: {t: float(E[m, t].X) for t in times} for m in vehicles}
    p_out: Dict[str, Dict[str, Dict[int, float]]] = {
        m: {s: {t: float(p_ch[m, s, t].X) for t in times} for s in stations} for m in vehicles
    }
    y_out: Dict[str, Dict[str, Dict[int, int]]] = {
        m: {s: {t: int(round(y[m, s, t].X)) for t in times} for s in stations} for m in vehicles
    }

    residuals = compute_charging_residuals(data, E_out, p_out, y_out)

    B_out = None
    P_out = None
    inv_residuals = None
    shadow_prices = None
    if e_dep and d_dep:
        if any(val > 0.0 for dep in e_dep for val in e_dep[dep].values()):
            B_out, P_out, inv_residuals, shadow_prices = solve_evtol_inventory(data, e_dep, d_dep)

    return E_out, p_out, y_out, residuals, B_out, P_out, inv_residuals, shadow_prices


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
            total_y = sum(y[m][s][t] for m in vehicles)
            P_site = station_params[s]["P_site"][t]
            residuals["C15"] = max(residuals["C15"], max(0.0, total_power - P_site))
            residuals["C16"] = max(residuals["C16"], max(0.0, total_y - cap))
    return residuals
