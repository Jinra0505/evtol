from typing import Any, Dict, Tuple

from . import pulp


def solve_evtol_inventory(
    data: Dict[str, Any],
    e_dep: Dict[str, Dict[int, float]],
    d_dep: Dict[str, Dict[int, float]],
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[int, float]], Dict[str, float]]:
    times = data["sets"]["time"]
    delta_t = data["meta"]["delta_t"]
    station_params = data["parameters"]["stations"]
    prices = data["parameters"]["electricity_price"]
    storage_params = data["parameters"].get("vertiport_storage")

    if storage_params is None:
        raise ValueError("Missing required key path: parameters.vertiport_storage")

    model = pulp.LpProblem("evtol_inventory", pulp.LpMinimize)

    B = {
        dep: {
            t: pulp.LpVariable(
                f"B_{dep}_{t}",
                lowBound=storage_params[dep]["E_min"],
                upBound=storage_params[dep]["E_max"],
            )
            for t in times
        }
        for dep in e_dep
    }
    P = {
        dep: {
            t: pulp.LpVariable(
                f"P_{dep}_{t}", lowBound=0.0, upBound=station_params[dep]["P_site"][t]
            )
            for t in times
        }
        for dep in e_dep
    }

    model += pulp.lpSum(prices[dep][t] * P[dep][t] * delta_t for dep in e_dep for t in times)

    for dep in e_dep:
        if dep not in storage_params:
            raise ValueError(f"Missing required key path: parameters.vertiport_storage.{dep}")
        model += B[dep][times[0]] == storage_params[dep]["E_init"]
        eta = storage_params[dep]["eta_ch"]
        for idx, t in enumerate(times):
            if idx < len(times) - 1:
                model += (
                    B[dep][times[idx + 1]]
                    == B[dep][t] + eta * P[dep][t] * delta_t - e_dep[dep][t]
                )

    cap_pax = data["parameters"].get("vertiport_cap_pax")
    if cap_pax:
        for dep, time_map in d_dep.items():
            for t in times:
                if d_dep[dep][t] > cap_pax[dep][t]:
                    raise ValueError(f"Vertiport pax cap exceeded at {dep}, t={t}")

    model.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[model.status] != "Optimal":
        raise ValueError(f"eVTOL inventory optimization did not solve to optimality: {pulp.LpStatus[model.status]}")

    B_out = {dep: {t: float(pulp.value(B[dep][t])) for t in times} for dep in e_dep}
    P_out = {dep: {t: float(pulp.value(P[dep][t])) for t in times} for dep in e_dep}

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
                max(storage_params[dep]["E_min"] - B_out[dep][t], B_out[dep][t] - storage_params[dep]["E_max"], 0.0),
            )
            residuals["INV3"] = max(
                residuals["INV3"],
                max(0.0, P_out[dep][t] - station_params[dep]["P_site"][t]),
            )
    if cap_pax:
        for dep, time_map in d_dep.items():
            for t in times:
                residuals["INV4"] = max(residuals["INV4"], max(0.0, d_dep[dep][t] - cap_pax[dep][t]))

    return B_out, P_out, residuals


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

    residuals = compute_charging_residuals(data, E_out, p_out, y_out)

    B_out = None
    P_out = None
    inv_residuals = None
    if e_dep and d_dep:
        if any(val > 0.0 for dep in e_dep for val in e_dep[dep].values()):
            B_out, P_out, inv_residuals = solve_evtol_inventory(data, e_dep, d_dep)

    return E_out, p_out, y_out, residuals, B_out, P_out, inv_residuals


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
