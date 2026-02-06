from typing import Any, Dict, List, Tuple


def solve_charging(
    data: Dict[str, Any],
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[str, Dict[int, float]]], Dict[str, Dict[str, Dict[int, int]]], Dict[str, float]]:
    times = data["sets"]["time"]
    vehicles = data["sets"]["vehicles"]
    stations = data["sets"]["stations"]
    delta_t = data["meta"]["delta_t"]
    charging_params = data["parameters"]["charging"]
    avail = data["parameters"]["avail"]
    e_fly_or_drive = data["parameters"]["e_fly_or_drive"]
    station_params = data["parameters"]["stations"]

    E: Dict[str, Dict[int, float]] = {m: {} for m in vehicles}
    p_ch: Dict[str, Dict[str, Dict[int, float]]] = {m: {s: {} for s in stations} for m in vehicles}
    y: Dict[str, Dict[str, Dict[int, int]]] = {m: {s: {} for s in stations} for m in vehicles}

    for m in vehicles:
        E_init = charging_params[m]["E_init"]
        E[m][times[0]] = E_init
        for t in times:
            for s in stations:
                p_ch[m][s][t] = 0.0
                y[m][s][t] = 0
        for idx, t in enumerate(times):
            if idx == len(times) - 1:
                continue
            E[m][times[idx + 1]] = E[m][t] - e_fly_or_drive[m][t]

    residuals = compute_charging_residuals(
        data,
        E,
        p_ch,
        y,
    )
    return E, p_ch, y, residuals


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
