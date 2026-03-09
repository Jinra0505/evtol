from typing import Dict, List


def _safe_den(x, eps: float = 1e-6) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return eps
    return v if v > 0 else eps


def compute_road_times(
    arc_flows: Dict[str, Dict[int, float]],
    arc_params: Dict[str, Dict[str, float]],
    g_values: Dict[int, float],
    times: List[int],
) -> Dict[str, Dict[int, float]]:
    tau = {arc: {} for arc in arc_params}
    for arc, params in arc_params.items():
        tau0 = params["tau0"]
        cap = _safe_den(params.get("cap", 0))
        alpha = params["alpha"]
        beta = params["beta"]
        arc_type = params["type"]
        theta = params.get("theta", 1.0)
        use_bpr = params.get("use_bpr", False)
        for t in times:
            x = arc_flows.get(arc, {}).get(t, 0.0)
            if arc_type == "G":
                tau[arc][t] = tau0 * (1.0 + alpha * (x / cap) ** beta)
            elif arc_type == "CBD":
                tau[arc][t] = tau0 * g_values.get(t, 1.0) * theta
            else:
                if use_bpr:
                    tau[arc][t] = tau0 * (1.0 + alpha * (x / cap) ** beta)
                else:
                    tau[arc][t] = tau0
    return tau


def compute_station_waits(
    utilization: Dict[str, Dict[int, float]],
    station_params: Dict[str, Dict[str, float]],
    times: List[int],
) -> Dict[str, Dict[int, float]]:
    waits = {s: {} for s in station_params}
    for station, params in station_params.items():
        cap = _safe_den(params.get("cap_stall", 0))
        w0 = params["w0"]
        for t in times:
            u = utilization.get(station, {}).get(t, 0.0)
            waits[station][t] = w0 * (1.0 + 0.15 * (u / cap) ** 4)
    return waits
