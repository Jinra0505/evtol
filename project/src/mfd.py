from typing import Dict, List, Tuple


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def compute_g(n: List[float], mfd_params: Dict[str, float]) -> List[float]:
    gamma = mfd_params["gamma"]
    n_crit = mfd_params["n_crit"]
    g_max = mfd_params["g_max"]
    g_values = []
    for n_t in n:
        g_val = 1.0 + gamma * (n_t / n_crit) ** 2
        g_values.append(clamp(g_val, 1.0, g_max))
    return g_values


def update_accumulation(
    n0: float, inflow: List[float], outflow: List[float], delta_t: float
) -> List[float]:
    n = [n0]
    for t in range(len(inflow)):
        n_next = n[t] + inflow[t] - outflow[t]
        n.append(n_next)
    return n


def boundary_flows(arc_flows: Dict[str, Dict[int, float]], boundary_in: List[str], boundary_out: List[str], times: List[int]) -> Tuple[List[float], List[float]]:
    inflow = []
    outflow = []
    for t in times:
        inflow.append(sum(arc_flows[a][t] for a in boundary_in))
        outflow.append(sum(arc_flows[a][t] for a in boundary_out))
    return inflow, outflow
