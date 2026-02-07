import copy
import itertools
from typing import Any, Dict, List, Tuple

from .runner import run_equilibrium


def _planning_defaults(data: Dict[str, Any]) -> Dict[str, Any]:
    defaults = {
        "delta_p_site": [0.0, 10.0, 20.0],
        "delta_stall": [0, 1, 2],
        "delta_cap_pax": [0.0, 5.0, 10.0],
        "cP": 1.0,
        "cS": 5.0,
        "cV": 2.0,
        "budget": None,
        "shed_penalty": 1.0e6,
    }
    planning = data["parameters"].get("planning", {})
    return {**defaults, **planning}


def generate_candidate_plans(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    planning = _planning_defaults(data)
    stations = data["sets"]["stations"]
    cap_pax = data["parameters"].get("vertiport_cap_pax", {})

    p_site_choices = planning["delta_p_site"]
    stall_choices = planning["delta_stall"]
    cap_choices = planning["delta_cap_pax"]

    plans: List[Dict[str, Any]] = []
    for p_site_delta in itertools.product(p_site_choices, repeat=len(stations)):
        for stall_delta in itertools.product(stall_choices, repeat=len(stations)):
            for cap_delta in itertools.product(cap_choices, repeat=len(cap_pax)):
                plan = {
                    "delta_p_site": dict(zip(stations, p_site_delta)),
                    "delta_stall": dict(zip(stations, stall_delta)),
                    "delta_cap_pax": dict(zip(cap_pax.keys(), cap_delta)),
                }
                plans.append(plan)
    return plans


def apply_plan_to_data(data: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
    data_copy = copy.deepcopy(data)
    times = data_copy["sets"]["time"]
    stations = data_copy["sets"]["stations"]
    params = data_copy["parameters"]

    delta_p_site = plan.get("delta_p_site", {})
    delta_stall = plan.get("delta_stall", {})
    delta_cap_pax = plan.get("delta_cap_pax", {})

    for s in stations:
        if s in delta_p_site:
            for t in times:
                params["stations"][s]["P_site"][t] += delta_p_site[s]
        if s in delta_stall:
            params["stations"][s]["cap_stall"] += delta_stall[s]

    if "vertiport_cap_pax" in params:
        for dep, time_map in params["vertiport_cap_pax"].items():
            for t in times:
                time_map[t] += delta_cap_pax.get(dep, 0.0)

    return data_copy


def evaluate_plan(data: Dict[str, Any], plan: Dict[str, Any]) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
    planning = _planning_defaults(data)
    data_copy = apply_plan_to_data(data, plan)
    results, residuals = run_equilibrium(data_copy)

    travel_cost = 0.0
    flows = results["f"]
    costs = results["costs"]
    vot = data_copy["parameters"]["VOT"]
    for it_id, group_map in flows.items():
        for group, time_map in group_map.items():
            for t, flow in time_map.items():
                comp = costs[it_id][t]
                travel_cost += flow * (vot[group][t] * comp["TT"] + comp["Money"])

    energy_cost = 0.0
    prices = data_copy["parameters"]["electricity_price"]
    delta_t = data_copy["meta"]["delta_t"]
    for m, station_map in results["charging"]["p_ch"].items():
        for s, time_map in station_map.items():
            for t, power in time_map.items():
                energy_cost += prices[s][t] * power * delta_t
    if results["inventory"]["P_vt"]:
        for dep, time_map in results["inventory"]["P_vt"].items():
            for t, power in time_map.items():
                energy_cost += prices[dep][t] * power * delta_t

    shed_penalty = planning["shed_penalty"]
    shed_cost = 0.0
    shared = results["shared_power"]
    if shared["shed_ev"]:
        for s, time_map in shared["shed_ev"].items():
            for t, val in time_map.items():
                shed_cost += val * delta_t
    if shared["shed_vt"]:
        for dep, time_map in shared["shed_vt"].items():
            for t, val in time_map.items():
                shed_cost += val
    shed_cost *= shed_penalty

    invest_cost = 0.0
    invest_cost += planning["cP"] * sum(plan.get("delta_p_site", {}).values())
    invest_cost += planning["cS"] * sum(plan.get("delta_stall", {}).values())
    invest_cost += planning["cV"] * sum(plan.get("delta_cap_pax", {}).values())

    budget = planning.get("budget")
    if budget is not None and invest_cost > budget:
        return 1.0e18, {"budget_violation": invest_cost - budget}, results

    total_cost = travel_cost + energy_cost + shed_cost + invest_cost
    diagnostics = {
        "travel_cost": travel_cost,
        "energy_cost": energy_cost,
        "shed_cost": shed_cost,
        "invest_cost": invest_cost,
        "cap_violation": results["validation"]["cap_violation"],
        "power_violation": results["validation"]["power_violation"],
        "residuals": residuals,
    }
    return total_cost, diagnostics, results


def solve_planning(data: Dict[str, Any]) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
    plans = generate_candidate_plans(data)
    cache: Dict[Tuple[Any, ...], Tuple[float, Dict[str, float], Dict[str, Any]]] = {}

    best_plan = None
    best_cost = float("inf")
    best_results: Dict[str, Any] = {}

    for plan in plans:
        key = (
            tuple(sorted(plan["delta_p_site"].items())),
            tuple(sorted(plan["delta_stall"].items())),
            tuple(sorted(plan["delta_cap_pax"].items())),
        )
        if key in cache:
            cost, _, results = cache[key]
        else:
            cost, diagnostics, results = evaluate_plan(data, plan)
            cache[key] = (cost, diagnostics, results)
        if cost < best_cost:
            best_cost = cost
            best_plan = plan
            best_results = results

    if best_plan is None:
        raise ValueError("No planning candidates evaluated.")

    return best_plan, best_cost, best_results


def main() -> None:
    from .data_loader import load_data

    data = load_data("project/data/toy_data.yaml", "project/data_schema.yaml")
    best_plan, best_cost, best_results = solve_planning(data)
    output = {
        "best_plan": best_plan,
        "best_cost": best_cost,
        "best_results": best_results,
    }
    with open("project/planning_output.json", "w", encoding="utf-8") as handle:
        import json

        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
