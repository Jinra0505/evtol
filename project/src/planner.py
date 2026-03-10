import copy
import itertools
import os
import sys
from typing import Any, Dict, List, Tuple

if __package__ is None:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    __package__ = "project.src"

from .runner import run_equilibrium


def _planning_defaults(data: Dict[str, Any]) -> Dict[str, Any]:
    defaults = {
        "delta_p_site": [0.0, 10.0, 20.0],
        "delta_stall": [0, 1, 2],
        "delta_cap_pax": [0.0, 5.0, 10.0],
        "price_factor": [1.0],
        "max_plans": None,
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
    price_choices = planning["price_factor"]

    plans: List[Dict[str, Any]] = []
    for p_site_delta in itertools.product(p_site_choices, repeat=len(stations)):
        for stall_delta in itertools.product(stall_choices, repeat=len(stations)):
            for cap_delta in itertools.product(cap_choices, repeat=len(cap_pax)):
                for price_factor in price_choices:
                    plan = {
                        "delta_p_site": dict(zip(stations, p_site_delta)),
                        "delta_stall": dict(zip(stations, stall_delta)),
                        "delta_cap_pax": dict(zip(cap_pax.keys(), cap_delta)),
                        "price_factor": price_factor,
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
    price_factor = plan.get("price_factor", 1.0)

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

    if price_factor != 1.0:
        for s in stations:
            for t in times:
                params["electricity_price"][s][t] *= price_factor

    return data_copy


def evaluate_plan(data: Dict[str, Any], plan: Dict[str, Any]) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
    planning = _planning_defaults(data)
    data_copy = apply_plan_to_data(data, plan)
    results, residuals = run_equilibrium(data_copy)

    time_cost = 0.0
    flows = results["f"]
    costs = results["costs"]
    vot = data_copy["parameters"]["VOT"]
    for it_id, group_map in flows.items():
        for group, time_map in group_map.items():
            for t, flow in time_map.items():
                comp = costs[it_id][t]
                time_cost += flow * (vot[group][t] * comp["TT"])

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

    total_cost = time_cost + energy_cost + shed_cost + invest_cost
    diagnostics = {
        "time_cost": time_cost,
        "energy_cost": energy_cost,
        "shed_cost": shed_cost,
        "invest_cost": invest_cost,
        "total_cost": total_cost,
        "cap_violation": results["validation"]["cap_violation"],
        "power_violation": results["validation"]["power_violation"],
        "residuals": residuals,
    }
    return total_cost, diagnostics, results


def solve_planning(data: Dict[str, Any]) -> Tuple[Dict[str, Any], float, Dict[str, Any], Dict[str, float]]:
    planning = _planning_defaults(data)
    plans = generate_candidate_plans(data)
    max_plans = planning.get("max_plans")
    if max_plans is not None:
        plans = plans[: int(max_plans)]
    cache: Dict[Tuple[Any, ...], Tuple[float, Dict[str, float], Dict[str, Any]]] = {}

    best_plan = None
    best_cost = float("inf")
    best_results: Dict[str, Any] = {}
    best_diag: Dict[str, float] = {}

    for plan in plans:
        key = (
            tuple(sorted(plan["delta_p_site"].items())),
            tuple(sorted(plan["delta_stall"].items())),
            tuple(sorted(plan["delta_cap_pax"].items())),
            plan.get("price_factor", 1.0),
        )
        if key in cache:
            cost, diag, results = cache[key]
        else:
            cost, diagnostics, results = evaluate_plan(data, plan)
            cache[key] = (cost, diagnostics, results)
            diag = diagnostics
        if cost < best_cost:
            best_cost = cost
            best_plan = plan
            best_results = results
            best_diag = diag

    if best_plan is None:
        raise ValueError("No planning candidates evaluated.")

    return best_plan, best_cost, best_results, best_diag


def main() -> None:
    import argparse
    import os
    from .data_loader import load_data

    def _resolve_path(path: str, fallback: str) -> str:
        if os.path.exists(path):
            return path
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        data_candidate = os.path.join(repo_root, "project", "data", os.path.basename(path))
        if os.path.exists(data_candidate):
            return data_candidate
        schema_candidate = os.path.join(repo_root, "project", os.path.basename(path))
        if os.path.exists(schema_candidate):
            return schema_candidate
        if os.path.exists(fallback):
            return fallback
        candidate = os.path.join(repo_root, fallback)
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError(f"Could not find data file: {path}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="toy_data.yaml", help="Path to data YAML")
    parser.add_argument("--schema", default="data_schema.yaml", help="Path to schema YAML")
    args = parser.parse_args()

    data_path = _resolve_path(args.data, "project/data/toy_data.yaml")
    schema_path = _resolve_path(args.schema, "project/data_schema.yaml")
    data = load_data(data_path, schema_path)
    best_plan, best_cost, best_results, best_diag = solve_planning(data)
    output = {
        "best_plan": best_plan,
        "best_cost": best_cost,
        "best_breakdown": best_diag,
        "best_results": best_results,
    }
    with open("project/planning_output.json", "w", encoding="utf-8") as handle:
        import json

        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
