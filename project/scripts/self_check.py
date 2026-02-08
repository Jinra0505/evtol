import json
import os
import sys
<<<<<<< codex/implement-multi-time-traffic-equilibrium-model-25bt3w
from contextlib import redirect_stdout
from io import StringIO
=======
>>>>>>> main
from pathlib import Path

if __package__ is None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    __package__ = "project.scripts"

from project.src.charging import HAS_GUROBI
from project.src.data_loader import load_data
from project.src.planner import apply_plan_to_data, solve_planning
from project.src.runner import run_equilibrium


def _resolve_path(path: str, fallback: str) -> str:
    if os.path.exists(path):
        return path
    repo_root = Path(__file__).resolve().parents[1]
    data_candidate = repo_root / "data" / os.path.basename(path)
    if data_candidate.exists():
        return str(data_candidate)
    schema_candidate = repo_root / os.path.basename(path)
    if schema_candidate.exists():
        return str(schema_candidate)
    if os.path.exists(fallback):
        return fallback
    candidate = repo_root / fallback
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(f"Could not find data file: {path}")


def _check_availability(data: dict, results: dict) -> None:
    times = data["sets"]["time"]
    itineraries = data["itineraries"]
    flows = results["f"]
    for it in itineraries:
        if it.get("mode") != "eVTOL":
            continue
        flight_time = it.get("flight_time", {})
        for group, time_map in flows.get(it["id"], {}).items():
            for t in times:
                if flight_time.get(t, 0.0) <= 0.0:
                    val = time_map.get(t, 0.0)
                    if abs(val) > 1e-8:
                        raise AssertionError(
                            f"eVTOL availability violated for {it['id']} group {group} t={t}: {val}"
                        )


def _check_constraints(data: dict, results: dict) -> None:
    times = data["sets"]["time"]
    stations = data["sets"]["stations"]
    vehicles = data["sets"]["vehicles"]
    station_params = data["parameters"]["stations"]
    cap_pax = data["parameters"].get("vertiport_cap_pax", {})

    p_ch = results["charging"]["p_ch"]
    y = results["charging"]["y"]
    P_vt = results["inventory"]["P_vt"] or {}

    for s in stations:
        cap_stall = station_params[s]["cap_stall"]
        for t in times:
            total_power = sum(p_ch[m][s][t] for m in vehicles)
            if s in P_vt:
                total_power += P_vt[s][t]
            if total_power > station_params[s]["P_site"][t] + 1e-6:
                raise AssertionError(f"Power constraint violated at {s}, t={t}: {total_power}")
            total_stalls = sum(y[m][s][t] for m in vehicles)
            if total_stalls > cap_stall + 1e-6:
                raise AssertionError(f"Stall constraint violated at {s}, t={t}: {total_stalls}")

    d_vt_dep = results["d_vt_dep"]
    for dep, time_map in cap_pax.items():
        for t in times:
            if d_vt_dep.get(dep, {}).get(t, 0.0) > time_map[t] + 1e-6:
                raise AssertionError(
                    f"Vertiport cap violated at {dep}, t={t}: {d_vt_dep.get(dep, {}).get(t, 0.0)}"
                )


def _check_planner_objective(data: dict, results: dict, breakdown: dict) -> None:
    flows = results["f"]
    costs = results["costs"]
    vot = data["parameters"]["VOT"]
    delta_t = data["meta"]["delta_t"]
    prices = data["parameters"]["electricity_price"]

    time_cost = 0.0
    for it_id, group_map in flows.items():
        for group, time_map in group_map.items():
            for t, flow in time_map.items():
                time_cost += flow * vot[group][t] * costs[it_id][t]["TT"]

    energy_cost = 0.0
    for m, station_map in results["charging"]["p_ch"].items():
        for s, time_map in station_map.items():
            for t, power in time_map.items():
                energy_cost += prices[s][t] * power * delta_t
    if results["inventory"]["P_vt"]:
        for dep, time_map in results["inventory"]["P_vt"].items():
            for t, power in time_map.items():
                energy_cost += prices[dep][t] * power * delta_t

    if abs(time_cost - breakdown["time_cost"]) > 1e-6:
        raise AssertionError("Planner time_cost mismatch")
    if abs(energy_cost - breakdown["energy_cost"]) > 1e-6:
        raise AssertionError("Planner energy_cost mismatch")
    if abs(breakdown["total_cost"] - (breakdown["time_cost"] + breakdown["energy_cost"] + breakdown["invest_cost"] + breakdown["shed_cost"])) > 1e-6:
        raise AssertionError("Planner total_cost mismatch")


def _check_convergence(data: dict, results: dict) -> None:
    tol = float(data["config"]["tol"])
    dx = results["convergence"]["dx"]
    dn = results["convergence"]["dn"]
<<<<<<< codex/implement-multi-time-traffic-equilibrium-model-25bt3w
    dprice = results["convergence"].get("dprice", 0.0)
    max_iter = int(data["config"]["max_iter"])
    if results["convergence"]["iterations"] < max_iter and max(dx, dn, dprice) > tol:
        raise AssertionError(f"Convergence tolerance not met: dx={dx}, dn={dn}, dprice={dprice}, tol={tol}")


def _check_price_feedback(data: dict, results: dict, require_nonzero: bool) -> None:
    diagnostics = results.get("diagnostics", {})
    dprice_history = diagnostics.get("dprice_history", [])
    if not dprice_history:
        raise AssertionError("Missing dprice history for price feedback check")
    tol = float(data["config"]["tol"])
    if len(dprice_history) >= 5:
        if max(dprice_history[-5:]) > 5.0 * tol:
            raise AssertionError("Price feedback dprice not within 5x tol in last 5 iterations")
    if dprice_history[-1] > tol:
        if len(dprice_history) < 3:
            raise AssertionError(f"Price feedback not converged: dprice={dprice_history[-1]}")
        last_three = dprice_history[-3:]
        if dprice_history[-1] > last_three[0]:
            raise AssertionError("Price feedback not stabilizing in last iterations")
    if require_nonzero:
        surcharge = results.get("surcharge_power", {})
        if not any(val > 1e-6 for station in surcharge.values() for val in station.values()):
            raise AssertionError("No nonzero surcharge detected in complex case")


def _check_mode_share_shift(results: dict, min_shift: float) -> None:
    diagnostics = results.get("diagnostics", {})
    mode_share = diagnostics.get("mode_share", [])
    if len(mode_share) < 2:
        raise AssertionError("Missing mode_share diagnostics")
    start = mode_share[0]
    end = mode_share[-1]
    shift = abs(end["vt_share"] - start["vt_share"])
    if shift < min_shift:
        raise AssertionError(f"Mode share shift too small: {shift}")


def _check_stop_reason(results: dict) -> None:
    stop_reason = results.get("diagnostics", {}).get("stop_reason")
    if stop_reason == "max_iters":
        raise AssertionError("Run stopped due to max_iters instead of convergence")


def run_case(data_path: str, schema_path: str, suppress_output: bool = False) -> dict:
    data = load_data(data_path, schema_path)
    if suppress_output:
        buffer = StringIO()
        with redirect_stdout(buffer):
            results, residuals = run_equilibrium(data)
    else:
        results, residuals = run_equilibrium(data)
    _check_convergence(data, results)
    _check_availability(data, results)
    _check_constraints(data, results)
    _check_price_feedback(data, results, require_nonzero="complex_case" in os.path.basename(data_path))
    if "complex_case" in os.path.basename(data_path):
        _check_stop_reason(results)
        _check_mode_share_shift(results, min_shift=0.01)
=======
    if max(dx, dn) > tol:
        raise AssertionError(f"Convergence tolerance not met: dx={dx}, dn={dn}, tol={tol}")


def run_case(data_path: str, schema_path: str) -> dict:
    data = load_data(data_path, schema_path)
    results, residuals = run_equilibrium(data)
    _check_convergence(data, results)
    _check_availability(data, results)
    _check_constraints(data, results)
>>>>>>> main
    if not HAS_GUROBI and results["solver_used"] != "pulp":
        raise AssertionError(f"Solver fallback mismatch: {results['solver_used']}")
    return {"results": results, "residuals": residuals}


<<<<<<< codex/implement-multi-time-traffic-equilibrium-model-25bt3w
def run_planner(data_path: str, schema_path: str, suppress_output: bool = False) -> dict:
    data = load_data(data_path, schema_path)
    if suppress_output:
        buffer = StringIO()
        with redirect_stdout(buffer):
            best_plan, best_cost, best_results, best_diag = solve_planning(data)
    else:
        best_plan, best_cost, best_results, best_diag = solve_planning(data)
=======
def run_planner(data_path: str, schema_path: str) -> dict:
    data = load_data(data_path, schema_path)
    best_plan, best_cost, best_results, best_diag = solve_planning(data)
>>>>>>> main
    data_plan = apply_plan_to_data(data, best_plan)
    _check_convergence(data_plan, best_results)
    _check_availability(data_plan, best_results)
    _check_constraints(data_plan, best_results)
    _check_planner_objective(data_plan, best_results, best_diag)
<<<<<<< codex/implement-multi-time-traffic-equilibrium-model-25bt3w
    _check_price_feedback(data_plan, best_results, require_nonzero="complex_case" in os.path.basename(data_path))
    if "complex_case" in os.path.basename(data_path):
        _check_stop_reason(best_results)
        _check_mode_share_shift(best_results, min_shift=0.01)
=======
>>>>>>> main
    if not HAS_GUROBI and best_results["solver_used"] != "pulp":
        raise AssertionError(f"Solver fallback mismatch: {best_results['solver_used']}")
    return {
        "best_plan": best_plan,
        "best_cost": best_cost,
        "breakdown": best_diag,
    }


def main() -> None:
<<<<<<< codex/implement-multi-time-traffic-equilibrium-model-25bt3w
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to data YAML (optional)")
    parser.add_argument("--schema", default="data_schema.yaml", help="Path to schema YAML")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    schema_path = _resolve_path(args.schema, str(repo_root / "data_schema.yaml"))

    if args.data:
        data_path = _resolve_path(args.data, str(repo_root / "data" / os.path.basename(args.data)))
        eq = run_case(data_path, schema_path, suppress_output=True)
        plan = run_planner(data_path, schema_path, suppress_output=True)
        diagnostics = eq["results"].get("diagnostics", {})
        dprice_history = diagnostics.get("dprice_history", [])
        share_history = diagnostics.get("mode_share", [])
        stop_reason = diagnostics.get("stop_reason")
        max_surcharge = max(
            val for station in eq["results"].get("surcharge_power", {}).values() for val in station.values()
        )
        start_share = share_history[0] if share_history else {}
        end_share = share_history[-1] if share_history else {}
        summary = {
            "equilibrium": eq["residuals"],
            "planning": plan["breakdown"],
            "diagnostics": {
                "stop_reason": stop_reason,
                "dprice_start": dprice_history[0] if dprice_history else None,
                "dprice_end": dprice_history[-1] if dprice_history else None,
                "max_surcharge": max_surcharge,
                "mode_share_start": start_share,
                "mode_share_end": end_share,
            },
        }
        print(json.dumps(summary, indent=2))
        return

    data_path = _resolve_path("toy_data.yaml", str(repo_root / "data" / "toy_data.yaml"))
=======
    repo_root = Path(__file__).resolve().parents[1]
    data_path = _resolve_path("toy_data.yaml", str(repo_root / "data" / "toy_data.yaml"))
    schema_path = _resolve_path("data_schema.yaml", str(repo_root / "data_schema.yaml"))

>>>>>>> main
    toy_eq = run_case(data_path, schema_path)
    toy_plan = run_planner(data_path, schema_path)

    complex_path = _resolve_path("complex_case.yaml", str(repo_root / "data" / "complex_case.yaml"))
    complex_eq = run_case(complex_path, schema_path)
    complex_plan = run_planner(complex_path, schema_path)

    summary = {
        "toy": {"equilibrium": toy_eq["residuals"], "planning": toy_plan["breakdown"]},
        "complex": {"equilibrium": complex_eq["residuals"], "planning": complex_plan["breakdown"]},
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
