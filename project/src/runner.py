import json
from typing import Any, Dict

from .assignment import (
    aggregate_arc_flows,
    aggregate_station_utilization,
    build_incidence,
    compute_itinerary_costs,
    logit_assignment,
)
from .charging import solve_charging
from .congestion import compute_road_times, compute_station_waits
from .data_loader import load_data
from .itinerary_generator import generate_itineraries
from .mfd import boundary_flows, compute_g, update_accumulation


def run_equilibrium(data: Dict[str, Any]) -> Dict[str, Any]:
    times = data["sets"]["time"]
    arcs = data["sets"]["arcs"]
    stations = data["sets"]["stations"]
    delta_t = data["meta"]["delta_t"]
    config = data["config"]
    itineraries = list(data["itineraries"])
    arc_params = data["parameters"]["arcs"]
    station_params = data["parameters"]["stations"]

    arc_flows = {a: {t: 0.0 for t in times} for a in arcs}
    utilization = {s: {t: 0.0 for t in times} for s in stations}
    n = [data["parameters"]["n0"]]
    g_series = compute_g(n, data["parameters"]["mfd"])

    for _ in range(config["max_iter"]):
        g_by_time = {t: g_series[min(len(g_series) - 1, idx)] for idx, t in enumerate(times)}
        tau = compute_road_times(arc_flows, arc_params, g_by_time, times)
        waits = compute_station_waits(utilization, station_params, times)
        itineraries += generate_itineraries(data, tau, waits, config)
        inc_road, inc_station = build_incidence(itineraries, arcs, stations, times)
        costs = compute_itinerary_costs(
            itineraries,
            tau,
            waits,
            data["parameters"]["electricity_price"],
            times,
        )
        flows, _ = logit_assignment(
            itineraries,
            costs,
            data["parameters"]["q"],
            data["parameters"]["VOT"],
            data["parameters"]["lambda"],
            times,
        )
        arc_flows = aggregate_arc_flows(itineraries, inc_road, flows, times, arcs)
        utilization = aggregate_station_utilization(itineraries, inc_station, flows, times, stations)
        inflow, outflow = boundary_flows(arc_flows, data["parameters"]["boundary_in"], data["parameters"]["boundary_out"], times)
        n = update_accumulation(data["parameters"]["n0"], inflow, outflow, delta_t)
        g_series = compute_g(n, data["parameters"]["mfd"])
        if max(abs(n[-1] - n[0]), max(max(abs(arc_flows[a][t]) for t in times) for a in arcs)) < config["tol"]:
            break

    E, p_ch, y, charging_residuals = solve_charging(data)

    results = {
        "x": arc_flows,
        "tau": tau,
        "n": n,
        "g": g_series,
        "u": utilization,
        "w": waits,
        "f": flows,
        "charging": {
            "E": E,
            "p_ch": p_ch,
            "y": y,
            "residuals": charging_residuals,
        },
    }
    return results


def save_outputs(results: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def main() -> None:
    data = load_data("project/data/toy_data.yaml", "project/data_schema.yaml")
    results = run_equilibrium(data)
    save_outputs(results, "project/output.json")
    print(json.dumps(results["charging"]["residuals"], indent=2))


if __name__ == "__main__":
    main()
