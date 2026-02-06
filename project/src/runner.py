import json
from typing import Any, Dict, List, Tuple

from .assignment import (
    aggregate_arc_flows,
    aggregate_evtol_demand,
    aggregate_station_utilization,
    build_incidence,
    compute_evtol_energy_demand,
    compute_itinerary_costs,
    logit_assignment,
)
from .charging import solve_charging
from .congestion import compute_road_times, compute_station_waits
from .data_loader import load_data
from .itinerary_generator import generate_itineraries
from .mfd import boundary_flows, compute_g, update_accumulation


def _fill_missing(mapping: Dict[str, Dict[int, float]], keys: List[str], times: List[int]) -> Dict[str, Dict[int, float]]:
    for key in keys:
        mapping.setdefault(key, {t: 0.0 for t in times})
        for t in times:
            mapping[key].setdefault(t, 0.0)
    return mapping


def compute_residuals(
    data: Dict[str, Any],
    itineraries: List[Dict[str, Any]],
    flows: Dict[str, Dict[str, Dict[int, float]]],
    arc_flows: Dict[str, Dict[int, float]],
    tau: Dict[str, Dict[int, float]],
    n_series: List[float],
    g_series: List[float],
    inc_road: Dict[str, Dict[str, Dict[int, float]]],
) -> Dict[str, float]:
    times = data["sets"]["time"]
    arcs = data["sets"]["arcs"]
    demand = data["parameters"]["q"]
    arc_params = data["parameters"]["arcs"]
    boundary_in = data["parameters"]["boundary_in"]
    boundary_out = data["parameters"]["boundary_out"]
    mfd_params = data["parameters"]["mfd"]

    residuals = {"C1": 0.0, "C2": 0.0, "C6": 0.0, "C7": 0.0, "C8": 0.0, "C9": 0.0, "C11": 0.0}

    for od_key, groups in demand.items():
        for group, time_map in groups.items():
            for t, q_val in time_map.items():
                total = 0.0
                for it in itineraries:
                    if f"{it['od'][0]}-{it['od'][1]}" != od_key:
                        continue
                    total += flows[it["id"]].get(group, {}).get(t, 0.0)
                residuals["C1"] = max(residuals["C1"], abs(total - q_val))
                residuals["C11"] = max(residuals["C11"], abs(total - q_val))

    for arc in arcs:
        for t in times:
            expected = 0.0
            for it in itineraries:
                for group, time_map in flows[it["id"]].items():
                    expected += inc_road[arc][it["id"]][t] * time_map[t]
            residuals["C2"] = max(residuals["C2"], abs(arc_flows[arc][t] - expected))

    inflow, outflow = boundary_flows(arc_flows, boundary_in, boundary_out, times)
    residuals["C6"] = 0.0

    for idx, t in enumerate(times):
        residuals["C7"] = max(
            residuals["C7"],
            abs(n_series[idx + 1] - (n_series[idx] + inflow[idx] - outflow[idx])),
        )

    expected_g = compute_g(n_series, mfd_params)
    for idx, t in enumerate(times):
        residuals["C8"] = max(residuals["C8"], abs(g_series[idx] - expected_g[idx]))
        for arc, params in arc_params.items():
            if params["type"] == "CBD":
                expected_tau = params["tau0"] * g_series[idx] * params.get("theta", 1.0)
                residuals["C9"] = max(residuals["C9"], abs(tau[arc][t] - expected_tau))
    return residuals


def run_equilibrium(data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float]]:
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
    n_series = [data["parameters"]["n0"]]
    g_series = compute_g(n_series, data["parameters"]["mfd"])

    dx = 0.0
    dn = 0.0
    last_iteration = 0

    d_vt_route: Dict[str, Dict[int, float]] = {}
    d_vt_dep: Dict[str, Dict[int, float]] = {}
    e_dep: Dict[str, Dict[int, float]] = {}

    for iteration in range(config["max_iter"]):
        last_iteration = iteration + 1
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
        d_vt_route, d_vt_dep = aggregate_evtol_demand(flows, itineraries, times)
        e_dep = compute_evtol_energy_demand(d_vt_route, itineraries, times, data["parameters"])

        x_new = aggregate_arc_flows(itineraries, flows, times)
        u_new = aggregate_station_utilization(itineraries, flows, times)
        x_new = _fill_missing(x_new, arcs, times)
        u_new = _fill_missing(u_new, stations, times)

        inflow, outflow = boundary_flows(x_new, data["parameters"]["boundary_in"], data["parameters"]["boundary_out"], times)
        n_new = update_accumulation(data["parameters"]["n0"], inflow, outflow, delta_t)

        dx = max(abs(x_new[a][t] - arc_flows[a][t]) for a in arcs for t in times)
        dn = max(abs(n_new[idx] - n_series[idx]) for idx in range(len(n_series)))

        phi = 1.0 / (iteration + 1.0)
        for a in arcs:
            for t in times:
                arc_flows[a][t] = (1.0 - phi) * arc_flows[a][t] + phi * x_new[a][t]
        for s in stations:
            for t in times:
                utilization[s][t] = (1.0 - phi) * utilization[s][t] + phi * u_new[s][t]
        n_series = n_new
        g_series = compute_g(n_series, data["parameters"]["mfd"])

        if max(dx, dn) < config["tol"]:
            break

    residuals = compute_residuals(
        data,
        itineraries,
        flows,
        arc_flows,
        tau,
        n_series,
        g_series,
        inc_road,
    )

    E, p_ch, y, charging_residuals, B_vt, P_vt, inv_residuals = solve_charging(data, e_dep, d_vt_dep)

    results = {
        "x": arc_flows,
        "tau": tau,
        "n": n_series,
        "g": g_series,
        "u": utilization,
        "w": waits,
        "f": flows,
        "d_vt_route": d_vt_route,
        "d_vt_dep": d_vt_dep,
        "e_dep": e_dep,
        "inventory": {"B_vt": B_vt, "P_vt": P_vt, "residuals": inv_residuals},
        "residuals": residuals,
        "convergence": {"dx": dx, "dn": dn, "iterations": last_iteration},
        "charging": {
            "E": E,
            "p_ch": p_ch,
            "y": y,
            "residuals": charging_residuals,
        },
    }
    return results, residuals


def save_outputs(results: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def main() -> None:
    data = load_data("project/data/toy_data.yaml", "project/data_schema.yaml")
    results, residuals = run_equilibrium(data)
    save_outputs(results, "project/output.json")
    report = {
        "Traffic": {"C1": residuals["C1"], "C2": residuals["C2"]},
        "CBD": {"C6": residuals["C6"], "C7": residuals["C7"], "C8": residuals["C8"], "C9": residuals["C9"]},
        "Choice": {"C11": residuals["C11"]},
        "Charging": results["charging"]["residuals"],
        "Inventory": results["inventory"]["residuals"],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
