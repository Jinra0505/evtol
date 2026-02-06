import json
from typing import Any, Dict, List, Tuple

from .assignment import (
    aggregate_arc_flows,
    aggregate_ev_energy_demand,
    aggregate_evtol_dep_demand,
    aggregate_evtol_demand,
    aggregate_station_utilization,
    build_incidence,
    compute_evtol_energy_demand,
    compute_itinerary_costs,
    logit_assignment,
)
from .charging import solve_charging, solve_shared_power_inventory_lp
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


def _apply_vertiport_caps(
    flows: Dict[str, Dict[str, Dict[int, float]]],
    itineraries: List[Dict[str, Any]],
    times: List[int],
    cap_pax: Dict[str, Dict[int, float]],
) -> Dict[str, Dict[str, Dict[int, float]]]:
    itineraries_by_od: Dict[str, List[Dict[str, Any]]] = {}
    evtol_by_dep: Dict[str, List[Dict[str, Any]]] = {}

    for it in itineraries:
        od_key = f"{it['od'][0]}-{it['od'][1]}"
        itineraries_by_od.setdefault(od_key, []).append(it)
        if it.get("mode") == "eVTOL" and it.get("dep_station") is not None:
            evtol_by_dep.setdefault(it["dep_station"], []).append(it)

    for dep, it_list in evtol_by_dep.items():
        for t in times:
            total = 0.0
            for it in it_list:
                for group, time_map in flows[it["id"]].items():
                    total += time_map[t]
            cap = cap_pax[dep][t]
            if total <= cap:
                continue
            if total <= 0.0:
                continue
            ratio = cap / total
            reductions: Dict[Tuple[str, str], float] = {}
            for it in it_list:
                for group, time_map in flows[it["id"]].items():
                    key = (f"{it['od'][0]}-{it['od'][1]}", group)
                    original = time_map[t]
                    reduced = original * ratio
                    reductions[key] = reductions.get(key, 0.0) + (original - reduced)
                    flows[it["id"]][group][t] = reduced

            for (od_key, group), delta in reductions.items():
                alternatives = [
                    it
                    for it in itineraries_by_od.get(od_key, [])
                    if it.get("mode") != "eVTOL"
                ]
                if not alternatives:
                    raise ValueError(f"No non-eVTOL alternative to enforce cap for {od_key}, {group}, t={t}")
                total_alt = sum(flows[it["id"]][group][t] for it in alternatives)
                if total_alt <= 0.0:
                    share = 1.0 / len(alternatives)
                    for it in alternatives:
                        flows[it["id"]][group][t] += delta * share
                else:
                    for it in alternatives:
                        flows[it["id"]][group][t] += delta * flows[it["id"]][group][t] / total_alt

    return flows


def compute_residuals(
    data: Dict[str, Any],
    itineraries: List[Dict[str, Any]],
    flows: Dict[str, Dict[str, Dict[int, float]]],
    arc_flows: Dict[str, Dict[int, float]],
    tau: Dict[str, Dict[int, float]],
    n_series: List[float],
    g_series: List[float],
    inc_road: Dict[str, Dict[str, Dict[int, float]]],
    inc_station: Dict[str, Dict[str, Dict[int, float]]],
    utilization: Dict[str, Dict[int, float]],
) -> Dict[str, float]:
    times = data["sets"]["time"]
    arcs = data["sets"]["arcs"]
    demand = data["parameters"]["q"]
    arc_params = data["parameters"]["arcs"]
    boundary_in = data["parameters"]["boundary_in"]
    boundary_out = data["parameters"]["boundary_out"]
    mfd_params = data["parameters"]["mfd"]
    delta_t = data["meta"]["delta_t"]

    residuals = {"C1": 0.0, "C2": 0.0, "C7": 0.0, "C8": 0.0, "C9": 0.0, "C11": 0.0, "C10": 0.0}

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

    inflow_total, outflow_total = boundary_flows(arc_flows, boundary_in, boundary_out, times)
    inflow = [val / delta_t for val in inflow_total]
    outflow = [val / delta_t for val in outflow_total]
    for idx in range(len(times)):
        residuals["C7"] = max(
            residuals["C7"],
            abs(n_series[idx + 1] - (n_series[idx] + delta_t * (inflow[idx] - outflow[idx]))),
        )

    expected_g = compute_g(n_series, mfd_params)
    for idx, t in enumerate(times):
        residuals["C8"] = max(residuals["C8"], abs(g_series[idx] - expected_g[idx]))
        for arc, params in arc_params.items():
            if params["type"] == "CBD":
                expected_tau = params["tau0"] * g_series[idx] * params.get("theta", 1.0)
                residuals["C9"] = max(residuals["C9"], abs(tau[arc][t] - expected_tau))

    for station in utilization:
        for t in times:
            expected_u = 0.0
            for it in itineraries:
                for group, time_map in flows[it["id"]].items():
                    expected_u += inc_station.get(station, {}).get(it["id"], {}).get(t, 0.0) * time_map[t]
            residuals["C10"] = max(residuals["C10"], abs(utilization[station][t] - expected_u))

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
    electricity_price = data["parameters"]["electricity_price"]

    arc_flows = {a: {t: 0.0 for t in times} for a in arcs}
    utilization = {s: {t: 0.0 for t in times} for s in stations}
    n_series = update_accumulation(
        data["parameters"]["n0"],
        [0.0 for _ in times],
        [0.0 for _ in times],
        delta_t,
    )
    g_series = compute_g(n_series, data["parameters"]["mfd"])
    energy_surcharge = {s: {t: 0.0 for t in times} for s in stations}

    dx = 0.0
    dn = 0.0
    last_iteration = 0

    d_vt_route: Dict[str, Dict[int, float]] = {}
    d_vt_dep: Dict[str, Dict[int, float]] = {}
    e_dep: Dict[str, Dict[int, float]] = {}
    shadow_prices: Dict[str, Dict[int, float]] | None = None
    shed_ev: Dict[str, Dict[int, float]] | None = None
    shed_vt: Dict[str, Dict[int, float]] | None = None
    P_vt_lp: Dict[str, Dict[int, float]] | None = None
    B_vt_lp: Dict[str, Dict[int, float]] | None = None
    shared_power_residuals: Dict[str, float] | None = None

    for iteration in range(config["max_iter"]):
        last_iteration = iteration + 1
        g_by_time = {t: g_series[min(len(g_series) - 1, idx)] for idx, t in enumerate(times)}
        tau = compute_road_times(arc_flows, arc_params, g_by_time, times)
        waits = compute_station_waits(utilization, station_params, times)
        effective_prices = {
            s: {t: electricity_price[s][t] + energy_surcharge[s][t] for t in times}
            for s in stations
        }
        itineraries += generate_itineraries(data, tau, waits, config)
        inc_road, inc_station = build_incidence(itineraries, arcs, stations, times)
        costs = compute_itinerary_costs(
            itineraries,
            tau,
            waits,
            effective_prices,
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

        cap_pax = data["parameters"].get("vertiport_cap_pax")
        if cap_pax:
            flows = _apply_vertiport_caps(flows, itineraries, times, cap_pax)

        d_vt_route = aggregate_evtol_demand(flows, itineraries, times)
        d_vt_dep = aggregate_evtol_dep_demand(itineraries, flows, times)
        e_dep = compute_evtol_energy_demand(d_vt_route, itineraries, times)

        x_new = aggregate_arc_flows(itineraries, flows, times)
        u_new = aggregate_station_utilization(itineraries, flows, times)
        x_new = _fill_missing(x_new, arcs, times)
        u_new = _fill_missing(u_new, stations, times)

        dx = max(abs(x_new[a][t] - arc_flows[a][t]) for a in arcs for t in times)

        phi = 1.0 / (iteration + 1.0)
        for a in arcs:
            for t in times:
                arc_flows[a][t] = (1.0 - phi) * arc_flows[a][t] + phi * x_new[a][t]
        for s in stations:
            for t in times:
                utilization[s][t] = (1.0 - phi) * utilization[s][t] + phi * u_new[s][t]
        inflow_total, outflow_total = boundary_flows(
            arc_flows, data["parameters"]["boundary_in"], data["parameters"]["boundary_out"], times
        )
        inflow = [val / delta_t for val in inflow_total]
        outflow = [val / delta_t for val in outflow_total]
        n_new = update_accumulation(data["parameters"]["n0"], inflow, outflow, delta_t)
        dn = max(abs(n_new[idx] - n_series[idx]) for idx in range(len(n_series)))
        n_series = n_new
        g_series = compute_g(n_series, data["parameters"]["mfd"])

        ev_energy = aggregate_ev_energy_demand(itineraries, flows, times)
        (
            B_vt_lp,
            P_vt_lp,
            shed_ev,
            shed_vt,
            shadow_prices,
            shared_power_residuals,
        ) = solve_shared_power_inventory_lp(data, e_dep, ev_energy)

        if shadow_prices:
            for s in stations:
                for t in times:
                    new_surcharge = max(0.0, shadow_prices.get(s, {}).get(t, 0.0))
                    energy_surcharge[s][t] = (1.0 - phi) * energy_surcharge[s][t] + phi * new_surcharge

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
        inc_station,
        utilization,
    )

    E, p_ch, y, charging_residuals, B_vt, P_vt, inv_residuals, _ = solve_charging(data, e_dep, d_vt_dep)

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
        "shadow_price_power": shadow_prices,
        "surcharge_power": energy_surcharge,
        "shared_power": {
            "B_vt": B_vt_lp,
            "P_vt": P_vt_lp,
            "shed_ev": shed_ev,
            "shed_vt": shed_vt,
            "residuals": shared_power_residuals,
        },
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
        "CBD": {"C7": residuals["C7"], "C8": residuals["C8"], "C9": residuals["C9"]},
        "Choice": {"C11": residuals["C11"]},
        "Stations": {"C10": residuals["C10"]},
        "Charging": results["charging"]["residuals"],
        "Inventory": results["inventory"]["residuals"],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
