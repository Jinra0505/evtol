import math
from typing import Any, Dict, List, Tuple

from .utils import logsumexp


def build_incidence(
    itineraries: List[Dict[str, Any]],
    arcs: List[str],
    stations: List[str],
    times: List[int],
) -> Tuple[Dict[str, Dict[str, Dict[int, float]]], Dict[str, Dict[str, Dict[int, float]]]]:
    inc_road = {arc: {it["id"]: {t: 0.0 for t in times} for it in itineraries} for arc in arcs}
    inc_station = {s: {it["id"]: {t: 0.0 for t in times} for it in itineraries} for s in stations}
    for it in itineraries:
        for seg in it.get("road_arcs", []):
            arc = seg["arc"]
            t = seg["t"]
            frac = seg.get("frac", 1.0)
            inc_road[arc][it["id"]][t] += frac
        for stop in it.get("stations", []):
            station = stop["station"]
            t = stop["t"]
            inc_station[station][it["id"]][t] += 1.0
        if it.get("mode") == "eVTOL":
            dep_station = it.get("dep_station")
            if dep_station is not None:
                for t in times:
                    inc_station[dep_station][it["id"]][t] += 1.0
    return inc_road, inc_station


def compute_itinerary_costs(
    itineraries: List[Dict[str, Any]],
    travel_times: Dict[str, Dict[int, float]],
    station_waits: Dict[str, Dict[int, float]],
    electricity_price: Dict[str, Dict[int, float]],
    times: List[int],
) -> Dict[str, Dict[int, Dict[str, float]]]:
    costs: Dict[str, Dict[int, Dict[str, float]]] = {it["id"]: {} for it in itineraries}
    for it in itineraries:
        flight_time = it.get("flight_time", {})
        base_money = it.get("money", 0.0)
        phi_markup = it.get("phi_energy_markup", 1.0)
        dep_station = it.get("dep_station")
        e_per_pax = it.get("e_per_pax", 0.0)
        for t in times:
            money_t = base_money
            tt = 0.0
            charge_cost = 0.0
            for seg in it.get("road_arcs", []):
                if seg["t"] == t:
                    tt += travel_times[seg["arc"]][t] * seg.get("frac", 1.0)
            tt += float(flight_time.get(t, 0.0))
            for stop in it.get("stations", []):
                if stop["t"] == t:
                    station = stop["station"]
                    tt += station_waits[station][t]
                    energy = stop.get("energy", 0.0)
                    charge_cost += energy * electricity_price[station][t]
            if it.get("mode") == "eVTOL" and dep_station is not None:
                tt += station_waits[dep_station][t]
                energy_fare = phi_markup * e_per_pax * electricity_price[dep_station][t]
                money_t += energy_fare
            costs[it["id"]][t] = {
                "TT": tt,
                "Money": money_t,
                "ChargeCost": charge_cost,
            }
    return costs


def logit_assignment(
    itineraries: List[Dict[str, Any]],
    costs: Dict[str, Dict[int, Dict[str, float]]],
    demand: Dict[str, Dict[str, Dict[int, float]]],
    vot: Dict[str, Dict[int, float]],
    lambdas: Dict[str, float],
    times: List[int],
) -> Tuple[Dict[str, Dict[str, Dict[int, float]]], Dict[str, Dict[str, Dict[int, float]]]]:
    flows: Dict[str, Dict[str, Dict[int, float]]] = {it["id"]: {} for it in itineraries}
    generalized_costs: Dict[str, Dict[str, Dict[int, float]]] = {it["id"]: {} for it in itineraries}
    itineraries_by_od: Dict[str, List[Dict[str, Any]]] = {}
    for it in itineraries:
        od_key = f"{it['od'][0]}-{it['od'][1]}"
        itineraries_by_od.setdefault(od_key, []).append(it)

    for od_key, groups in demand.items():
        for group, time_map in groups.items():
            for t in times:
                alts = itineraries_by_od.get(od_key, [])
                if not alts:
                    raise ValueError(f"No itineraries for OD {od_key}")
                cost_values = []
                for it in alts:
                    comp = costs[it["id"]][t]
                    gen_cost = vot[group][t] * comp["TT"] + comp["Money"] + comp["ChargeCost"]
                    generalized_costs[it["id"]].setdefault(group, {})[t] = gen_cost
                    cost_values.append(-lambdas[group] * gen_cost)
                log_denom = logsumexp(cost_values)
                total_demand = time_map[t]
                for it, util in zip(alts, cost_values):
                    share = math.exp(util - log_denom)
                    flows[it["id"]].setdefault(group, {})[t] = total_demand * share
    return flows, generalized_costs


def aggregate_arc_flows(
    itineraries: List[Dict[str, Any]],
    flows: Dict[str, Dict[str, Dict[int, float]]],
    times: List[int],
) -> Dict[str, Dict[int, float]]:
    arc_flows: Dict[str, Dict[int, float]] = {}
    for it in itineraries:
        for seg in it.get("road_arcs", []):
            arc_flows.setdefault(seg["arc"], {t: 0.0 for t in times})
    for it in itineraries:
        for group, time_map in flows[it["id"]].items():
            for seg in it.get("road_arcs", []):
                arc = seg["arc"]
                t = seg["t"]
                arc_flows[arc][t] += seg.get("frac", 1.0) * time_map[t]
    return arc_flows


def aggregate_station_utilization(
    itineraries: List[Dict[str, Any]],
    flows: Dict[str, Dict[str, Dict[int, float]]],
    times: List[int],
) -> Dict[str, Dict[int, float]]:
    utilization: Dict[str, Dict[int, float]] = {}
    for it in itineraries:
        for stop in it.get("stations", []):
            utilization.setdefault(stop["station"], {t: 0.0 for t in times})
        if it.get("mode") == "eVTOL":
            dep_station = it.get("dep_station")
            if dep_station is not None:
                utilization.setdefault(dep_station, {t: 0.0 for t in times})
    for it in itineraries:
        for group, time_map in flows[it["id"]].items():
            for stop in it.get("stations", []):
                station = stop["station"]
                t = stop["t"]
                utilization[station][t] += time_map[t]
            if it.get("mode") == "eVTOL":
                dep_station = it.get("dep_station")
                if dep_station is not None:
                    for t in times:
                        utilization[dep_station][t] += time_map[t]
    return utilization


def aggregate_evtol_dep_demand(
    itineraries: List[Dict[str, Any]],
    flows: Dict[str, Dict[str, Dict[int, float]]],
    times: List[int],
) -> Dict[str, Dict[int, float]]:
    d_dep: Dict[str, Dict[int, float]] = {}
    for it in itineraries:
        if it.get("mode") != "eVTOL":
            continue
        dep_station = it.get("dep_station")
        if dep_station is None:
            continue
        d_dep.setdefault(dep_station, {t: 0.0 for t in times})
        for group, time_map in flows.get(it["id"], {}).items():
            for t in times:
                d_dep[dep_station][t] += time_map[t]
    return d_dep


def aggregate_ev_energy_demand(
    itineraries: List[Dict[str, Any]],
    flows: Dict[str, Dict[str, Dict[int, float]]],
    times: List[int],
) -> Dict[str, Dict[int, float]]:
    energy: Dict[str, Dict[int, float]] = {}
    for it in itineraries:
        for stop in it.get("stations", []):
            station = stop["station"]
            energy.setdefault(station, {t: 0.0 for t in times})
        for group, time_map in flows.get(it["id"], {}).items():
            for stop in it.get("stations", []):
                station = stop["station"]
                t = stop["t"]
                energy[station][t] += stop.get("energy", 0.0) * time_map[t]
    return energy


def aggregate_evtol_demand(
    flows: Dict[str, Dict[str, Dict[int, float]]],
    itineraries: List[Dict[str, Any]],
    times: List[int],
) -> Dict[str, Dict[int, float]]:
    d_route: Dict[str, Dict[int, float]] = {}
    for it in itineraries:
        if it.get("mode") != "eVTOL":
            continue
        it_id = it["id"]
        d_route[it_id] = {t: 0.0 for t in times}
        for group, time_map in flows.get(it_id, {}).items():
            for t in times:
                d_route[it_id][t] += time_map[t]
    return d_route


def compute_evtol_energy_demand(
    d_route: Dict[str, Dict[int, float]],
    itineraries: List[Dict[str, Any]],
    times: List[int],
) -> Dict[str, Dict[int, float]]:
    e_dep: Dict[str, Dict[int, float]] = {}
    for it in itineraries:
        if it.get("mode") != "eVTOL":
            continue
        it_id = it["id"]
        dep_station = it.get("dep_station")
        if dep_station is None:
            continue
        e_per_pax = it.get("e_per_pax", 0.0)
        e_dep.setdefault(dep_station, {t: 0.0 for t in times})
        for t in times:
            e_dep[dep_station][t] += d_route.get(it_id, {}).get(t, 0.0) * e_per_pax
    return e_dep
