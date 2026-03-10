import math
from typing import Any, Dict, List, Tuple

from .utils import logsumexp


EVTOL_MODES = {"eVTOL", "eVTOL_fast", "eVTOL_slow", "EV_to_eVTOL_fast", "EV_to_eVTOL_slow"}
MULTIMODAL_MODES = {"EV_to_eVTOL_fast", "EV_to_eVTOL_slow"}


def _value_by_time(x: Any, t: int, default: float = 0.0) -> float:
    if isinstance(x, dict):
        if t in x:
            return float(x[t])
        ts = str(t)
        if ts in x:
            return float(x[ts])
        return float(default)
    if x is None:
        return float(default)
    return float(x)


def is_multimodal_evtol(it: Dict[str, Any]) -> bool:
    return str(it.get("mode", "")) in MULTIMODAL_MODES


def is_pure_evtol(it: Dict[str, Any]) -> bool:
    return str(it.get("mode", "")) in {"eVTOL", "eVTOL_fast", "eVTOL_slow"}


def is_evtol_itinerary(it: Dict[str, Any]) -> bool:
    mode = str(it.get("mode", ""))
    if mode in EVTOL_MODES:
        return True
    return str(it.get("service_class", "")).lower() in {"fast", "slow"}


def get_evtol_service_class(it: Dict[str, Any]) -> str:
    mode = str(it.get("mode", ""))
    if mode.endswith("_fast"):
        return "fast"
    if mode.endswith("_slow"):
        return "slow"
    cls = str(it.get("service_class", "")).lower()
    if cls in {"fast", "slow"}:
        return cls
    return "slow"


def _road_segments(it: Dict[str, Any]) -> List[Dict[str, Any]]:
    segs = []
    segs.extend(it.get("road_arcs", []))
    segs.extend(it.get("access_arcs", []))
    segs.extend(it.get("egress_arcs", []))
    return segs


def _ev_stops(it: Dict[str, Any]) -> List[Dict[str, Any]]:
    stops = []
    if str(it.get("mode")) == "EV":
        stops.extend(it.get("stations", []))
    if is_multimodal_evtol(it):
        stops.extend(it.get("access_stations", []))
        if not it.get("access_stations") and it.get("stations"):
            stops.extend(it.get("stations", []))
    return stops


def build_incidence(
    itineraries: List[Dict[str, Any]],
    arcs: List[str],
    stations: List[str],
    times: List[int],
) -> Tuple[Dict[str, Dict[str, Dict[int, float]]], Dict[str, Dict[str, Dict[int, float]]]]:
    arc_set = set(arcs)
    station_set = set(stations)
    time_set = set(times)
    inc_road = {arc: {it["id"]: {t: 0.0 for t in times} for it in itineraries} for arc in arcs}
    inc_station = {s: {it["id"]: {t: 0.0 for t in times} for it in itineraries} for s in stations}
    for it in itineraries:
        it_id = it.get("id", "<unknown>")
        for seg in _road_segments(it):
            arc = seg["arc"]
            t = seg["t"]
            if arc not in arc_set:
                raise KeyError(f"Unknown arc in itinerary {it_id}: arc={arc}")
            if t not in time_set:
                raise KeyError(f"Unknown time in itinerary {it_id}: t={t}")
            frac = seg.get("frac", 1.0)
            inc_road[arc][it["id"]][t] += frac
        for stop in _ev_stops(it):
            station = stop["station"]
            t = stop["t"]
            if station not in station_set:
                raise KeyError(f"Unknown station in itinerary {it_id}: station={station}")
            if t not in time_set:
                raise KeyError(f"Unknown time in itinerary {it_id}: t={t}")
            inc_station[station][it["id"]][t] += 1.0
        if is_evtol_itinerary(it):
            dep_station = it.get("dep_station")
            if dep_station is not None:
                if dep_station not in station_set:
                    raise KeyError(f"Unknown dep_station in itinerary {it_id}: dep_station={dep_station}")
                for t in times:
                    if _value_by_time(it.get("flight_time", {}), t, 0.0) > 0.0:
                        inc_station[dep_station][it["id"]][t] += 1.0
    return inc_road, inc_station


def compute_itinerary_costs(
    itineraries: List[Dict[str, Any]],
    travel_times: Dict[str, Dict[int, float]],
    ev_station_waits: Dict[str, Dict[int, float]],
    electricity_price: Dict[str, Dict[int, float]],
    times: List[int],
    vt_departure_waits: Dict[str, Dict[str, Dict[int, float]]] | None = None,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    costs: Dict[str, Dict[int, Dict[str, float]]] = {it["id"]: {} for it in itineraries}
    for it in itineraries:
        it_id = it.get("id", "<unknown>")
        phi_markup = float(it.get("phi_energy_markup", 1.0))
        dep_station = it.get("dep_station")
        for t in times:
            money_t = _value_by_time(it.get("money", 0.0), t, 0.0)
            tt = 0.0
            charge_cost = 0.0
            for seg in _road_segments(it):
                if seg["t"] != t:
                    continue
                arc = seg["arc"]
                if arc not in travel_times or t not in travel_times[arc]:
                    raise KeyError(f"Missing travel_times for itinerary {it_id}, arc={arc}, t={t}")
                tt += float(travel_times[arc][t]) * float(seg.get("frac", 1.0))

            # EV component (pure EV or multimodal access)
            for stop in _ev_stops(it):
                if stop.get("t") != t:
                    continue
                station = stop.get("station")
                if station not in ev_station_waits or t not in ev_station_waits[station]:
                    raise KeyError(f"Missing ev_station_waits for itinerary {it_id}, station={station}, t={t}")
                if station not in electricity_price or t not in electricity_price[station]:
                    raise KeyError(f"Missing electricity_price for itinerary {it_id}, station={station}, t={t}")
                tt += float(ev_station_waits[station][t])
                charge_cost += float(stop.get("energy", 0.0)) * float(electricity_price[station][t])

            # Optional explicit access energy (multimodal)
            access_energy_kwh = _value_by_time(it.get("access_energy_kwh", 0.0), t, 0.0)
            if access_energy_kwh > 0.0:
                ref_station = dep_station
                if ref_station is None and _ev_stops(it):
                    ref_station = _ev_stops(it)[0].get("station")
                if ref_station is not None and ref_station in electricity_price and t in electricity_price[ref_station]:
                    charge_cost += access_energy_kwh * float(electricity_price[ref_station][t])

            if is_evtol_itinerary(it) and dep_station is not None:
                flight_time = _value_by_time(it.get("flight_time", {}), t, 0.0)
                if flight_time <= 0.0:
                    costs[it_id][t] = {"TT": float("inf"), "Money": float("inf"), "ChargeCost": 0.0}
                    continue
                tt += flight_time
                svc_class = get_evtol_service_class(it)
                if vt_departure_waits is not None:
                    if dep_station not in vt_departure_waits or svc_class not in vt_departure_waits[dep_station] or t not in vt_departure_waits[dep_station][svc_class]:
                        raise KeyError(f"Missing vt_departure_waits for itinerary {it_id}, dep_station={dep_station}, class={svc_class}, t={t}")
                    tt += float(vt_departure_waits[dep_station][svc_class][t])
                e_per_pax = _value_by_time(it.get("e_per_pax", 0.0), t, 0.0)
                if dep_station not in electricity_price or t not in electricity_price[dep_station]:
                    raise KeyError(f"Missing electricity_price for itinerary {it_id}, dep_station={dep_station}, t={t}")
                money_t += phi_markup * e_per_pax * float(electricity_price[dep_station][t])

            costs[it_id][t] = {"TT": tt, "Money": money_t, "ChargeCost": charge_cost}
    return costs


def logit_assignment(
    itineraries: List[Dict[str, Any]],
    costs: Dict[str, Dict[int, Dict[str, float]]],
    demand: Dict[str, Dict[str, Dict[int, float]]],
    vot: Dict[str, Dict[int, float]],
    lambdas: Dict[str, float],
    times: List[int],
    vt_service_prob: Dict[str, Dict[int, float]] | None = None,
    ev_service_prob: Dict[str, Dict[int, float]] | None = None,
    vt_service_prob_floor: float = 1.0e-4,
    ev_service_prob_floor: float = 1.0e-4,
    vt_reliability_gamma: float = 0.0,
    ev_reliability_gamma: float = 0.0,
    vt_service_prob_skip_below: float = 0.0,
    ev_service_prob_skip_below: float = 0.0,
) -> Tuple[Dict[str, Dict[str, Dict[int, float]]], Dict[str, Dict[str, Dict[int, float]]]]:
    all_groups = sorted({g for od_groups in demand.values() for g in od_groups.keys()})
    flows: Dict[str, Dict[str, Dict[int, float]]] = {
        it["id"]: {group: {t: 0.0 for t in times} for group in all_groups} for it in itineraries
    }
    generalized_costs: Dict[str, Dict[str, Dict[int, float]]] = {
        it["id"]: {group: {t: 0.0 for t in times} for group in all_groups} for it in itineraries
    }
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
                available_alts = []
                for it in alts:
                    if is_evtol_itinerary(it) and _value_by_time(it.get("flight_time", {}), t, 0.0) <= 0.0:
                        continue
                    available_alts.append(it)
                if not available_alts:
                    raise ValueError(f"No available itineraries for OD {od_key} at t={t}")

                feasible_alts = []
                for it in available_alts:
                    comp = costs[it["id"]][t]
                    gen_cost = float(vot[group][t]) * comp["TT"] + comp["Money"] + comp["ChargeCost"]
                    generalized_costs[it["id"]][group][t] = gen_cost
                    if math.isinf(gen_cost):
                        continue

                    vt_prob = 1.0
                    if is_evtol_itinerary(it):
                        dep_station = it.get("dep_station")
                        if vt_service_prob and dep_station in vt_service_prob:
                            vt_prob = float(vt_service_prob[dep_station].get(t, 1.0))
                        vt_prob = min(1.0, max(vt_service_prob_floor, vt_prob))
                        if vt_service_prob_skip_below > 0.0 and vt_prob < vt_service_prob_skip_below:
                            continue

                    ev_prob = 1.0
                    if ev_service_prob is not None:
                        ev_candidates = []
                        for stop in _ev_stops(it):
                            if stop.get("t") != t:
                                continue
                            station = stop.get("station")
                            if station in ev_service_prob:
                                ev_candidates.append(float(ev_service_prob[station].get(t, 1.0)))
                        if ev_candidates:
                            ev_prob = min(ev_candidates)
                    ev_prob = min(1.0, max(ev_service_prob_floor, ev_prob))
                    if (str(it.get("mode", "")) == "EV" or is_multimodal_evtol(it)) and ev_service_prob_skip_below > 0.0 and ev_prob < ev_service_prob_skip_below:
                        continue

                    utility = (
                        vt_reliability_gamma * math.log(max(vt_prob, vt_service_prob_floor))
                        + ev_reliability_gamma * math.log(max(ev_prob, ev_service_prob_floor))
                        - lambdas[group] * gen_cost
                    )
                    feasible_alts.append((it, utility))

                total_demand = time_map.get(t, 0.0)
                if total_demand <= 0.0 or not feasible_alts:
                    continue
                log_denom = logsumexp(util for _, util in feasible_alts)
                for it, util in feasible_alts:
                    flows[it["id"]][group][t] = total_demand * math.exp(util - log_denom)
    return flows, generalized_costs


def aggregate_arc_flows(
    itineraries: List[Dict[str, Any]],
    flows: Dict[str, Dict[str, Dict[int, float]]],
    times: List[int],
) -> Dict[str, Dict[int, float]]:
    arc_flows: Dict[str, Dict[int, float]] = {}
    for it in itineraries:
        for seg in _road_segments(it):
            arc_flows.setdefault(seg["arc"], {t: 0.0 for t in times})
    for it in itineraries:
        for _, time_map in flows[it["id"]].items():
            for seg in _road_segments(it):
                arc = seg["arc"]
                t = seg["t"]
                arc_flows[arc][t] += float(seg.get("frac", 1.0)) * float(time_map.get(t, 0.0))
    return arc_flows


def aggregate_ev_station_utilization(
    itineraries: List[Dict[str, Any]],
    flows: Dict[str, Dict[str, Dict[int, float]]],
    times: List[int],
) -> Dict[str, Dict[int, float]]:
    utilization: Dict[str, Dict[int, float]] = {}
    for it in itineraries:
        for stop in _ev_stops(it):
            utilization.setdefault(stop["station"], {t: 0.0 for t in times})
    for it in itineraries:
        for _, time_map in flows.get(it.get("id"), {}).items():
            for stop in _ev_stops(it):
                station = stop["station"]
                t = stop["t"]
                utilization[station][t] += float(time_map.get(t, 0.0))
    return utilization


def aggregate_station_utilization(
    itineraries: List[Dict[str, Any]],
    flows: Dict[str, Dict[str, Dict[int, float]]],
    times: List[int],
) -> Dict[str, Dict[int, float]]:
    utilization = aggregate_ev_station_utilization(itineraries, flows, times)
    for it in itineraries:
        if not is_evtol_itinerary(it):
            continue
        dep_station = it.get("dep_station")
        if dep_station is None:
            continue
        utilization.setdefault(dep_station, {t: 0.0 for t in times})
        for _, time_map in flows.get(it.get("id"), {}).items():
            for t in times:
                if _value_by_time(it.get("flight_time", {}), t, 0.0) > 0.0:
                    utilization[dep_station][t] += float(time_map.get(t, 0.0))
    return utilization


def aggregate_evtol_dep_demand(
    itineraries: List[Dict[str, Any]],
    flows: Dict[str, Dict[str, Dict[int, float]]],
    times: List[int],
) -> Dict[str, Dict[int, float]]:
    d_dep: Dict[str, Dict[int, float]] = {}
    for it in itineraries:
        if not is_evtol_itinerary(it):
            continue
        dep_station = it.get("dep_station")
        if dep_station is None:
            continue
        d_dep.setdefault(dep_station, {t: 0.0 for t in times})
        for _, time_map in flows.get(it["id"], {}).items():
            for t in times:
                d_dep[dep_station][t] += float(time_map.get(t, 0.0))
    return d_dep


def aggregate_ev_energy_demand(
    itineraries: List[Dict[str, Any]],
    flows: Dict[str, Dict[str, Dict[int, float]]],
    times: List[int],
) -> Dict[str, Dict[int, float]]:
    energy: Dict[str, Dict[int, float]] = {}
    for it in itineraries:
        for stop in _ev_stops(it):
            energy.setdefault(stop["station"], {t: 0.0 for t in times})
    for it in itineraries:
        for _, time_map in flows.get(it.get("id"), {}).items():
            for stop in _ev_stops(it):
                station = stop["station"]
                t = stop["t"]
                energy[station][t] += float(stop.get("energy", 0.0)) * float(time_map.get(t, 0.0))
            for t in times:
                access_energy = _value_by_time(it.get("access_energy_kwh", 0.0), t, 0.0)
                if access_energy <= 0.0:
                    continue
                dep_station = it.get("dep_station")
                if dep_station is None:
                    continue
                energy.setdefault(dep_station, {tt: 0.0 for tt in times})
                energy[dep_station][t] += access_energy * float(time_map.get(t, 0.0))
    return energy


def aggregate_evtol_demand(
    flows: Dict[str, Dict[str, Dict[int, float]]],
    itineraries: List[Dict[str, Any]],
    times: List[int],
) -> Dict[str, Dict[int, float]]:
    d_route: Dict[str, Dict[int, float]] = {}
    for it in itineraries:
        if not is_evtol_itinerary(it):
            continue
        it_id = it["id"]
        d_route[it_id] = {t: 0.0 for t in times}
        for _, time_map in flows.get(it_id, {}).items():
            for t in times:
                d_route[it_id][t] += float(time_map.get(t, 0.0))
    return d_route


def compute_evtol_energy_demand(
    d_route: Dict[str, Dict[int, float]],
    itineraries: List[Dict[str, Any]],
    times: List[int],
) -> Dict[str, Dict[int, float]]:
    e_dep: Dict[str, Dict[int, float]] = {}
    for it in itineraries:
        if not is_evtol_itinerary(it):
            continue
        dep_station = it.get("dep_station")
        if dep_station is None:
            continue
        e_dep.setdefault(dep_station, {t: 0.0 for t in times})
        for t in times:
            e_dep[dep_station][t] += d_route.get(it["id"], {}).get(t, 0.0) * _value_by_time(it.get("e_per_pax", 0.0), t, 0.0)
    return e_dep


def aggregate_vt_departure_flow_by_class(
    itineraries: List[Dict[str, Any]],
    flows: Dict[str, Dict[str, Dict[int, float]]],
    times: List[int],
) -> Dict[str, Dict[str, Dict[int, float]]]:
    out: Dict[str, Dict[str, Dict[int, float]]] = {}
    for it in itineraries:
        if not is_evtol_itinerary(it):
            continue
        dep_station = it.get("dep_station")
        if dep_station is None:
            continue
        cls = get_evtol_service_class(it)
        out.setdefault(dep_station, {}).setdefault(cls, {t: 0.0 for t in times})
        for _, time_map in flows.get(it.get("id"), {}).items():
            for t in times:
                out[dep_station][cls][t] += float(time_map.get(t, 0.0))
    for dep in out:
        out[dep].setdefault("fast", {t: 0.0 for t in times})
        out[dep].setdefault("slow", {t: 0.0 for t in times})
    return out
