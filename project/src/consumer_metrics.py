from __future__ import annotations

import copy
import math
from typing import Any, Dict, List

from .assignment import (
    classify_mode_label,
    classify_supermode,
    get_evtol_service_class,
    is_evtol_itinerary,
    is_multimodal_evtol,
)


def _finite_or_none(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _safe_ratio(a: float, b: float) -> float:
    return float(a) / max(float(b), 1.0e-12)


def _aggregate_weighted(entries: List[Dict[str, Any]], key: str, weight_key: str = "flow") -> float:
    num = 0.0
    den = 0.0
    for e in entries:
        w = float(e.get(weight_key, 0.0) or 0.0)
        v = e.get(key)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(fv):
            continue
        num += w * fv
        den += w
    return num / den if den > 1.0e-12 else 0.0


def _mode_labels() -> List[str]:
    return [
        "pure_EV",
        "pure_eVTOL_fast",
        "pure_eVTOL_slow",
        "multimodal_EV_to_eVTOL_fast",
        "multimodal_EV_to_eVTOL_slow",
    ]


def build_itinerary_consumer_metrics(
    itineraries: List[Dict[str, Any]],
    times: List[int],
    groups: List[str],
    costs: Dict[str, Dict[int, Dict[str, float]]],
    utility_breakdown: Dict[str, Dict[str, Dict[int, Dict[str, float]]]],
    flows: Dict[str, Dict[str, Dict[int, float]]],
) -> Dict[str, Dict[str, Dict[int, Dict[str, Any]]]]:
    """Build consumer-facing itinerary metrics by itinerary/group/time."""
    out: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {}
    by_id = {str(it.get("id")): it for it in itineraries}
    for it_id, _group_map in flows.items():
        it = by_id.get(it_id, {"id": it_id, "mode": "EV", "od": ["?", "?"]})
        out[it_id] = {}
        for g in groups:
            out[it_id][g] = {}
            for t in times:
                comp = costs.get(it_id, {}).get(t, {})
                cb = comp.get("cost_breakdown", {}) if isinstance(comp, dict) else {}
                ub = utility_breakdown.get(it_id, {}).get(g, {}).get(t, {})
                mode_label = str(cb.get("mode_label", classify_mode_label(it)))
                supermode = str(cb.get("supermode", classify_supermode(it)))
                service_class = str(cb.get("service_class", get_evtol_service_class(it) if is_evtol_itinerary(it) else "none"))
                raw_gc = _finite_or_none(ub.get("generalized_cost_raw", ub.get("raw_cost")))
                perceived_gc = _finite_or_none(ub.get("generalized_cost_perceived", ub.get("perceived_cost")))
                vt_prob = _finite_or_none(ub.get("vt_prob", 1.0))
                ev_prob = _finite_or_none(ub.get("ev_prob", 1.0))
                booking_ok = bool(ub.get("booking_feasibility_flag", False))
                money = _finite_or_none(comp.get("Money"))
                charge = _finite_or_none(comp.get("ChargeCost"))
                out[it_id][g][t] = {
                    "road_time": _finite_or_none(cb.get("road_time")),
                    "ev_station_wait_time": _finite_or_none(cb.get("ev_station_wait_time")),
                    "vt_departure_wait_time": _finite_or_none(cb.get("vt_departure_wait_time")),
                    "flight_time": _finite_or_none(cb.get("flight_time")),
                    "transfer_time": _finite_or_none(cb.get("transfer_time_applied")),
                    "total_travel_time": _finite_or_none(comp.get("TT")),
                    "base_money_fare": _finite_or_none(cb.get("base_money_fare")),
                    "access_energy_charge_cost": _finite_or_none(cb.get("access_energy_charge_cost", charge)),
                    "flight_energy_charge_cost": _finite_or_none(cb.get("flight_energy_charge_cost")),
                    "total_monetary_cost": _finite_or_none(cb.get("total_monetary_cost", (money or 0.0) + (charge or 0.0))),
                    "money_component": money,
                    "charge_cost_component": charge,
                    "generalized_cost_raw": raw_gc,
                    "generalized_cost_perceived": perceived_gc,
                    "raw_time_valuation_component": _finite_or_none((raw_gc - (money or 0.0) - (charge or 0.0)) if raw_gc is not None else None),
                    "reliability_adjustment": _finite_or_none((perceived_gc - raw_gc) if (raw_gc is not None and perceived_gc is not None) else None),
                    "vt_service_probability": vt_prob,
                    "ev_service_probability": ev_prob,
                    "service_probability_aux": min(vt_prob or 1.0, ev_prob or 1.0),
                    "booking_feasibility_flag": booking_ok,
                    "service_class": service_class,
                    "supermode": supermode,
                    "mode_label": mode_label,
                    "flow": float(flows.get(it_id, {}).get(g, {}).get(t, 0.0) or 0.0),
                    "od": f"{it.get('od', ['?', '?'])[0]}-{it.get('od', ['?', '?'])[1]}",
                    "itinerary_id": it_id,
                }
    return out


def _entries_for_group_time(
    itinerary_metrics: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]],
    group: str,
    t: int,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for it_map in itinerary_metrics.values():
        e = it_map.get(group, {}).get(t)
        if e is not None:
            entries.append(e)
    return entries


def build_group_mode_results(
    itinerary_metrics: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]],
    groups: List[str],
    times: List[int],
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    out: Dict[str, Dict[int, Dict[str, Any]]] = {g: {t: {} for t in times} for g in groups}
    for g in groups:
        for t in times:
            entries = _entries_for_group_time(itinerary_metrics, g, t)
            served = sum(float(e.get("flow", 0.0)) for e in entries)
            mode_mass = {m: 0.0 for m in _mode_labels()}
            super_mass = {"EV": 0.0, "eVTOL": 0.0, "EV_to_eVTOL": 0.0}
            for e in entries:
                m = str(e.get("mode_label", "pure_EV"))
                sm = str(e.get("supermode", "EV"))
                mode_mass[m] = mode_mass.get(m, 0.0) + float(e.get("flow", 0.0))
                super_mass[sm] = super_mass.get(sm, 0.0) + float(e.get("flow", 0.0))
            out[g][t] = {
                "mode_shares": {k: _safe_ratio(v, served) for k, v in mode_mass.items()},
                "supermode_shares": {k: _safe_ratio(v, served) for k, v in super_mass.items()},
                "served_flow_by_mode": mode_mass,
                "served_passengers": served,
            }
    return out


def build_group_cost_decomposition(
    itinerary_metrics: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]],
    groups: List[str],
    times: List[int],
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    out: Dict[str, Dict[int, Dict[str, Any]]] = {g: {t: {} for t in times} for g in groups}
    for g in groups:
        for t in times:
            entries = _entries_for_group_time(itinerary_metrics, g, t)
            out[g][t] = {
                "average_generalized_cost_raw": _aggregate_weighted(entries, "generalized_cost_raw"),
                "average_generalized_cost_perceived": _aggregate_weighted(entries, "generalized_cost_perceived"),
                "average_raw_time_valuation_component": _aggregate_weighted(entries, "raw_time_valuation_component"),
                "average_money_component": _aggregate_weighted(entries, "money_component"),
                "average_charge_cost_component": _aggregate_weighted(entries, "charge_cost_component"),
                "reliability_adjustment_aux": _aggregate_weighted(entries, "reliability_adjustment"),
                "service_probability_aux": _aggregate_weighted(entries, "service_probability_aux"),
            }
    return out


def build_group_time_decomposition(
    itinerary_metrics: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]],
    groups: List[str],
    times: List[int],
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    out: Dict[str, Dict[int, Dict[str, Any]]] = {g: {t: {} for t in times} for g in groups}
    for g in groups:
        for t in times:
            entries = _entries_for_group_time(itinerary_metrics, g, t)
            out[g][t] = {
                "road_time": _aggregate_weighted(entries, "road_time"),
                "ev_station_wait_time": _aggregate_weighted(entries, "ev_station_wait_time"),
                "vt_departure_wait_time": _aggregate_weighted(entries, "vt_departure_wait_time"),
                "flight_time": _aggregate_weighted(entries, "flight_time"),
                "transfer_time": _aggregate_weighted(entries, "transfer_time"),
                "total_travel_time": _aggregate_weighted(entries, "total_travel_time"),
            }
    return out


def build_group_monetary_decomposition(
    itinerary_metrics: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]],
    groups: List[str],
    times: List[int],
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    out: Dict[str, Dict[int, Dict[str, Any]]] = {g: {t: {} for t in times} for g in groups}
    for g in groups:
        for t in times:
            entries = _entries_for_group_time(itinerary_metrics, g, t)
            out[g][t] = {
                "base_money_fare": _aggregate_weighted(entries, "base_money_fare"),
                "access_energy_charge_cost": _aggregate_weighted(entries, "access_energy_charge_cost"),
                "flight_energy_charge_cost": _aggregate_weighted(entries, "flight_energy_charge_cost"),
                "total_monetary_cost": _aggregate_weighted(entries, "total_monetary_cost"),
            }
    return out


def build_behavior_summary(
    itinerary_metrics: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]],
    demand_q: Dict[str, Dict[str, Dict[int, float]]],
    groups: List[str],
    times: List[int],
    group_mode_results: Dict[str, Dict[int, Dict[str, Any]]],
    group_cost_decomp: Dict[str, Dict[int, Dict[str, Any]]],
    group_time_decomp: Dict[str, Dict[int, Dict[str, Any]]],
    group_money_decomp: Dict[str, Dict[int, Dict[str, Any]]],
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    out: Dict[str, Dict[int, Dict[str, Any]]] = {g: {t: {} for t in times} for g in groups}
    for g in groups:
        for t in times:
            total_demand = sum(float(od.get(g, {}).get(t, 0.0) or 0.0) for od in demand_q.values())
            mode_res = group_mode_results.get(g, {}).get(t, {})
            served = float(mode_res.get("served_passengers", 0.0))
            unserved = max(0.0, total_demand - served)
            mode_shares = mode_res.get("mode_shares", {})
            super_shares = mode_res.get("supermode_shares", {})
            dominant_super = max(super_shares.items(), key=lambda kv: kv[1])[0] if super_shares else "EV"
            dominant_mode = max(mode_shares.items(), key=lambda kv: kv[1])[0] if mode_shares else "pure_EV"
            fast_share = float(mode_shares.get("pure_eVTOL_fast", 0.0)) + float(mode_shares.get("multimodal_EV_to_eVTOL_fast", 0.0))
            slow_share = float(mode_shares.get("pure_eVTOL_slow", 0.0)) + float(mode_shares.get("multimodal_EV_to_eVTOL_slow", 0.0))
            out[g][t] = {
                "demand": total_demand,
                "served_demand": served,
                "unserved_demand": unserved,
                "EV_share": float(super_shares.get("EV", 0.0)),
                "eVTOL_share": float(super_shares.get("eVTOL", 0.0)),
                "EV_to_eVTOL_share": float(super_shares.get("EV_to_eVTOL", 0.0)),
                "fast_share": fast_share,
                "slow_share": slow_share,
                "dominant_supermode": dominant_super,
                "dominant_detailed_mode": dominant_mode,
                "average_generalized_cost_raw": float(group_cost_decomp.get(g, {}).get(t, {}).get("average_generalized_cost_raw", 0.0)),
                "average_generalized_cost_perceived": float(group_cost_decomp.get(g, {}).get(t, {}).get("average_generalized_cost_perceived", 0.0)),
                "average_total_travel_time": float(group_time_decomp.get(g, {}).get(t, {}).get("total_travel_time", 0.0)),
                "average_total_monetary_cost": float(group_money_decomp.get(g, {}).get(t, {}).get("total_monetary_cost", 0.0)),
            }
    return out


def build_vot_mechanism_summary(
    behavior_summary: Dict[str, Dict[int, Dict[str, Any]]],
    group_cost_decomp: Dict[str, Dict[int, Dict[str, Any]]],
    diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    groups = list(behavior_summary.keys())
    if not groups:
        return {
            "time_sensitive_groups": [],
            "price_sensitive_groups": [],
            "direct_evtol_high_vot_signal": "insufficient_data",
            "ev_to_evtol_intermediate_signal": "insufficient_data",
            "ev_low_vot_competitiveness_signal": "insufficient_data",
            "constraint_shift_signal": "insufficient_data",
        }

    avg_time_comp = {g: sum(float(v.get("average_raw_time_valuation_component", 0.0)) for v in group_cost_decomp.get(g, {}).values()) for g in groups}
    avg_money_comp = {g: sum(float(v.get("average_money_component", 0.0)) for v in group_cost_decomp.get(g, {}).values()) for g in groups}
    top_time = sorted(groups, key=lambda g: avg_time_comp.get(g, 0.0), reverse=True)
    top_money = sorted(groups, key=lambda g: avg_money_comp.get(g, 0.0), reverse=True)

    direct_evtol_signal: List[str] = []
    ev_comp_signal: List[str] = []
    for g in groups:
        vals = list(behavior_summary[g].values())
        evtol_share = sum(float(v.get("eVTOL_share", 0.0)) for v in vals) / max(1, len(vals))
        mm_share = sum(float(v.get("EV_to_eVTOL_share", 0.0)) for v in vals) / max(1, len(vals))
        ev_share = sum(float(v.get("EV_share", 0.0)) for v in vals) / max(1, len(vals))
        if evtol_share >= max(ev_share, mm_share):
            direct_evtol_signal.append(g)
        if ev_share >= max(evtol_share, mm_share):
            ev_comp_signal.append(g)

    return {
        "time_sensitive_groups": top_time[:2],
        "price_sensitive_groups": top_money[:2],
        "direct_evtol_high_vot_signal": direct_evtol_signal,
        "ev_to_evtol_intermediate_signal": [g for g in groups if g not in direct_evtol_signal and g not in ev_comp_signal],
        "ev_low_vot_competitiveness_signal": ev_comp_signal,
        "constraint_shift_signal": [
            f"bottleneck_summary={diagnostics.get('bottleneck_summary')}",
            f"power_binding_count={diagnostics.get('power_binding_count')}",
            f"aircraft_binding_count={diagnostics.get('aircraft_binding_count')}",
        ],
    }


def aggregate_consumer_metrics_by_group(
    itinerary_metrics: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]],
    demand_q: Dict[str, Dict[str, Dict[int, float]]],
    groups: List[str],
    times: List[int],
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Backward-compatible aggregate block; now auxiliary to behavior summaries."""
    out: Dict[str, Dict[int, Dict[str, Any]]] = {g: {t: {} for t in times} for g in groups}
    for g in groups:
        for t in times:
            entries = _entries_for_group_time(itinerary_metrics, g, t)
            served = sum(float(e.get("flow", 0.0)) for e in entries)
            total_demand = sum(float(gmap.get(g, {}).get(t, 0.0)) for gmap in demand_q.values())
            mode_mass: Dict[str, float] = {}
            for e in entries:
                mode_mass[e.get("mode_label", "unknown")] = mode_mass.get(e.get("mode_label", "unknown"), 0.0) + float(e.get("flow", 0.0))
            out[g][t] = {
                "served_passengers": served,
                "total_demand": total_demand,
                "booking_success_rate": _safe_ratio(served, total_demand) if total_demand > 1.0e-12 else 1.0,
                "average_door_to_door_travel_time": _aggregate_weighted(entries, "total_travel_time"),
                "average_fare": _aggregate_weighted(entries, "base_money_fare"),
                "average_monetary_payment": _aggregate_weighted(entries, "total_monetary_cost"),
                "average_generalized_cost": _aggregate_weighted(entries, "generalized_cost_perceived"),
                "average_waiting_time": _aggregate_weighted(entries, "ev_station_wait_time") + _aggregate_weighted(entries, "vt_departure_wait_time"),
                "average_transfer_burden": _aggregate_weighted(entries, "transfer_time"),
                "average_service_reliability_aux": _aggregate_weighted(entries, "service_probability_aux"),
                "mode_adoption_shares": {k: _safe_ratio(v, served) for k, v in mode_mass.items()} if served > 1.0e-12 else {},
            }
    return out


def aggregate_consumer_metrics_by_supermode(
    itinerary_metrics: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]],
    groups: List[str],
    times: List[int],
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    modes = ["EV", "eVTOL", "EV_to_eVTOL"]
    out: Dict[str, Dict[int, Dict[str, Any]]] = {m: {t: {} for t in times} for m in modes}
    for sm in modes:
        for t in times:
            entries: List[Dict[str, Any]] = []
            for it_map in itinerary_metrics.values():
                for g in groups:
                    e = it_map.get(g, {}).get(t)
                    if e and e.get("supermode") == sm:
                        entries.append(e)
            out[sm][t] = {
                "served_passengers": sum(float(e.get("flow", 0.0)) for e in entries),
                "average_travel_time": _aggregate_weighted(entries, "total_travel_time"),
                "average_fare": _aggregate_weighted(entries, "base_money_fare"),
                "average_reliability_aux": _aggregate_weighted(entries, "service_probability_aux"),
                "average_effective_energy_price_paid": _aggregate_weighted(entries, "access_energy_charge_cost"),
                "average_transfer_time": _aggregate_weighted(entries, "transfer_time"),
                "average_departure_wait": _aggregate_weighted(entries, "vt_departure_wait_time"),
            }
    return out


def aggregate_consumer_metrics_by_mode(
    itinerary_metrics: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]],
    groups: List[str],
    times: List[int],
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    labels = _mode_labels()
    out: Dict[str, Dict[int, Dict[str, Any]]] = {m: {t: {} for t in times} for m in labels}
    for m in labels:
        for t in times:
            entries: List[Dict[str, Any]] = []
            for it_map in itinerary_metrics.values():
                for g in groups:
                    e = it_map.get(g, {}).get(t)
                    if e and e.get("mode_label") == m:
                        entries.append(e)
            out[m][t] = {
                "served_passengers": sum(float(e.get("flow", 0.0)) for e in entries),
                "average_travel_time": _aggregate_weighted(entries, "total_travel_time"),
                "average_fare": _aggregate_weighted(entries, "base_money_fare"),
                "average_reliability_aux": _aggregate_weighted(entries, "service_probability_aux"),
            }
    return out


def build_station_time_consumer_signals(
    diagnostics: Dict[str, Any],
    times: List[int],
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    power_tightness = diagnostics.get("power_tightness", {})
    vt_waits = diagnostics.get("vt_departure_waits", {})
    out: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for s, tmap in power_tightness.items():
        out[s] = {}
        for t in times:
            e = tmap.get(t, {})
            fast_w = _finite_or_none((vt_waits.get(s, {}).get("fast", {}) or {}).get(t))
            slow_w = _finite_or_none((vt_waits.get(s, {}).get("slow", {}) or {}).get(t))
            ev_req = float(e.get("P_ev_req_kw", 0.0) or 0.0)
            vt_req = float(e.get("P_vt_req_kw_grid", 0.0) or 0.0)
            out[s][t] = {
                # mechanism-facing canonical fields
                "effective_price": _finite_or_none(e.get("effective_price")),
                "surcharge": _finite_or_none(e.get("surcharge_smoothed")),
                "total_power_request": ev_req + vt_req,
                "EV_power_request": ev_req,
                "VT_power_request": vt_req,
                "cap_binding_flag": bool(e.get("cap_binding", False)),
                "departure_wait_fast": fast_w,
                "departure_wait_slow": slow_w,
                "service_probability_aux": _finite_or_none(e.get("vt_service_prob")),
                # backward-compatible aliases
                "expected_wait": fast_w,
                "service_probability": _finite_or_none(e.get("vt_service_prob")),
                "energy_load": ev_req + vt_req,
                "power_tightness_flag": bool(e.get("cap_binding", False)),
            }
    return out


def build_service_design_summary(
    behavior_summary: Dict[str, Dict[int, Dict[str, Any]]],
    station_signals: Dict[str, Dict[int, Dict[str, Any]]],
    peak_t: int,
) -> Dict[str, Any]:
    vals = [v.get(peak_t, {}) for v in behavior_summary.values()]
    return {
        "peak_time_consumer_summary": {"peak_t": peak_t, "group_count": len(behavior_summary)},
        "consumer_accessibility_summary": {
            "avg_booking_success_rate_peak": sum(float(v.get("served_demand", 0.0)) / max(1.0e-12, float(v.get("demand", 0.0) or 1.0)) for v in vals) / max(1, len(vals)),
        },
        "reliability_summary": {
            "avg_service_reliability_aux_peak": sum(float(v.get("average_generalized_cost_perceived", 0.0) - v.get("average_generalized_cost_raw", 0.0)) for v in vals) / max(1, len(vals)),
        },
        "waiting_summary": {
            "avg_waiting_time_peak": sum(float(v.get("average_total_travel_time", 0.0)) for v in vals) / max(1, len(vals)),
        },
        "affordability_summary": {
            "avg_payment_peak": sum(float(v.get("average_total_monetary_cost", 0.0)) for v in vals) / max(1, len(vals)),
        },
        "station_consumer_signals": station_signals,
    }


def build_booking_platform_summary(
    itinerary_metrics: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]],
    groups: List[str],
    times: List[int],
    demand_q: Dict[str, Dict[str, Dict[int, float]]],
    wt_time: float,
    wt_fare: float,
    wt_rel: float,
    enabled: bool,
) -> Dict[str, Any]:
    if not enabled:
        return {"status": "disabled", "reason": "booking_recommendation_enabled=false"}

    by_group_time: Dict[str, Dict[int, Dict[str, Any]]] = {g: {t: {} for t in times} for g in groups}
    for g in groups:
        for t in times:
            cands: List[Dict[str, Any]] = []
            for _it_id, gmap in itinerary_metrics.items():
                e = gmap.get(g, {}).get(t)
                if e and bool(e.get("booking_feasibility_flag")):
                    cands.append(e)
            if not cands:
                by_group_time[g][t] = {
                    "feasible_itinerary_count": 0,
                    "recommended_itinerary": None,
                    "rejected_unserved_demand": sum(float(od.get(g, {}).get(t, 0.0)) for od in demand_q.values()),
                }
                continue

            cheapest = min(cands, key=lambda x: float(x.get("total_monetary_cost") or 1e18))
            fastest = min(cands, key=lambda x: float(x.get("total_travel_time") or 1e18))
            min_fare = min(float(c.get("total_monetary_cost") or 0.0) for c in cands)
            max_fare = max(float(c.get("total_monetary_cost") or 0.0) for c in cands)
            min_tt = min(float(c.get("total_travel_time") or 0.0) for c in cands)
            max_tt = max(float(c.get("total_travel_time") or 0.0) for c in cands)
            min_rel = min(float(c.get("service_probability_aux") or 0.0) for c in cands)
            max_rel = max(float(c.get("service_probability_aux") or 0.0) for c in cands)

            def _score(c: Dict[str, Any]) -> float:
                fare = float(c.get("total_monetary_cost") or 0.0)
                tt = float(c.get("total_travel_time") or 0.0)
                rel = float(c.get("service_probability_aux") or 0.0)
                fare_n = (fare - min_fare) / max(1.0e-9, max_fare - min_fare)
                tt_n = (tt - min_tt) / max(1.0e-9, max_tt - min_tt)
                rel_n = (rel - min_rel) / max(1.0e-9, max_rel - min_rel)
                return wt_time * tt_n + wt_fare * fare_n + wt_rel * (1.0 - rel_n)

            best_balanced = min(cands, key=_score)
            recommended = min(cands, key=lambda x: float(x.get("generalized_cost_perceived") or 1e18))
            by_group_time[g][t] = {
                "feasible_itinerary_count": len(cands),
                "cheapest_itinerary": cheapest.get("itinerary_id"),
                "fastest_itinerary": fastest.get("itinerary_id"),
                "best_balanced_itinerary": best_balanced.get("itinerary_id"),
                "recommended_itinerary": recommended.get("itinerary_id"),
                "recommended_explanation": {
                    "mode_label": recommended.get("mode_label"),
                    "supermode": recommended.get("supermode"),
                    "perceived_cost": recommended.get("generalized_cost_perceived"),
                    "travel_time": recommended.get("total_travel_time"),
                    "monetary_cost": recommended.get("total_monetary_cost"),
                    "service_probability_aux": recommended.get("service_probability_aux"),
                },
            }
    return {"status": "enabled", "by_group_time": by_group_time}


def build_parameter_registry(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    p = data.get("parameters", {})
    return {
        "core_mechanism_parameters": {
            "demand_by_group": p.get("q"),
            "VOT_by_group": p.get("VOT"),
            "lambda_by_group": p.get("lambda"),
            "common_lambda_override": config.get("common_lambda_override"),
            "itinerary_fare_time_structure": "itinerary.money + itinerary time components + eVTOL energy adders",
            "station_power_limits": {k: v.get("P_site") for k, v in p.get("stations", {}).items()},
            "vt_departure_capacities": p.get("vt_departure_capacity_total"),
            "congestion_core_parameters": p.get("arcs"),
        },
        "supporting_physical_parameters": {
            "storage_bounds": p.get("vertiport_storage"),
            "charging_efficiency": {k: v.get("eta_ch") for k, v in p.get("vertiport_storage", {}).items()},
            "vt_pax_per_departure_fast": p.get("vt_pax_per_departure_fast"),
            "vt_pax_per_departure_slow": p.get("vt_pax_per_departure_slow"),
            "vt_turnaround_lag": p.get("vt_turnaround_lag"),
        },
        "numerical_solver_parameters": {
            "shadow_price_caps": {
                "shadow_price_cap_mult": config.get("shadow_price_cap_mult"),
                "shadow_price_cap_abs": config.get("shadow_price_cap_abs"),
            },
            "reroute_temperature": config.get("reroute_logit_temperature"),
            "queue_coefficients": {
                "vt_queue_a_fast": config.get("vt_queue_a_fast"),
                "vt_queue_a_slow": config.get("vt_queue_a_slow"),
                "vt_queue_gamma_fast": config.get("vt_queue_gamma_fast"),
                "vt_queue_gamma_slow": config.get("vt_queue_gamma_slow"),
            },
            "convergence_controls": {
                "tol": config.get("tol"),
                "patience": config.get("patience"),
                "min_iters": config.get("min_iters"),
            },
            "msa_smoothing": {
                "surcharge_msa_alpha": config.get("surcharge_msa_alpha"),
                "flow_msa_alpha": config.get("flow_msa_alpha"),
                "surcharge_decay_factor": config.get("surcharge_decay_factor"),
            },
        },
    }


def build_tce_summary(vot_mechanism_summary: Dict[str, Any], diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "consumer_centric_findings": [
            "primary outputs emphasize group-wise behavior under heterogeneous VOT",
            "mode and supermode shares are decomposed by traveler group and time",
        ],
        "smart_city_integration_findings": [
            "station-level mechanism signals expose power-price-wait coupling",
            f"bottleneck_summary={diagnostics.get('bottleneck_summary')}",
        ],
        "charging_and_energy_findings": [
            "effective price and surcharge are tied to shared-power tightness",
            "group-level cost decomposition links monetary and time components",
        ],
        "service_design_findings": [
            "platform recommendation outputs are secondary/optional",
            f"platform_recommendation_enabled={diagnostics.get('parameter_effective', {}).get('booking_recommendation_enabled', 'unknown')}",
            f"vot_mechanism_keys={list(vot_mechanism_summary.keys())}",
        ],
    }


def build_consumer_output_bundle(
    *,
    data: Dict[str, Any],
    config: Dict[str, Any],
    itineraries: List[Dict[str, Any]],
    times: List[int],
    groups: List[str],
    costs: Dict[str, Dict[int, Dict[str, float]]],
    logit_details: Dict[str, Any],
    flows: Dict[str, Dict[str, Dict[int, float]]],
    diagnostics: Dict[str, Any],
    output_full_json: bool,
) -> Dict[str, Any]:
    utility_breakdown = logit_details.get("utility_breakdown", {})
    itinerary_metrics = build_itinerary_consumer_metrics(itineraries, times, groups, costs, utility_breakdown, flows)

    group_mode_results = build_group_mode_results(itinerary_metrics, groups, times)
    group_cost_decomp = build_group_cost_decomposition(itinerary_metrics, groups, times)
    group_time_decomp = build_group_time_decomposition(itinerary_metrics, groups, times)
    group_money_decomp = build_group_monetary_decomposition(itinerary_metrics, groups, times)
    behavior_summary = build_behavior_summary(
        itinerary_metrics,
        data.get("parameters", {}).get("q", {}),
        groups,
        times,
        group_mode_results,
        group_cost_decomp,
        group_time_decomp,
        group_money_decomp,
    )
    vot_mechanism_summary = build_vot_mechanism_summary(behavior_summary, group_cost_decomp, diagnostics)

    # Keep prior sections for backward compatibility (secondary)
    by_group_legacy = aggregate_consumer_metrics_by_group(itinerary_metrics, data.get("parameters", {}).get("q", {}), groups, times)
    by_supermode_legacy = aggregate_consumer_metrics_by_supermode(itinerary_metrics, groups, times)
    by_mode_legacy = aggregate_consumer_metrics_by_mode(itinerary_metrics, groups, times)

    station_signals = build_station_time_consumer_signals(diagnostics, times)
    peak_t = int(diagnostics.get("peak_t", times[0] if times else 0))
    service_summary = build_service_design_summary(behavior_summary, station_signals, peak_t)

    platform = build_booking_platform_summary(
        itinerary_metrics,
        groups,
        times,
        data.get("parameters", {}).get("q", {}),
        float(config.get("best_balanced_weight_time", 0.4)),
        float(config.get("best_balanced_weight_fare", 0.3)),
        float(config.get("best_balanced_weight_reliability", 0.3)),
        enabled=bool(config.get("booking_recommendation_enabled", False)),
    )

    consumer_metrics: Dict[str, Any] = {
        "by_group_time": by_group_legacy,
        "by_supermode_time": by_supermode_legacy,
        "by_mode_time": by_mode_legacy,
    }
    if output_full_json and bool(config.get("report_consumer_itinerary_details", True)):
        consumer_metrics["itinerary_level_metrics"] = itinerary_metrics

    return {
        "consumer_metrics": consumer_metrics,
        "service_summary": service_summary,
        "platform_recommendations": platform,
        "tce_summary": build_tce_summary(vot_mechanism_summary, diagnostics),
        "parameter_registry": build_parameter_registry(data, config),
        # New primary behavioral outputs
        "behavior_summary": behavior_summary,
        "group_mode_results": group_mode_results,
        "group_cost_decomposition": group_cost_decomp,
        "group_time_decomposition": group_time_decomp,
        "group_monetary_decomposition": group_money_decomp,
        "vot_mechanism_summary": vot_mechanism_summary,
    }


def apply_named_scenario(data: Dict[str, Any], scenario_name: str) -> Dict[str, Any]:
    """Return a copied data object with interpretable scenario modifications."""
    d = copy.deepcopy(data)
    name = str(scenario_name).strip().lower()
    if name in {"", "base"}:
        return d

    params = d.setdefault("parameters", {})
    config = d.setdefault("config", {})

    if name == "power_upgrade":
        for s, sp in params.get("stations", {}).items():
            if s in d.get("sets", {}).get("hybrid_stations", []):
                for t, v in list(sp.get("P_site", {}).items()):
                    sp["P_site"][t] = float(v) * 1.1
    elif name == "departure_capacity_upgrade":
        for blk in ["vt_departure_capacity_total", "vt_departure_capacity_fast"]:
            for _s, tm in params.get(blk, {}).items():
                for t, v in list(tm.items()):
                    tm[t] = float(v) * 1.15
    elif name == "direct_fare_increase":
        for it in d.get("itineraries", []):
            if classify_supermode(it) == "eVTOL":
                it["money"] = float(it.get("money", 0.0)) * 1.1
    elif name == "park_and_fly_discount":
        for it in d.get("itineraries", []):
            if classify_supermode(it) == "EV_to_eVTOL":
                it["money"] = float(it.get("money", 0.0)) * 0.9
    elif name == "reliability_improvement":
        config["vt_queue_a_fast"] = float(config.get("vt_queue_a_fast", 1.0)) * 0.9
        config["vt_queue_a_slow"] = float(config.get("vt_queue_a_slow", 1.0)) * 0.9
        config["vt_reliability_floor"] = min(1.0, float(config.get("vt_reliability_floor", 0.05)) + 0.05)
    else:
        raise ValueError(f"Unknown scenario name: {scenario_name}")

    return d


def build_tce_scenarios(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    scenarios = [
        "base",
        "power_upgrade",
        "departure_capacity_upgrade",
        "direct_fare_increase",
        "park_and_fly_discount",
        "reliability_improvement",
    ]
    return {name: apply_named_scenario(data, name) for name in scenarios}
