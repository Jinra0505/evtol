from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Tuple

from .assignment import classify_mode_label, classify_supermode, get_evtol_service_class, is_evtol_itinerary, is_multimodal_evtol


Numeric = float | int


def _finite_or_none(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isfinite(v):
        return v
    return None


def _safe_ratio(a: float, b: float) -> float:
    return float(a) / max(float(b), 1.0e-12)


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
    for it_id, group_map in flows.items():
        it = by_id.get(it_id, {"id": it_id, "mode": "EV"})
        out[it_id] = {}
        for g in groups:
            out[it_id][g] = {}
            for t in times:
                comp = costs.get(it_id, {}).get(t, {})
                cb = comp.get("cost_breakdown", {}) if isinstance(comp, dict) else {}
                ub = utility_breakdown.get(it_id, {}).get(g, {}).get(t, {})
                road_time = _finite_or_none(cb.get("road_time"))
                ev_wait = _finite_or_none(cb.get("ev_station_wait_time"))
                vt_wait = _finite_or_none(cb.get("vt_departure_wait_time"))
                flight_time = _finite_or_none(cb.get("flight_time"))
                transfer_time = _finite_or_none(cb.get("transfer_time_applied"))
                total_tt = _finite_or_none(comp.get("TT"))
                base_fare = _finite_or_none(cb.get("base_money_fare"))
                access_energy_cost = _finite_or_none(cb.get("access_energy_charge_cost", comp.get("ChargeCost")))
                flight_energy_cost = _finite_or_none(cb.get("flight_energy_charge_cost"))
                total_money_cost = _finite_or_none(cb.get("total_monetary_cost", (comp.get("Money", 0.0) + comp.get("ChargeCost", 0.0))))
                raw_gc = _finite_or_none(ub.get("raw_cost"))
                perceived_gc = _finite_or_none(ub.get("perceived_cost"))
                vt_prob = _finite_or_none(ub.get("vt_prob", 1.0))
                ev_prob = _finite_or_none(ub.get("ev_prob", 1.0))
                booking_ok = bool(ub.get("booking_feasibility_flag", False))
                out[it_id][g][t] = {
                    "road_time": road_time,
                    "ev_station_wait_time": ev_wait,
                    "vt_departure_wait_time": vt_wait,
                    "flight_time": flight_time,
                    "transfer_time": transfer_time,
                    "total_travel_time": total_tt,
                    "base_money_fare": base_fare,
                    "access_energy_charge_cost": access_energy_cost,
                    "flight_energy_charge_cost": flight_energy_cost,
                    "total_monetary_cost": total_money_cost,
                    "generalized_cost_raw": raw_gc,
                    "generalized_cost_perceived": perceived_gc,
                    "vt_service_probability": vt_prob,
                    "ev_service_probability": ev_prob,
                    "booking_feasibility_flag": booking_ok,
                    "service_class": get_evtol_service_class(it) if is_evtol_itinerary(it) else "none",
                    "supermode": classify_supermode(it),
                    "mode_label": classify_mode_label(it),
                    "flow": float(flows.get(it_id, {}).get(g, {}).get(t, 0.0)),
                    "od": f"{it.get('od', ['?','?'])[0]}-{it.get('od', ['?','?'])[1]}",
                    "itinerary_id": it_id,
                }
    return out


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


def aggregate_consumer_metrics_by_group(
    itinerary_metrics: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]],
    demand_q: Dict[str, Dict[str, Dict[int, float]]],
    groups: List[str],
    times: List[int],
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    out: Dict[str, Dict[int, Dict[str, Any]]] = {g: {t: {} for t in times} for g in groups}
    for g in groups:
        for t in times:
            entries: List[Dict[str, Any]] = []
            mode_mass: Dict[str, float] = {}
            for it_map in itinerary_metrics.values():
                e = it_map.get(g, {}).get(t)
                if e is None:
                    continue
                entries.append(e)
                mode_mass[e.get("mode_label", "unknown")] = mode_mass.get(e.get("mode_label", "unknown"), 0.0) + float(e.get("flow", 0.0))
            served = sum(float(e.get("flow", 0.0)) for e in entries)
            total_demand = sum(float(gmap.get(g, {}).get(t, 0.0)) for gmap in demand_q.values())
            mode_shares = {k: _safe_ratio(v, served) for k, v in mode_mass.items()} if served > 1.0e-12 else {}
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
                "average_service_reliability": _aggregate_weighted(entries, "vt_service_probability"),
                "mode_adoption_shares": mode_shares,
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
                "average_reliability": _aggregate_weighted(entries, "vt_service_probability"),
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
    mode_labels = sorted({
        str(e.get("mode_label"))
        for it_map in itinerary_metrics.values()
        for g in groups
        for t in times
        for e in [it_map.get(g, {}).get(t)]
        if e is not None
    })
    out: Dict[str, Dict[int, Dict[str, Any]]] = {m: {t: {} for t in times} for m in mode_labels}
    for m in mode_labels:
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
                "average_reliability": _aggregate_weighted(entries, "vt_service_probability"),
            }
    return out


def build_service_design_summary(
    consumer_by_group: Dict[str, Dict[int, Dict[str, Any]]],
    station_signals: Dict[str, Dict[int, Dict[str, Any]]],
    peak_t: int,
) -> Dict[str, Any]:
    waiting = [v.get(peak_t, {}).get("average_waiting_time", 0.0) for v in consumer_by_group.values()]
    reliab = [v.get(peak_t, {}).get("average_service_reliability", 0.0) for v in consumer_by_group.values()]
    money = [v.get(peak_t, {}).get("average_monetary_payment", 0.0) for v in consumer_by_group.values()]
    return {
        "peak_time_consumer_summary": {"peak_t": peak_t, "group_count": len(consumer_by_group)},
        "consumer_accessibility_summary": {
            "avg_booking_success_rate_peak": sum(v.get(peak_t, {}).get("booking_success_rate", 0.0) for v in consumer_by_group.values()) / max(1, len(consumer_by_group)),
        },
        "reliability_summary": {"avg_service_reliability_peak": sum(reliab) / max(1, len(reliab))},
        "waiting_summary": {"avg_waiting_time_peak": sum(waiting) / max(1, len(waiting))},
        "affordability_summary": {"avg_payment_peak": sum(money) / max(1, len(money))},
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
) -> Dict[str, Any]:
    by_group_time: Dict[str, Dict[int, Dict[str, Any]]] = {g: {t: {} for t in times} for g in groups}

    for g in groups:
        for t in times:
            cands: List[Dict[str, Any]] = []
            for it_id, gmap in itinerary_metrics.items():
                e = gmap.get(g, {}).get(t)
                if not e:
                    continue
                if e.get("booking_feasibility_flag"):
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
            min_rel = min(float(c.get("vt_service_probability") or 0.0) for c in cands)
            max_rel = max(float(c.get("vt_service_probability") or 0.0) for c in cands)

            def _score(c: Dict[str, Any]) -> float:
                fare = float(c.get("total_monetary_cost") or 0.0)
                tt = float(c.get("total_travel_time") or 0.0)
                rel = float(c.get("vt_service_probability") or 0.0)
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
                    "reliability": recommended.get("vt_service_probability"),
                },
                "comparison_table": [
                    {
                        "itinerary_id": c.get("itinerary_id"),
                        "mode_label": c.get("mode_label"),
                        "perceived_cost": c.get("generalized_cost_perceived"),
                        "total_travel_time": c.get("total_travel_time"),
                        "total_monetary_cost": c.get("total_monetary_cost"),
                        "reliability": c.get("vt_service_probability"),
                    }
                    for c in cands
                ],
            }
    return {"by_group_time": by_group_time}


def build_station_time_consumer_signals(
    diagnostics: Dict[str, Any],
    times: List[int],
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    power_tightness = diagnostics.get("power_tightness", {})
    out: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for s, tmap in power_tightness.items():
        out[s] = {}
        for t in times:
            e = tmap.get(t, {})
            out[s][t] = {
                "effective_price": e.get("effective_price"),
                "surcharge": e.get("surcharge_smoothed"),
                "expected_wait": e.get("vt_service_prob"),
                "service_probability": e.get("vt_service_prob"),
                "energy_load": e.get("P_ev_req_kw"),
                "power_tightness_flag": bool(e.get("cap_binding", False)),
            }
    return out


def build_parameter_registry(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    p = data.get("parameters", {})
    return {
        "core_mechanism_parameters": {
            "demand_by_group": p.get("q"),
            "VOT": p.get("VOT"),
            "lambda": p.get("lambda"),
            "common_lambda_override": config.get("common_lambda_override"),
            "money_fields_in_itineraries": "itinerary.money and eVTOL energy adders",
            "station_power_limits": {k: v.get("P_site") for k, v in p.get("stations", {}).items()},
            "departure_capacities": p.get("vt_departure_capacity_total"),
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


def build_tce_summary(
    consumer_by_group: Dict[str, Dict[int, Dict[str, Any]]],
    by_supermode: Dict[str, Dict[int, Dict[str, Any]]],
    diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "consumer_centric_findings": [
            "group_time averages include travel time, payment, generalized cost, reliability, and booking success",
            "mode adoption shares are provided per group and time",
        ],
        "smart_city_integration_findings": [
            "station-level consumer signals expose effective price, surcharge, and power tightness",
            "peak-time summary aligns with congestion and service reliability indicators",
        ],
        "charging_and_energy_findings": [
            "shared-power tightness and surcharges are linked to consumer-facing cost fields",
            "storage and departure-capacity effects remain in backend equilibrium engine",
        ],
        "service_design_findings": [
            "platform recommendations include cheapest, fastest, best-balanced, and perceived-cost-optimal options",
            f"bottleneck summary: {diagnostics.get('bottleneck_summary')}",
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
    by_group = aggregate_consumer_metrics_by_group(itinerary_metrics, data.get("parameters", {}).get("q", {}), groups, times)
    by_supermode = aggregate_consumer_metrics_by_supermode(itinerary_metrics, groups, times)
    by_mode = aggregate_consumer_metrics_by_mode(itinerary_metrics, groups, times)
    station_signals = build_station_time_consumer_signals(diagnostics, times)
    peak_t = int(diagnostics.get("peak_t", times[0] if times else 0))
    service_summary = build_service_design_summary(by_group, station_signals, peak_t)
    platform = build_booking_platform_summary(
        itinerary_metrics,
        groups,
        times,
        data.get("parameters", {}).get("q", {}),
        float(config.get("best_balanced_weight_time", 0.4)),
        float(config.get("best_balanced_weight_fare", 0.3)),
        float(config.get("best_balanced_weight_reliability", 0.3)),
    )
    tce_summary = build_tce_summary(by_group, by_supermode, diagnostics)

    consumer_metrics: Dict[str, Any] = {
        "by_group_time": by_group,
        "by_supermode_time": by_supermode,
        "by_mode_time": by_mode,
    }
    if output_full_json and bool(config.get("report_consumer_itinerary_details", True)):
        consumer_metrics["itinerary_level_metrics"] = itinerary_metrics

    return {
        "consumer_metrics": consumer_metrics,
        "service_summary": service_summary,
        "platform_recommendations": platform,
        "tce_summary": tce_summary,
        "parameter_registry": build_parameter_registry(data, config),
    }


def apply_named_scenario(data: Dict[str, Any], scenario_name: str) -> Dict[str, Any]:
    """Return a copied data object with interpretable scenario modifications.

    Scenarios intentionally perform small, explicit parameter edits and preserve
    the base equilibrium formulation.
    """
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
            for s, tm in params.get(blk, {}).items():
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
