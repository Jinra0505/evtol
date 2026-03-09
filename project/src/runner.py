import copy
import math
import json
import os
import sys
from pathlib import Path

if __package__ is None:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    __package__ = "project.src"
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
from . import charging
from .charging import compute_station_loads_from_flows, solve_charging
from .congestion import compute_road_times, compute_station_waits
from .data_loader import load_data
from .itinerary_generator import generate_itineraries
from .mfd import boundary_flows, compute_g, update_accumulation






def _clean_number(x: float, eps: float = 1.0e-12) -> float:
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return x
        if abs(x) < eps:
            return 0.0
    return float(x)


def _clean_nested_numbers(obj: Any, eps: float = 1.0e-12) -> Any:
    if isinstance(obj, dict):
        return {k: _clean_nested_numbers(v, eps=eps) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nested_numbers(v, eps=eps) for v in obj]
    if isinstance(obj, float):
        return _clean_number(obj, eps=eps)
    return obj

def _safe_den(x: Any, eps: float = 1.0e-6) -> float:
    try:
        val = float(x)
    except (TypeError, ValueError):
        return eps
    return val if val > 0.0 else eps


def self_audit(results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    floor = float(config.get("vt_reliability_floor", 0.05))
    warnings: List[str] = []
    severe: List[str] = []
    cap_binding = False

    diagnostics = results.get("diagnostics", {})
    power_tightness = diagnostics.get("power_tightness", {})

    def _scan_numeric(path: str, obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                _scan_numeric(f"{path}.{k}" if path else str(k), v)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _scan_numeric(f"{path}[{i}]", v)
        elif isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                severe.append(f"nan/inf detected at diagnostics.{path}")

    _scan_numeric("", diagnostics)

    delta_t = float(results.get("meta", {}).get("delta_t", 1.0))
    eta_map = results.get("data_parameters", {}).get("vertiport_storage", {})

    for station, t_map in power_tightness.items():
        for t, entry in t_map.items():
            mu_entry = entry.get("mu_kw")
            mu_kw = float(mu_entry) if mu_entry is not None else 0.0
            solver_entry = entry.get("solver_used")
            surcharge = float(entry.get("raw_surcharge", 0.0))
            uncapped_entry = entry.get("raw_surcharge_uncapped")
            uncapped = float(uncapped_entry) if uncapped_entry is not None else 0.0
            cap = float(entry.get("cap", 0.0))
            if mu_entry is not None and not (mu_kw >= 0.0 and mu_kw < float("inf")):
                severe.append(f"invalid mu_kw at {station},{t}: {mu_kw}")
            if not (surcharge >= 0.0 and surcharge < float("inf")):
                severe.append(f"invalid surcharge at {station},{t}: {surcharge}")
            if solver_entry != "highs" and mu_entry is not None:
                severe.append("mu_kw is not LP dual")
            if abs(surcharge - cap) <= 1.0e-9:
                cap_binding = True
            if uncapped_entry is not None and cap > 0.0 and uncapped > 100.0 * cap:
                warnings.append(f"uncapped surcharge very large at {station},{t}; report VOLL and capped surcharge")

            vt_prob = float(entry.get("vt_service_prob", 1.0))
            if vt_prob < floor - 1.0e-8 or vt_prob > 1.0 + 1.0e-8:
                severe.append(f"vt_service_prob out of range at {station},{t}: {vt_prob}")

            p_net_served = float(entry.get("p_net_served_kw", 0.0))
            p_site_raw = float(entry.get("P_site_raw", 0.0))
            if p_net_served - p_site_raw > 1.0e-6:
                severe.append(f"served power exceeds site cap at {station},{t}")

            e_vt_req = float(entry.get("E_vt_req_kwh", 0.0))
            shed_vt_kwh = float(entry.get("shed_vt_kwh", 0.0))
            p_vt_charge = float(entry.get("P_vt_charge_kw", 0.0))
            b_before = entry.get("B_before_kwh")
            b_after = entry.get("B_after_kwh")
            if b_before is None or b_after is None:
                severe.append(f"B_before/B_after missing at {station},{t}")
            else:
                eta = float(eta_map.get(station, {}).get("eta_ch", 1.0))
                lhs = float(b_after)
                rhs = float(b_before) + eta * p_vt_charge * delta_t - (e_vt_req - shed_vt_kwh)
                if abs(lhs - rhs) > 1.0e-6:
                    severe.append(f"inventory balance broken at {station},{t}: err={abs(lhs-rhs)}")

            ratio_req = float(entry.get("ratio_req_exogenous", 0.0))
            if ratio_req > 1.2 and abs(mu_kw) <= 1.0e-9:
                warnings.append(f"ratio_req high but mu_kw ~ 0 at {station},{t}")

    n_series = diagnostics.get("n_series", {})
    n_values = n_series.values() if isinstance(n_series, dict) else n_series
    has_negative_n = any(float(v) < -1.0e-9 for v in n_values)
    if has_negative_n:
        severe.append("negative value detected in n_series")

    solver_used = diagnostics.get("shared_power_solver_used")
    requested_solver = str(config.get("shared_power_solver", "highs")).lower()
    if solver_used != "highs":
        warnings.append(f"shared_power solver is {solver_used}, not highs; dual interpretation may differ")
    if requested_solver == "highs" and solver_used != "highs":
        severe.append("Requested HiGHS but non-HiGHS solver used")
    if bool(diagnostics.get("missing_inventory_reporting", False)):
        severe.append("missing inventory reporting")
    if diagnostics.get("lp_failure") is not None:
        severe.append("LP failed")
    if any(entry.get("fallback_used") for t_map in power_tightness.values() for entry in t_map.values()):
        severe.append("LP failed and fallback path used")
    if cap_binding:
        warnings.append("price cap binding detected; report capped and uncapped surcharge in analysis")

    c2 = float(results.get("residuals", {}).get("C2", 0.0))
    c2_raw = float(results.get("residuals", {}).get("C2_raw", 0.0))
    if c2_raw <= 1.0e-9 and c2 > 1.0e-3:
        warnings.append("MSA not converged: C2 is smoothed mismatch while C2_raw is tiny")

    warnings = sorted(set(warnings))
    severe = sorted(set(severe))
    warnings = sorted(set(warnings + severe))
    audit = {
        "ok": len(severe) == 0,
        "warnings": warnings,
        "severe": severe,
        "cap_binding": cap_binding,
        "has_negative_n": has_negative_n,
    }
    if bool(config.get("audit_raise", False)) and severe:
        raise ValueError("SelfAudit severe issues: " + "; ".join(severe))
    return audit

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
                    total += time_map.get(t, 0.0)
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
                    original = time_map.get(t, 0.0)
                    reduced = original * ratio
                    reductions[key] = reductions.get(key, 0.0) + (original - reduced)
                    flows[it["id"]].setdefault(group, {})[t] = reduced

            for (od_key, group), delta in reductions.items():
                alternatives = [
                    it
                    for it in itineraries_by_od.get(od_key, [])
                    if it.get("mode") != "eVTOL"
                ]
                if not alternatives:
                    raise ValueError(f"No non-eVTOL alternative to enforce cap for {od_key}, {group}, t={t}")
                total_alt = sum(flows[it["id"]].get(group, {}).get(t, 0.0) for it in alternatives)
                if total_alt <= 0.0:
                    share = 1.0 / len(alternatives)
                    for it in alternatives:
                        flows[it["id"]].setdefault(group, {})[t] = flows[it["id"]].get(group, {}).get(t, 0.0) + delta * share
                else:
                    for it in alternatives:
                        current = flows[it["id"]].get(group, {}).get(t, 0.0)
                        flows[it["id"]].setdefault(group, {})[t] = current + delta * current / total_alt

    return flows


def compute_residuals(
    data: Dict[str, Any],
    itineraries: List[Dict[str, Any]],
    flows: Dict[str, Dict[str, Dict[int, float]]],
    arc_flows: Dict[str, Dict[int, float]],
    arc_flows_raw: Dict[str, Dict[int, float]] | None,
    tau: Dict[str, Dict[int, float]],
    n_series: List[float],
    g_series: List[float],
    inc_road: Dict[str, Dict[str, Dict[int, float]]],
    inc_station: Dict[str, Dict[str, Dict[int, float]]],
    utilization: Dict[str, Dict[int, float]],
    utilization_raw: Dict[str, Dict[int, float]] | None,
) -> Dict[str, float]:
    times = data["sets"]["time"]
    arcs = data["sets"]["arcs"]
    demand = data["parameters"]["q"]
    arc_params = data["parameters"]["arcs"]
    boundary_in = data["parameters"]["boundary_in"]
    boundary_out = data["parameters"]["boundary_out"]
    mfd_params = data["parameters"]["mfd"]
    delta_t = data["meta"]["delta_t"]

    residuals = {
        "C1": 0.0,
        "C2": 0.0,
        "C2_raw": 0.0,
        "C7": 0.0,
        "C8": 0.0,
        "C9": 0.0,
        "C11": 0.0,
        "C10": 0.0,
        "C10_raw": 0.0,
        "C2_rel": 0.0,
    }

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
                    expected += inc_road[arc][it["id"]][t] * time_map.get(t, 0.0)
            residuals["C2"] = max(residuals["C2"], abs(arc_flows[arc][t] - expected))
            if arc_flows_raw is not None:
                residuals["C2_raw"] = max(residuals["C2_raw"], abs(arc_flows_raw[arc][t] - expected))

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
                    expected_u += inc_station.get(station, {}).get(it["id"], {}).get(t, 0.0) * time_map.get(t, 0.0)
            residuals["C10"] = max(
                residuals["C10"],
                abs(utilization[station][t] - expected_u),
            )
            if utilization_raw is not None:
                residuals["C10_raw"] = max(
                    residuals["C10_raw"],
                    abs(utilization_raw[station][t] - expected_u),
                )

    max_flow = max(max(abs(arc_flows[a][t]), 1.0) for a in arcs for t in times)
    residuals["C2_rel"] = residuals["C2"] / _safe_den(max_flow, 1.0)
    return residuals


def run_equilibrium(data: Dict[str, Any], overrides: Dict[str, Any] | None = None, run_dispatch: bool = False) -> Tuple[Dict[str, Any], Dict[str, float]]:
    if overrides:
        data = copy.deepcopy(data)
        for key, value in overrides.items():
            data[key] = value

    times = data["sets"]["time"]
    arcs = data["sets"]["arcs"]
    stations = data["sets"]["stations"]
    delta_t = data["meta"]["delta_t"]
    config = data["config"]
    config["tol"] = float(config["tol"])
    config.setdefault("surcharge_method", "shadow_lp")
    config.setdefault("shadow_price_scale", 1.0)
    config.setdefault("shadow_price_cap_mult", 10.0)
    config.setdefault("vt_reliability_floor", 0.05)
    config.setdefault("vt_reliability_alpha", None)
    config.setdefault("vt_reliability_skip_below", 0.0)
    config.setdefault("vt_reliability_gamma", 0.0)
    config.setdefault("strict_audit", True)
    config.setdefault("audit_raise", False)
    config.setdefault("shared_power_solver", "highs")
    config.setdefault("output_full_json", True)
    config.setdefault("power_violation_mode", "net")
    config.setdefault("voll_ev_per_kwh", 50.0)
    config.setdefault("voll_vt_per_kwh", 200.0)
    itineraries = list(data["itineraries"])
    seen_ids = {it["id"] for it in itineraries}
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
    vt_service_prob = {s: {t: 1.0 for t in times} for s in stations}
    min_iters = int(config.get("min_iters", 10))
    patience = int(config.get("patience", 5))
    stop_reason = "max_iters"
    dprice = 0.0
    peak_t = 5 if 5 in times else times[-1]
    demand_keys = list(data["parameters"]["q"].keys())
    representative_od = config.get("representative_od")
    if not representative_od:
        representative_od = "A-B" if "A-B" in demand_keys else (demand_keys[0] if demand_keys else "")
    diagnostics = {
        "dx_history": [],
        "dn_history": [],
        "dprice_history": [],
        "max_surcharge_history": [],
        "price_snapshots": [],
        "mode_share": [],
        "surcharge_history": [],
        "mode_share_by_group": [],
        "power_tightness": {},
        "shared_power_solver_used": "unknown",
        "module_paths": {
            "runner_file": str(Path(__file__).resolve()),
            "charging_file": str(Path(charging.__file__).resolve()),
            "sys_path_head": list(sys.path[:5]),
        },
        "scipy_version": charging.SCIPY_VERSION,
        "scipy_ok": bool(charging.HAS_SCIPY),
        "lp_ok": None,
    }

    dx = 0.0
    dn = 0.0
    last_iteration = 0

    d_vt_route: Dict[str, Dict[int, float]] = {}
    d_vt_dep: Dict[str, Dict[int, float]] = {}
    e_dep: Dict[str, Dict[int, float]] = {}
    shadow_prices: Dict[str, Dict[int, float]] | None = None
    raw_surcharge: Dict[str, Dict[int, float]] = {s: {t: 0.0 for t in times} for s in stations}
    raw_surcharge_uncapped: Dict[str, Dict[int, float | None]] = {s: {t: 0.0 for t in times} for s in stations}
    shed_ev: Dict[str, Dict[int, float]] | None = None
    shed_vt: Dict[str, Dict[int, float]] | None = None
    P_vt_lp: Dict[str, Dict[int, float]] | None = None
    B_vt_lp: Dict[str, Dict[int, float]] | None = None
    shared_power_residuals: Dict[str, float] | None = None
    shared_power_lp_diag: Dict[str, Any] = {}
    load_agg: Dict[str, Dict[str, Dict[int, float]]] = {}
    lp_ok = False
    missing_inventory_reporting = False

    costs: Dict[str, Dict[int, Dict[str, float]]] = {}
    x_new = arc_flows
    u_new = utilization
    good_count = 0
    for iteration in range(config["max_iter"]):
        last_iteration = iteration + 1
        g_by_time = {t: g_series[min(len(g_series) - 1, idx)] for idx, t in enumerate(times)}
        tau = compute_road_times(arc_flows, arc_params, g_by_time, times)
        waits = compute_station_waits(utilization, station_params, times)
        effective_prices = {
            s: {t: electricity_price[s][t] + energy_surcharge[s][t] for t in times}
            for s in stations
        }
        generated = generate_itineraries(data, tau, waits, config)
        if generated:
            for it in generated:
                if it["id"] not in seen_ids:
                    itineraries.append(it)
                    seen_ids.add(it["id"])
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
            vt_service_prob=vt_service_prob,
            vt_service_prob_floor=1.0e-4,
            vt_reliability_gamma=float(config["vt_reliability_gamma"]),
            vt_service_prob_skip_below=float(config["vt_reliability_skip_below"]),
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

        dx_abs = max(abs(x_new[a][t] - arc_flows[a][t]) for a in arcs for t in times)
        max_flow = max(
            max(abs(x_new[a][t]), abs(arc_flows[a][t])) for a in arcs for t in times
        )
        dx = dx_abs / max(1.0, max_flow)

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

        load_agg = compute_station_loads_from_flows(data, itineraries, flows, times)
        e_vt_req = load_agg["E_vt_req"]
        p_ev_req = load_agg["P_ev_req_kw"]
        p_vt_req_energy = load_agg["P_vt_req_kw_energy"]
        p_vt_req_grid = load_agg["P_vt_req_kw_grid"]
        ev_energy_kwh = load_agg["E_ev_req"]
        surcharge_method = config.get("surcharge_method", "shadow_lp")
        shadow_scale = float(config.get("shadow_price_scale", 1.0))
        cap_mult = float(config.get("shadow_price_cap_mult", 10.0))
        lp_ok = False
        fallback_used = False
        solver_requested = str(config.get("shared_power_solver", "highs")).lower()
        if solver_requested != "highs":
            raise ValueError("config.shared_power_solver must be 'highs' (LP-only mode)")

        shadow_prices: Dict[str, Dict[int, float | None]] = {s: {t: None for t in times} for s in stations}
        shed_ev = {s: {t: 0.0 for t in times} for s in stations}
        shed_vt = {s: {t: 0.0 for t in times} for s in stations}
        shared_power_residuals = {"INV3": 0.0}
        B_vt_lp = {s: {t: 0.0 for t in times} for s in stations}
        P_vt_lp = {s: {t: 0.0 for t in times} for s in stations}
        heuristic_surcharge = {s: {t: 0.0 for t in times} for s in stations}

        if surcharge_method != "shadow_lp":
            raise ValueError("Only surcharge_method='shadow_lp' is supported")
        (
            B_vt_lp,
            P_vt_lp,
            shed_ev,
            shed_vt,
            shadow_prices,
            shared_power_residuals,
            shared_power_lp_diag,
        ) = charging.solve_shared_power_inventory_lp(data, e_dep, ev_energy_kwh)
        for s in stations:
            B_vt_lp.setdefault(s, {t: 0.0 for t in times})
            P_vt_lp.setdefault(s, {t: 0.0 for t in times})
        diagnostics["shared_power_solver_used"] = charging.LAST_SHARED_POWER_SOLVER_USED
        runtime_diag = data.get("diagnostics_runtime", {})
        diagnostics["lp_ok"] = bool(runtime_diag.get("lp_ok", diagnostics["shared_power_solver_used"] == "highs"))
        if runtime_diag.get("lp_failure") is not None:
            diagnostics["lp_failure"] = runtime_diag.get("lp_failure")
        lp_ok = diagnostics["shared_power_solver_used"] == "highs" and bool(diagnostics.get("lp_ok", False))
        fallback_used = diagnostics["shared_power_solver_used"] != "highs" or not lp_ok

        vt_service_prob_target = {s: {t: 1.0 for t in times} for s in stations}
        shed_ratio_map = {s: {t: 0.0 for t in times} for s in stations}
        cap_binding_map = {s: {t: False for t in times} for s in stations}

        for s in stations:
            for t in times:
                base_price = max(1.0e-6, float(electricity_price[s][t]))
                cap = cap_mult * base_price
                mu_kw = shadow_prices[s][t]
                if diagnostics.get("shared_power_solver_used") == "highs":
                    raw = shadow_scale * float(mu_kw or 0.0) / max(1e-6, delta_t)
                    raw_surcharge_uncapped[s][t] = raw
                    raw_surcharge[s][t] = min(raw, cap)
                else:
                    raw_surcharge_uncapped[s][t] = None
                    heuristic_surcharge[s][t] = max(0.0, (float(p_ev_req[s][t]) + float(p_vt_req_grid[s][t])) / max(1e-6, float(station_params[s]["P_site"][t])) - 1.0) * cap
                    raw_surcharge[s][t] = min(heuristic_surcharge[s][t], cap)
                cap_binding_map[s][t] = raw_surcharge[s][t] >= cap - 1.0e-9

                req_e_vt = float(e_vt_req.get(s, {}).get(t, 0.0))
                shed_vt_kwh = float((shed_vt or {}).get(s, {}).get(t, 0.0))
                if req_e_vt <= 1.0e-9:
                    target_prob = 1.0
                    shed_ratio = 0.0
                else:
                    shed_ratio = min(1.0, max(0.0, shed_vt_kwh / max(1.0e-9, req_e_vt)))
                    target_prob = max(float(config["vt_reliability_floor"]), 1.0 - shed_ratio)
                vt_service_prob_target[s][t] = target_prob
                shed_ratio_map[s][t] = shed_ratio

        prev_surcharge = {s: {t: energy_surcharge[s][t] for t in times} for s in stations}
        alpha_override = config.get("surcharge_msa_alpha")
        alpha = float(alpha_override) if alpha_override is not None else 1.0 / (iteration + 1.0)
        alpha_vt = config.get("vt_reliability_alpha")
        alpha_vt = float(alpha_vt) if alpha_vt is not None else alpha

        for s in stations:
            for t in times:
                energy_surcharge[s][t] = (1.0 - alpha) * energy_surcharge[s][t] + alpha * raw_surcharge[s][t]
                vt_next = (1.0 - alpha_vt) * vt_service_prob[s][t] + alpha_vt * vt_service_prob_target[s][t]
                vt_floor = float(config.get("vt_reliability_floor", 0.05))
                vt_service_prob[s][t] = min(1.0, max(vt_floor, vt_next))
                base_price = max(1.0e-6, float(electricity_price[s][t]))
                p_site_raw = float(station_params[s]["P_site"][t])
                p_site_den = _safe_den(p_site_raw)
                p_ev_req_kw = float(p_ev_req[s][t])
                p_vt_req_energy_kw = float(p_vt_req_energy[s][t])
                p_vt_req_grid_kw = float(p_vt_req_grid[s][t])
                p_ev_served_kw = max(0.0, p_ev_req_kw - (shed_ev or {}).get(s, {}).get(t, 0.0))
                p_vt_served_kw = (P_vt_lp or {}).get(s, {}).get(t, 0.0)
                t_idx = times.index(t)
                t_next = times[t_idx + 1] if t_idx + 1 < len(times) else None
                b_before = (B_vt_lp or {}).get(s, {}).get(t)
                terminal_key = times[-1] + 1
                if t_next is not None:
                    b_after = (B_vt_lp or {}).get(s, {}).get(t_next)
                elif (B_vt_lp or {}).get(s, {}).get(terminal_key) is not None:
                    b_after = (B_vt_lp or {}).get(s, {}).get(terminal_key)
                else:
                    eta = 1.0
                    if s in data["parameters"].get("vertiport_storage", {}):
                        eta = float(data["parameters"]["vertiport_storage"][s].get("eta_ch", 1.0))
                    if b_before is not None:
                        b_after = float(b_before) + eta * p_vt_served_kw * delta_t - (
                            float(e_vt_req[s][t]) - float((shed_vt or {}).get(s, {}).get(t, 0.0))
                        )
                    else:
                        b_after = None
                if b_before is None:
                    missing_inventory_reporting = True
                    b_before = 0.0
                if b_after is None:
                    missing_inventory_reporting = True
                    b_after = 0.0
                b_before = float(b_before)
                b_after = float(b_after)
                p_net_served_kw = p_ev_served_kw + p_vt_served_kw
                current_shed_ratio = 0.0 if float(e_vt_req[s][t]) <= 1.0e-12 else float((shed_vt or {}).get(s, {}).get(t, 0.0)) / max(1.0e-12, float(e_vt_req[s][t]))
                current_shed_ratio = min(1.0, max(0.0, current_shed_ratio))
                vt_prob_from_shed = min(1.0, max(0.0, 1.0 - current_shed_ratio))
                solver_used = diagnostics.get("shared_power_solver_used")
                mu_entry = _clean_number(shadow_prices[s][t]) if (shadow_prices[s][t] is not None and solver_used == "highs") else None
                uncapped_entry = _clean_number(raw_surcharge_uncapped[s][t]) if (raw_surcharge_uncapped[s][t] is not None and solver_used == "highs") else None
                ratio_req_exogenous = (p_ev_req_kw + p_vt_req_grid_kw) / p_site_den
                ratio_req_energy = (p_ev_req_kw + p_vt_req_energy_kw) / p_site_den
                diagnostics["power_tightness"].setdefault(s, {})[t] = {
                    "P_site_raw": p_site_raw,
                    "P_site_den": p_site_den,
                    "ratio_req_exogenous": _clean_number(ratio_req_exogenous),
                    "ratio_req_energy": _clean_number(ratio_req_energy),
                    "ratio_net_actual": _clean_number(p_net_served_kw / p_site_den),
                    "P_ev_req_kw": _clean_number(p_ev_req_kw),
                    "P_vt_req_kw_energy": _clean_number(p_vt_req_energy_kw),
                    "P_vt_req_kw_grid": _clean_number(p_vt_req_grid_kw),
                    "mu_kw": mu_entry,
                    "raw_surcharge_uncapped": uncapped_entry,
                    "raw_surcharge_capped": _clean_number(raw_surcharge[s][t]),
                    "raw_surcharge": _clean_number(raw_surcharge[s][t]),
                    "heuristic_surcharge": _clean_number(heuristic_surcharge[s][t]) if solver_used != "highs" else None,
                    "surcharge_smoothed": _clean_number(energy_surcharge[s][t]),
                    "cap": cap_mult * base_price,
                    "cap_binding": cap_binding_map[s][t],
                    "E_vt_req_kwh": e_vt_req[s][t],
                    "P_vt_charge_kw": p_vt_served_kw,
                    "B_before_kwh": b_before,
                    "B_after_kwh": b_after,
                    "shed_ev_kw": _clean_number((shed_ev or {}).get(s, {}).get(t, 0.0)),
                    "shed_vt_kwh": _clean_number((shed_vt or {}).get(s, {}).get(t, 0.0)),
                    "p_ev_served_kw": p_ev_served_kw,
                    "p_vt_served_kw": p_vt_served_kw,
                    "p_net_served_kw": p_net_served_kw,
                    "vt_service_prob_target": vt_service_prob_target[s][t],
                    "vt_service_prob": vt_service_prob[s][t],
                    "shed_ratio": current_shed_ratio,
                    "solver_used": solver_used,
                    "fallback_used": fallback_used,
                }

        dprice = max(
            abs(energy_surcharge[s][t] - prev_surcharge[s][t]) for s in stations for t in times
        )
        max_surcharge = max(energy_surcharge[s][t] for s in stations for t in times)
        diagnostics["dx_history"].append(dx)
        diagnostics["dn_history"].append(dn)
        diagnostics["dprice_history"].append(dprice)
        diagnostics["max_surcharge_history"].append(max_surcharge)

        snapshot = {
            "iteration": iteration + 1,
            "t": peak_t,
            "stations": {},
        }
        for s in stations[:3]:
            base_price = electricity_price[s][peak_t]
            surcharge = energy_surcharge[s][peak_t]
            snapshot["stations"][s] = {
                "base_price": base_price,
                "surcharge": surcharge,
                "effective_price": base_price + surcharge,
            }
        diagnostics["price_snapshots"].append(snapshot)

        ev_total = 0.0
        vt_total = 0.0
        if representative_od:
            for it in itineraries:
                if f"{it['od'][0]}-{it['od'][1]}" != representative_od:
                    continue
                for group, time_map in flows[it["id"]].items():
                    val = time_map.get(peak_t, 0.0)
                    if it.get("mode") == "eVTOL":
                        vt_total += val
                    else:
                        ev_total += val
        total_flow = ev_total + vt_total
        ev_share = ev_total / total_flow if total_flow > 0.0 else 0.0
        vt_share = vt_total / total_flow if total_flow > 0.0 else 0.0
        diagnostics["mode_share"].append(
            {
                "iteration": iteration + 1,
                "od": representative_od,
                "t": peak_t,
                "ev_share": ev_share,
                "vt_share": vt_share,
            }
        )
        mode_share_by_group = {}
        if representative_od:
            rep_its = [it for it in itineraries if f"{it['od'][0]}-{it['od'][1]}" == representative_od]
            groups_in_flows = sorted(
                {
                    grp
                    for it in rep_its
                    for grp in flows.get(it["id"], {}).keys()
                }
            )
            if not groups_in_flows:
                flow_keys = {it_id: list(group_map.keys()) for it_id, group_map in flows.items()}
                raise ValueError(
                    f"No group keys found in flows for OD {representative_od}. "
                    f"Representative itineraries={[it['id'] for it in rep_its]}, flow keys={flow_keys}"
                )
            for grp in groups_in_flows:
                ev_g = 0.0
                vt_g = 0.0
                for it in rep_its:
                    val = flows[it["id"]].get(grp, {}).get(peak_t, 0.0)
                    if it.get("mode") == "eVTOL":
                        vt_g += val
                    else:
                        ev_g += val
                den = max(1e-12, ev_g + vt_g)
                mode_share_by_group[grp] = {"ev": ev_g / den, "vt": vt_g / den}
        diagnostics["mode_share_by_group"].append(
            {"iteration": iteration + 1, "od": representative_od, "t": peak_t, "shares": mode_share_by_group}
        )
        diagnostics["surcharge_history"].append(
            {
                "iteration": iteration + 1,
                "surcharge": {s: {t: energy_surcharge[s][t] for t in times} for s in stations},
            }
        )
        if len(diagnostics["surcharge_history"]) > 20:
            diagnostics["surcharge_history"] = diagnostics["surcharge_history"][-20:]

        should_print = (iteration == 0) or ((iteration + 1) % 5 == 0)
        if should_print:
            print(
                "iter="
                f"{iteration + 1} dx={dx:.6f} dn={dn:.6f} dprice={dprice:.6f} "
                f"max_surcharge={max_surcharge:.6f} alpha={alpha:.3f}"
            )
            if "s1" in snapshot["stations"]:
                vals = snapshot["stations"]["s1"]
                print(
                    f"peak_t={peak_t} s1_eff_price={vals['effective_price']:.3f} "
                    f"(base={vals['base_price']:.3f},sur={vals['surcharge']:.3f})"
                )
            if representative_od:
                print(
                    f"peak_t={peak_t} od={representative_od} ev_share={ev_share:.3f} vt_share={vt_share:.3f}"
                )

        if iteration + 1 >= min_iters:
            if max(dx, dn, dprice) < config["tol"]:
                good_count += 1
            else:
                good_count = 0
            if good_count >= patience:
                if not should_print:
                    print(
                        "iter="
                        f"{iteration + 1} dx={dx:.6f} dn={dn:.6f} dprice={dprice:.6f} "
                        f"max_surcharge={max_surcharge:.6f} alpha={alpha:.3f}"
                    )
                    if "s1" in snapshot["stations"]:
                        vals = snapshot["stations"]["s1"]
                        print(
                            f"peak_t={peak_t} s1_eff_price={vals['effective_price']:.3f} "
                            f"(base={vals['base_price']:.3f},sur={vals['surcharge']:.3f})"
                        )
                    if representative_od:
                        print(
                            f"peak_t={peak_t} od={representative_od} ev_share={ev_share:.3f} vt_share={vt_share:.3f}"
                        )
                stop_reason = "patience"
                break

    residuals = compute_residuals(
        data,
        itineraries,
        flows,
        arc_flows,
        x_new,
        tau,
        n_series,
        g_series,
        inc_road,
        inc_station,
        utilization,
        u_new,
    )

    if run_dispatch:
        E, p_ch, y, charging_residuals, B_vt, P_vt, inv_residuals, _ = solve_charging(data, e_dep, d_vt_dep)
    else:
        E = {m: {t: 0.0 for t in times} for m in data["sets"]["vehicles"]}
        p_ch = {m: {s: {t: 0.0 for t in times} for s in stations} for m in data["sets"]["vehicles"]}
        y = {m: {s: {t: 0 for t in times} for s in stations} for m in data["sets"]["vehicles"]}
        charging_residuals = {"SKIPPED": 0.0}
        B_vt = B_vt_lp or {}
        if lp_ok and P_vt_lp:
            P_vt = P_vt_lp
        else:
            P_vt = {dep: {t: load_agg.get("P_vt_req", {}).get(dep, {}).get(t, 0.0) for t in times} for dep in load_agg.get("P_vt_req", {})}
        inv_residuals = shared_power_residuals or {}

    cap_violation = 0.0
    cap_pax = data["parameters"].get("vertiport_cap_pax")
    if cap_pax:
        for dep, time_map in d_vt_dep.items():
            for t in times:
                cap_violation = max(cap_violation, max(0.0, time_map[t] - cap_pax[dep][t]))

    power_violation = 0.0
    power_mode = str(config.get("power_violation_mode", "net"))
    for s in stations:
        for t in times:
            if run_dispatch:
                total_power = sum(p_ch[m][s][t] for m in data["sets"]["vehicles"])
                if P_vt and s in P_vt:
                    total_power += P_vt[s][t]
            else:
                if power_mode == "request":
                    total_power = load_agg.get("P_total_req", {}).get(s, {}).get(t, 0.0)
                else:
                    p_ev_served = max(0.0, p_ev_req[s][t] - (shed_ev or {}).get(s, {}).get(t, 0.0))
                    p_vt_served = (P_vt_lp or {}).get(s, {}).get(t, p_vt_req_grid[s][t])
                    total_power = p_ev_served + p_vt_served
            power_violation = max(power_violation, max(0.0, total_power - station_params[s]["P_site"][t]))

    inflow_total, outflow_total = boundary_flows(
        arc_flows, data["parameters"]["boundary_in"], data["parameters"]["boundary_out"], times
    )
    g_used_by_time = {t: g_series[min(idx, len(g_series) - 1)] for idx, t in enumerate(times)}
    times_ext = list(times) + [times[-1] + 1] if times else []
    B_vt_report: Dict[str, Dict[int, float]] = {}
    for s in stations:
        B_vt_report[s] = {}
        for idx, t_ext in enumerate(times_ext):
            if B_vt_lp is not None and s in B_vt_lp and t_ext in B_vt_lp[s]:
                B_vt_report[s][t_ext] = float(B_vt_lp[s][t_ext])
            elif idx == len(times_ext) - 1 and len(times_ext) >= 2:
                prev_t = times_ext[-2]
                B_vt_report[s][t_ext] = float((B_vt_lp or {}).get(s, {}).get(prev_t, 0.0))
            else:
                B_vt_report[s][t_ext] = float((B_vt_lp or {}).get(s, {}).get(t_ext, 0.0))

    cbd_tau = {}
    for arc, params in arc_params.items():
        if params.get("type") != "CBD":
            continue
        theta = float(params.get("theta", 1.0))
        cbd_tau[arc] = {}
        for t in times:
            g_t = g_used_by_time[t]
            cbd_tau[arc][t] = float(params["tau0"]) * g_t * theta

    results = {
        "x": arc_flows,
        "tau": tau,
        "n": n_series,
        "g": g_series,
        "u": utilization,
        "w": waits,
        "f": flows,
        "costs": costs,
        "d_vt_route": d_vt_route,
        "d_vt_dep": d_vt_dep,
        "e_dep": e_dep,
        "inventory": {"B_vt": B_vt, "P_vt": P_vt, "residuals": inv_residuals},
        "shadow_price_power": shadow_prices,
        "surcharge_power": energy_surcharge,
        "surcharge_power_raw": raw_surcharge,
        "surcharge_power_uncapped": raw_surcharge_uncapped,
        "shared_power": {
            "B_vt": B_vt_lp,
            "P_vt": P_vt_lp,
            "shed_ev": shed_ev,
            "shed_vt": shed_vt,
            "residuals": shared_power_residuals,
        },
        "validation": {
            "cap_violation": cap_violation,
            "power_violation": power_violation,
        },
        "residuals": residuals,
        "convergence": {"dx": dx, "dn": dn, "dprice": dprice, "iterations": last_iteration},
        "diagnostics": {
            **diagnostics,
            "stop_reason": stop_reason,
            "boundary_inflow": {t: inflow_total[idx] for idx, t in enumerate(times)},
            "boundary_outflow": {t: outflow_total[idx] for idx, t in enumerate(times)},
            "n_series": {idx: n_series[idx] for idx in range(len(n_series))},
            "g_series": {idx: g_series[idx] for idx in range(len(g_series))},
            "g_state_series": [float(v) for v in g_series],
            "g_used_by_time": g_used_by_time,
            "cbd_tau": cbd_tau,
            "cbd_tau_used": cbd_tau,
            "station_loads": load_agg,
            "vt_service_prob": vt_service_prob,
            "vt_inventory_B": B_vt_report,
            "vt_charge_power_P": P_vt_lp,
            "shared_power_lp": shared_power_lp_diag,
            "missing_inventory_reporting": missing_inventory_reporting,
        },
        "data_parameters": data.get("parameters", {}),
        "meta": data.get("meta", {}),
        "charging": {
            "E": E,
            "p_ch": p_ch,
            "y": y,
            "residuals": charging_residuals,
        },
        "solver_used": charging.LAST_SOLVER_USED if run_dispatch else "aggregate",
    }
    audit = self_audit(results, config)
    results["SelfAudit"] = audit
    if bool(config.get("strict_audit", True)) and audit.get("warnings"):
        diagnostics["audit_warnings"] = audit["warnings"]
    return results, residuals


def save_outputs(results: Dict[str, Any], path: str) -> None:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def _resolve_path(path: str, fallback: str) -> str:
    import os

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


def main() -> None:
    import argparse
    import datetime
    import platform
    import subprocess

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="simple_case.yaml", help="Path to data YAML")
    parser.add_argument("--schema", default="data_schema.yaml", help="Path to schema YAML")
    parser.add_argument("--dispatch", action="store_true", help="Run vehicle-level dispatch module")
    parser.add_argument("--report-out", "--report_out", dest="report_out", default="report.json", help="Path to report JSON output")
    parser.add_argument("--full_report", action="store_true", help="Write full report payload")
    args = parser.parse_args()

    try:
        data_path = _resolve_path(args.data, "project/data/simple_case.yaml")
        schema_path = _resolve_path(args.schema, "project/data_schema.yaml")
        data = load_data(data_path, schema_path)

        results, residuals = run_equilibrium(data, run_dispatch=args.dispatch)
        save_outputs(results, "project/output.json")
        dprice_hist = results["diagnostics"].get("dprice_history", [])
        max_surcharge = max(v for m in results["surcharge_power"].values() for v in m.values())
        cbd_tau = results["diagnostics"].get("cbd_tau", {})
        cbd_tau_one = {}
        if cbd_tau:
            first_arc = next(iter(cbd_tau.keys()))
            cbd_tau_one = {first_arc: cbd_tau[first_arc]}
        peak_snapshot = results["diagnostics"].get("price_snapshots", [])
        peak_prices_last = peak_snapshot[-1] if peak_snapshot else {}
        mode_share_last = results["diagnostics"].get("mode_share_by_group", [])
        mode_share_last = mode_share_last[-1] if mode_share_last else {}
        output_full_json = bool(data.get("config", {}).get("output_full_json", True) or args.full_report)

        git_commit = None
        try:
            git_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        except Exception:
            git_commit = None

        equilibrium = {
            "C1": residuals.get("C1", 0.0),
            "C2": residuals.get("C2", 0.0),
            "C2_rel": residuals.get("C2_rel", 0.0),
            "C2_raw": residuals.get("C2_raw", 0.0),
            "C7": residuals.get("C7", 0.0),
            "C8": residuals.get("C8", 0.0),
            "C9": residuals.get("C9", 0.0),
            "C10": residuals.get("C10", 0.0),
            "C10_raw": residuals.get("C10_raw", 0.0),
            "C11": residuals.get("C11", 0.0),
        }

        diagnostics_full = {**results.get("diagnostics", {})}
        diagnostics_full.update({
            "iter_count": results["convergence"].get("iterations"),
            "dx_end": results["convergence"].get("dx"),
            "dn_end": results["convergence"].get("dn"),
            "dprice_start": dprice_hist[0] if dprice_hist else None,
            "dprice_end": dprice_hist[-1] if dprice_hist else None,
            "max_surcharge": max_surcharge,
            "peak_t": peak_prices_last.get("t"),
            "peak_prices": peak_prices_last.get("stations", {}),
            "objective_by_station_time": results.get("diagnostics", {}).get("shared_power_lp", {}).get("objective_components", {}),
            "objective_totals": results.get("diagnostics", {}).get("shared_power_lp", {}).get("objective_totals", {}),
            "dual_trace": results.get("diagnostics", {}).get("shared_power_lp", {}).get("dual_trace", {}),
        })

        report = {
            "meta": {
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "git_commit_if_available": git_commit,
                "python_version": sys.version,
                "platform": platform.platform(),
                "module_paths": diagnostics_full.get("module_paths", {}),
            },
            "config_used": data.get("config", {}),
            "equilibrium": equilibrium,
            "surcharge_power": results["surcharge_power"],
            "surcharge_power_uncapped": results.get("surcharge_power_uncapped", {}),
            "diagnostics": diagnostics_full if output_full_json else {
                "stop_reason": diagnostics_full.get("stop_reason"),
                "iter_count": diagnostics_full.get("iter_count"),
                "dx_end": diagnostics_full.get("dx_end"),
                "dn_end": diagnostics_full.get("dn_end"),
                "dprice_end": diagnostics_full.get("dprice_end"),
                "max_surcharge": diagnostics_full.get("max_surcharge"),
                "peak_t": diagnostics_full.get("peak_t"),
                "peak_prices": diagnostics_full.get("peak_prices"),
            },
            "SelfAudit": results.get("SelfAudit", {}),
        }
        report["diagnostics"].setdefault("cbd_tau", cbd_tau_one)
        report["diagnostics"]["shared_power_solver_used"] = results["diagnostics"].get("shared_power_solver_used")
        report["diagnostics"]["delta_t"] = data.get("meta", {}).get("delta_t")
        report["diagnostics"]["surcharge_kappa"] = data.get("config", {}).get("shadow_price_scale")
        report["diagnostics"]["surcharge_cap_mult"] = data.get("config", {}).get("shadow_price_cap_mult")
        report["diagnostics"]["VOLL_EV"] = data.get("config", {}).get("voll_ev_per_kwh")
        report["diagnostics"]["VOLL_VT"] = data.get("config", {}).get("voll_vt_per_kwh")
        report["diagnostics"]["vt_reliability_gamma"] = data.get("config", {}).get("vt_reliability_gamma")
        report = _clean_nested_numbers(report)

    except Exception as exc:
        report = {
            "meta": {
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "git_commit_if_available": None,
                "python_version": sys.version,
                "platform": platform.platform(),
            },
            "config_used": {},
            "equilibrium": {},
            "surcharge_power": {},
            "surcharge_power_uncapped": {},
            "diagnostics": {
                "module_paths": {
                    "runner_file": str(Path(__file__).resolve()),
                    "charging_file": str(Path(charging.__file__).resolve()),
                    "sys_path_head": list(sys.path[:5]),
                },
                "scipy_version": charging.SCIPY_VERSION,
                "scipy_ok": bool(charging.HAS_SCIPY),
            },
            "SelfAudit": {"ok": False, "warnings": [], "severe": [repr(exc)], "cap_binding": False, "has_negative_n": False},
        }
        payload = json.dumps(_clean_nested_numbers(report), ensure_ascii=False, indent=2)
        with open(args.report_out, "w", encoding="utf-8") as handle:
            handle.write(payload)
        print(payload)
        raise

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    with open(args.report_out, "w", encoding="utf-8") as handle:
        handle.write(payload)
    print(
        f"SUMMARY iter_count={report['diagnostics'].get('iter_count')} "
        f"max_surcharge={report['diagnostics'].get('max_surcharge')} "
        f"peak_t={report['diagnostics'].get('peak_t')} solver={report['diagnostics'].get('shared_power_solver_used')}"
    )
    print(f"Wrote {args.report_out} (bytes={len(payload.encode('utf-8'))})")
    print(payload)


if __name__ == "__main__":
    main()
