import copy
import json
import os
import sys

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
from .charging import solve_charging, solve_shared_power_lp
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

    arc_check = arc_flows_raw or arc_flows
    for arc in arcs:
        for t in times:
            expected = 0.0
            for it in itineraries:
                for group, time_map in flows[it["id"]].items():
                    expected += inc_road[arc][it["id"]][t] * time_map.get(t, 0.0)
            residuals["C2"] = max(residuals["C2"], abs(arc_check[arc][t] - expected))

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

    utilization_check = utilization_raw or utilization
    for station in utilization_check:
        for t in times:
            expected_u = 0.0
            for it in itineraries:
                for group, time_map in flows[it["id"]].items():
                    expected_u += inc_station.get(station, {}).get(it["id"], {}).get(t, 0.0) * time_map.get(t, 0.0)
            residuals["C10"] = max(
                residuals["C10"],
                abs(utilization_check[station][t] - expected_u),
            )

    return residuals


def run_equilibrium(data: Dict[str, Any], overrides: Dict[str, Any] | None = None) -> Tuple[Dict[str, Any], Dict[str, float]]:
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
        "dprice_history": [],
        "max_surcharge_history": [],
        "price_snapshots": [],
        "mode_share": [],
        "surcharge_history": [],
    }

    dx = 0.0
    dn = 0.0
    last_iteration = 0

    d_vt_route: Dict[str, Dict[int, float]] = {}
    d_vt_dep: Dict[str, Dict[int, float]] = {}
    e_dep: Dict[str, Dict[int, float]] = {}
    shadow_prices: Dict[str, Dict[int, float]] | None = None
    raw_surcharge: Dict[str, Dict[int, float]] = {s: {t: 0.0 for t in times} for s in stations}
    shed_ev: Dict[str, Dict[int, float]] | None = None
    shed_vt: Dict[str, Dict[int, float]] | None = None
    P_vt_lp: Dict[str, Dict[int, float]] | None = None
    B_vt_lp: Dict[str, Dict[int, float]] | None = None
    shared_power_residuals: Dict[str, float] | None = None

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

        ev_energy = aggregate_ev_energy_demand(itineraries, flows, times)
        (
            B_vt_lp,
            P_vt_lp,
            shed_ev,
            shed_vt,
            shadow_prices,
            shared_power_residuals,
        ) = solve_shared_power_lp(data, itineraries, flows, times)

        prev_surcharge = {s: {t: energy_surcharge[s][t] for t in times} for s in stations}
        alpha_override = config.get("surcharge_msa_alpha")
        alpha = float(alpha_override) if alpha_override is not None else 1.0 / (iteration + 1.0)
        if shadow_prices:
            for s in stations:
                for t in times:
                    base_price = electricity_price[s][t]
                    cap = base_price * 10.0
                    shadow_scale = float(config.get("surcharge_kappa", 1.0))
                    new_surcharge = max(0.0, shadow_scale * shadow_prices.get(s, {}).get(t, 0.0))
                    raw_surcharge[s][t] = new_surcharge
                    if cap > 0.0:
                        new_surcharge = min(new_surcharge, cap)
                    energy_surcharge[s][t] = (1.0 - alpha) * energy_surcharge[s][t] + alpha * new_surcharge

        dprice = max(
            abs(energy_surcharge[s][t] - prev_surcharge[s][t]) for s in stations for t in times
        )
        max_surcharge = max(energy_surcharge[s][t] for s in stations for t in times)
        diagnostics["dx_history"].append(dx)
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
        diagnostics["surcharge_history"].append(
            {
                "iteration": iteration + 1,
                "surcharge": {s: {t: energy_surcharge[s][t] for t in times} for s in stations},
            }
        )
        if len(diagnostics["surcharge_history"]) > 20:
            diagnostics["surcharge_history"] = diagnostics["surcharge_history"][-20:]

        print(
            "iter="
            f"{iteration + 1} dx={dx:.6f} dn={dn:.6f} dprice={dprice:.6f} "
            f"max_surcharge={max_surcharge:.6f} alpha={alpha:.3f}"
        )
        if snapshot["stations"]:
            station_repr = ", ".join(
                f"{s}:base={vals['base_price']:.3f},sur={vals['surcharge']:.3f},eff={vals['effective_price']:.3f}"
                for s, vals in snapshot["stations"].items()
            )
            print(f"peak_t={peak_t} prices [{station_repr}]")
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

    E, p_ch, y, charging_residuals, B_vt, P_vt, inv_residuals, _ = solve_charging(data, e_dep, d_vt_dep)

    cap_violation = 0.0
    cap_pax = data["parameters"].get("vertiport_cap_pax")
    if cap_pax:
        for dep, time_map in d_vt_dep.items():
            for t in times:
                cap_violation = max(cap_violation, max(0.0, time_map[t] - cap_pax[dep][t]))

    power_violation = 0.0
    for s in stations:
        for t in times:
            total_power = sum(p_ch[m][s][t] for m in data["sets"]["vehicles"])
            if P_vt and s in P_vt:
                total_power += P_vt[s][t]
            power_violation = max(power_violation, max(0.0, total_power - station_params[s]["P_site"][t]))

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
        "diagnostics": {**diagnostics, "stop_reason": stop_reason},
        "charging": {
            "E": E,
            "p_ch": p_ch,
            "y": y,
            "residuals": charging_residuals,
        },
        "solver_used": charging.LAST_SOLVER_USED,
    }
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true", help="Run planning layer search")
    parser.add_argument("--data", default="toy_data.yaml", help="Path to data YAML")
    parser.add_argument("--schema", default="data_schema.yaml", help="Path to schema YAML")
    args = parser.parse_args()

    data_path = _resolve_path(args.data, "project/data/toy_data.yaml")
    schema_path = _resolve_path(args.schema, "project/data_schema.yaml")
    data = load_data(data_path, schema_path)
    if args.plan:
        from .planner import solve_planning

        best_plan, best_cost, best_results, best_diag = solve_planning(data)
        save_outputs(best_results, "project/output.json")
        print(json.dumps({"best_plan": best_plan, "best_cost": best_cost, "best_breakdown": best_diag}, indent=2))
        return

    results, residuals = run_equilibrium(data)
    save_outputs(results, "project/output.json")
    report = {
        "Traffic": {"C1": residuals["C1"], "C2": residuals["C2"]},
        "CBD": {"C7": residuals["C7"], "C8": residuals["C8"], "C9": residuals["C9"]},
        "Choice": {"C11": residuals["C11"]},
        "Stations": {"C10": residuals["C10"]},
        "Charging": results["charging"]["residuals"],
        "Inventory": results["inventory"]["residuals"],
        "Validation": results["validation"],
        "Surcharge": results["surcharge_power"],
        "SurchargeRaw": results["surcharge_power_raw"],
        "ShadowPrice": results["shadow_price_power"],
        "Solver": results["solver_used"],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
