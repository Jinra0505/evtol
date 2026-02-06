import copy
import pytest

pytest.importorskip("gurobipy")

from project.src.assignment import aggregate_arc_flows, compute_itinerary_costs, logit_assignment
from project.src.congestion import compute_road_times, compute_station_waits
from project.src.data_loader import load_data
from project.src.mfd import compute_g
from project.src.runner import run_equilibrium


def test_convergence_uses_dx_dn():
    data = load_data("project/data/toy_data.yaml", "project/data_schema.yaml")
    data = copy.deepcopy(data)
    data["config"]["max_iter"] = 1

    times = data["sets"]["time"]
    arcs = data["sets"]["arcs"]
    stations = data["sets"]["stations"]
    itineraries = data["itineraries"]

    arc_flows = {a: {t: 0.0 for t in times} for a in arcs}
    utilization = {s: {t: 0.0 for t in times} for s in stations}
    g_series = compute_g([data["parameters"]["n0"]], data["parameters"]["mfd"])
    g_by_time = {t: g_series[min(len(g_series) - 1, idx)] for idx, t in enumerate(times)}

    tau = compute_road_times(arc_flows, data["parameters"]["arcs"], g_by_time, times)
    waits = compute_station_waits(utilization, data["parameters"]["stations"], times)
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
    x_new = aggregate_arc_flows(itineraries, flows, times)
    for arc in arcs:
        x_new.setdefault(arc, {t: 0.0 for t in times})
    dx_expected = max(abs(x_new[arc][t]) for arc in arcs for t in times)

    results, _ = run_equilibrium(data)
    assert abs(results["convergence"]["dx"] - dx_expected) <= 1.0e-6
