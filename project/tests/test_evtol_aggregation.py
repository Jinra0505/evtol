from project.src.assignment import (
    aggregate_evtol_dep_demand,
    aggregate_evtol_demand,
    compute_evtol_energy_demand,
    compute_itinerary_costs,
    logit_assignment,
)
from project.src.congestion import compute_road_times, compute_station_waits
from project.src.data_loader import load_data


def test_evtol_aggregation():
    data = load_data("project/data/toy_data.yaml", "project/data_schema.yaml")
    times = data["sets"]["time"]
    arcs = data["sets"]["arcs"]
    stations = data["sets"]["stations"]
    itineraries = data["itineraries"]

    arc_flows = {a: {t: 0.0 for t in times} for a in arcs}
    g_by_time = {t: 1.0 for t in times}
    tau = compute_road_times(arc_flows, data["parameters"]["arcs"], g_by_time, times)
    utilization = {s: {t: 0.0 for t in times} for s in stations}
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
    d_route = aggregate_evtol_demand(flows, itineraries, times)
    d_dep = aggregate_evtol_dep_demand(itineraries, flows, times)
    e_dep = compute_evtol_energy_demand(d_route, itineraries, times)

    assert "vt1" in d_route
    for t in times:
        assert d_route["vt1"][t] >= 0.0
    assert "s1" in d_dep
    assert "s1" in e_dep
