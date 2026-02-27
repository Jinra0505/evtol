from project.src.assignment import compute_itinerary_costs, logit_assignment
from project.src.congestion import compute_road_times, compute_station_waits
from project.src.data_loader import load_data


def test_logit_assignment_stability():
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

    flows, generalized_costs = logit_assignment(
        itineraries,
        costs,
        data["parameters"]["q"],
        data["parameters"]["VOT"],
        data["parameters"]["lambda"],
        times,
    )

    for t in times:
        total = 0.0
        for it in itineraries:
            total += flows[it["id"]]["k1"][t]
        assert abs(total - data["parameters"]["q"]["A-B"]["k1"][t]) <= 1.0e-6

    for it in itineraries:
        assert "k1" in generalized_costs[it["id"]]
        for t in times:
            assert t in generalized_costs[it["id"]]["k1"]
