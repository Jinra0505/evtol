from project.src.congestion import compute_road_times
from project.src.mfd import boundary_flows, compute_g, update_accumulation


def test_mfd_scheme2_residuals():
    times = [1, 2]
    arc_params = {
        "a_cbd1": {"tau0": 2.0, "cap": 100.0, "alpha": 0.15, "beta": 4.0, "type": "CBD", "theta": 1.2},
        "a_in": {"tau0": 1.0, "cap": 100.0, "alpha": 0.15, "beta": 4.0, "type": "CBD", "theta": 1.0},
        "a_out": {"tau0": 1.0, "cap": 100.0, "alpha": 0.15, "beta": 4.0, "type": "CBD", "theta": 1.0},
    }
    arc_flows = {
        "a_cbd1": {1: 10.0, 2: 12.0},
        "a_in": {1: 5.0, 2: 6.0},
        "a_out": {1: 4.0, 2: 5.0},
    }
    inflow, outflow = boundary_flows(arc_flows, ["a_in"], ["a_out"], times)
    n = update_accumulation(5.0, inflow, outflow, 1.0)
    g_series = compute_g(n, {"gamma": 0.1, "n_crit": 50.0, "g_max": 2.0})
    g_by_time = {t: g_series[idx] for idx, t in enumerate(times)}
    tau = compute_road_times(arc_flows, arc_params, g_by_time, times)

    for idx, t in enumerate(times):
        g_expected = g_series[idx]
        assert abs(tau["a_cbd1"][t] - 2.0 * g_expected * 1.2) <= 1.0e-6
