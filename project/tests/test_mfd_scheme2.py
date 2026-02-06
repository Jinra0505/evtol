from project.src.data_loader import load_data
from project.src.mfd import boundary_flows
from project.src.runner import run_equilibrium


def test_mfd_scheme2_residuals():
    data = load_data("project/data/toy_data.yaml", "project/data_schema.yaml")
    results, _ = run_equilibrium(data)
    times = data["sets"]["time"]

    n_series = results["n"]
    g_series = results["g"]

    inflow, outflow = boundary_flows(
        results["x"],
        data["parameters"]["boundary_in"],
        data["parameters"]["boundary_out"],
        times,
    )
    assert any(val > 0.0 for val in inflow + outflow)

    assert any(abs(n_series[idx + 1] - n_series[idx]) > 0.0 for idx in range(len(times)))
    assert any(abs(g_series[idx + 1] - g_series[idx]) > 0.0 for idx in range(len(times)))

    for t_idx, t in enumerate(times):
        tau_expected = (
            data["parameters"]["arcs"]["a_cbd1"]["tau0"]
            * g_series[t_idx]
            * data["parameters"]["arcs"]["a_cbd1"]["theta"]
        )
        assert abs(results["tau"]["a_cbd1"][t] - tau_expected) <= 1.0e-6
