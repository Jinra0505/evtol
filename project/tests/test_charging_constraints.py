from project.src.charging import solve_charging
from project.src.data_loader import load_data


def test_charging_constraints_residuals():
    data = load_data("project/data/toy_data.yaml", "project/data_schema.yaml")
    E, p_ch, _, residuals = solve_charging(data)
    for key, val in residuals.items():
        assert val <= 1.0e-6, f"{key} residual too large: {val}"

    total_charge = sum(p_ch["m1"]["s1"][t] for t in data["sets"]["time"])
    assert total_charge > 0.0
    assert E["m1"][1] >= data["parameters"]["charging"]["m1"]["E_res"]
