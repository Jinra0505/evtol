from project.src.charging import solve_charging
from project.src.data_loader import load_data


def test_charging_constraints_residuals():
    data = load_data("project/data/toy_data.yaml", "project/data_schema.yaml")
    E, p_ch, y, residuals, _, _, _ = solve_charging(data)
    for key, val in residuals.items():
        assert val <= 1.0e-6, f"{key} residual too large: {val}"

    assert p_ch["m1"]["s1"][1] >= 4.444
    assert y["m1"]["s1"][1] == 1
    assert E["m1"][1] >= data["parameters"]["charging"]["m1"]["E_res"]
