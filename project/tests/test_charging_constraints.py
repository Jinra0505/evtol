from project.src.charging import solve_charging
from project.src.data_loader import load_data


def test_charging_constraints_residuals():
    data = load_data("project/data/toy_data.yaml", "project/data_schema.yaml")
    _, _, _, residuals = solve_charging(data)
    for key, val in residuals.items():
        assert val <= 1.0e-6, f"{key} residual too large: {val}"
