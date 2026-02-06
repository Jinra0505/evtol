from project.src.data_loader import load_data
from project.src.runner import run_equilibrium


def test_inventory_feasible():
    data = load_data("project/data/toy_data.yaml", "project/data_schema.yaml")
    results, _ = run_equilibrium(data)
    inventory = results["inventory"]
    assert inventory["P_vt"]["s1"][1] > 0.0
    for key, val in inventory["residuals"].items():
        assert val <= 1.0e-6
