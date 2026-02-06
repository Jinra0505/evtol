from project.src.data_loader import load_data


def test_schema_loads():
    data = load_data("project/data/toy_data.yaml", "project/data_schema.yaml")
    assert data["meta"]["T"] == 2
    assert "itineraries" in data
