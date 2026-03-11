from project.src.data_loader import load_data


def test_manual_itineraries_consistent():
    data = load_data("project/data/toy_data.yaml", "project/data_schema.yaml")
    arcs = set(data["sets"]["arcs"])
    stations = set(data["sets"]["stations"])
    times = set(data["sets"]["time"])

    for it in data["itineraries"]:
        for seg in it.get("road_arcs", []):
            assert seg["arc"] in arcs
            assert seg["t"] in times
        for stop in it.get("stations", []):
            assert stop["station"] in stations
            assert stop["t"] in times
