from typing import Any, Dict, List


def generate_itineraries(
    data: Dict[str, Any],
    travel_times: Dict[str, Dict[int, float]],
    station_waits: Dict[str, Dict[int, float]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not config.get("use_generator", False):
        return []
    # Placeholder for V1+ MILP generation (not activated in toy data).
    return []
