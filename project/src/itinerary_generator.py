from typing import Any, Dict, List


def _default_evtol_fare(config: Dict[str, Any], service_class: str) -> float:
    fare_base = float(config.get("vt_fare_base", 4.0))
    premium = float(config.get("vt_fare_premium_fast", 1.2))
    return fare_base + premium if service_class == "fast" else fare_base


def generate_itineraries(
    data: Dict[str, Any],
    travel_times: Dict[str, Dict[int, float]],
    ev_station_waits: Dict[str, Dict[int, float]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not config.get("use_generator", False):
        return []
    # Placeholder for V1+ MILP generation (not activated in toy data).
    # Future generator should support pure EV/eVTOL and multimodal EV_to_eVTOL fast/slow.
    # Keep service-class-aware fallback fares for compatibility.
    _ = _default_evtol_fare(config, "fast")
    _ = _default_evtol_fare(config, "slow")
    return []
