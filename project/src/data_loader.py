from typing import Any, Dict, List, Set

from .assignment import is_evtol_itinerary
from .utils import load_yaml, require_paths


def _coerce_numeric_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        new_obj: Dict[Any, Any] = {}
        for key, value in obj.items():
            new_key = int(key) if isinstance(key, str) and key.isdigit() else key
            new_obj[new_key] = _coerce_numeric_keys(value)
        return new_obj
    if isinstance(obj, list):
        return [_coerce_numeric_keys(item) for item in obj]
    return obj


def _coerce_numeric_values(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _coerce_numeric_values(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_coerce_numeric_values(item) for item in obj]
    if isinstance(obj, str):
        try:
            num = float(obj)
        except ValueError:
            return obj
        if obj.strip().isdigit():
            return int(obj)
        return num
    return obj


def _normalize_od_key(od: Any) -> str:
    if isinstance(od, (list, tuple)) and len(od) >= 2:
        return f"{od[0]}-{od[1]}"
    txt = str(od)
    if "->" in txt:
        a, b = txt.split("->", 1)
        return f"{a.strip()}-{b.strip()}"
    if "-" in txt:
        a, b = txt.split("-", 1)
        return f"{a.strip()}-{b.strip()}"
    return txt


def _normalize_od_structures(data: Dict[str, Any]) -> None:
    params = data.setdefault("parameters", {})
    q = params.get("q")
    if isinstance(q, dict):
        q_new = {}
        for od, v in q.items():
            q_new[_normalize_od_key(od)] = v
        params["q"] = q_new
    its = data.get("itineraries", [])
    for it in its if isinstance(its, list) else []:
        od = it.get("od")
        if isinstance(od, str):
            key = _normalize_od_key(od)
            if "-" in key:
                a, b = key.split("-", 1)
                it["od"] = [a, b]


def _harmonize_access_energy_fields(data: Dict[str, Any]) -> None:
    """Use explicit access_stations energy as canonical source for EV_to_eVTOL itineraries."""
    its = data.get("itineraries", [])
    for it in its if isinstance(its, list) else []:
        mode = str(it.get("mode", "")).lower()
        if not mode.startswith("ev_to_evtol"):
            continue
        if "access_energy_kwh" in it:
            # Keep one canonical field to avoid overlap/conflict.
            it.pop("access_energy_kwh", None)


def _validate_basic_shapes(data: Dict[str, Any]) -> None:
    sets = data.get("sets", {})
    params = data.get("parameters", {})
    times: List[int] = list(sets.get("time", []))
    groups: List[str] = list(sets.get("groups", []))

    if not isinstance(times, list) or not times:
        raise ValueError("Invalid input: sets.time must be a non-empty list of time indices")
    if not isinstance(groups, list) or not groups:
        raise ValueError("Invalid input: sets.groups must be a non-empty list of traveler groups")

    q = params.get("q", {})
    if not isinstance(q, dict):
        raise ValueError("Invalid input: parameters.q must be a dict {OD -> group -> time -> demand}")

    for od, g_map in q.items():
        if not isinstance(g_map, dict):
            raise ValueError(f"Invalid demand block for OD={od}: expected dict by group")
        for g in groups:
            if g not in g_map:
                raise ValueError(f"Demand missing group '{g}' under parameters.q['{od}']")
            t_map = g_map[g]
            if not isinstance(t_map, dict):
                raise ValueError(f"Demand for OD={od}, group={g} must be a dict by time")
            missing_t = [t for t in times if t not in t_map]
            if missing_t:
                raise ValueError(f"Demand missing times for OD={od}, group={g}: {missing_t[:5]}")


def _require_hybrid_param_complete(params: Dict[str, Any], hybrid_stations: Set[str], key: str, times: List[int] | None = None) -> None:
    mapping = params.get(key, {})
    kset = set(str(k) for k in mapping.keys())
    missing = sorted(hybrid_stations - kset)
    extra = sorted(kset - hybrid_stations)
    if missing:
        raise ValueError(f"{key} missing hybrid stations: {missing}")
    if extra:
        raise ValueError(f"{key} contains non-hybrid stations: {extra}")
    if times is not None:
        for s in hybrid_stations:
            tmap = mapping.get(s, {})
            if not isinstance(tmap, dict):
                raise ValueError(f"{key}.{s} must be a dict by time")
            miss_t = [t for t in times if t not in tmap]
            if miss_t:
                raise ValueError(f"{key}.{s} missing time keys: {miss_t[:5]}")


def _validate_station_facility_consistency(data: Dict[str, Any]) -> None:
    sets = data.get("sets", {})
    params = data.get("parameters", {})
    ev_stations: Set[str] = set(str(x) for x in sets.get("ev_stations", []))
    hybrid_stations: Set[str] = set(str(x) for x in sets.get("hybrid_stations", []))
    stations: Set[str] = set(str(x) for x in sets.get("stations", []))

    if not ev_stations:
        raise ValueError("sets.ev_stations must be non-empty")
    if not hybrid_stations:
        raise ValueError("sets.hybrid_stations must be non-empty")
    if not hybrid_stations.issubset(ev_stations):
        raise ValueError("Invalid station sets: hybrid_stations must be subset of ev_stations")
    if stations and stations != ev_stations:
        raise ValueError("Invalid station sets: sets.stations must equal sets.ev_stations in EV-only/Hybrid mode")
    if params.get("vertiports") is not None:
        raise ValueError("Do not provide parameters.vertiports; use sets.hybrid_stations only")

    station_params = set(str(k) for k in params.get("stations", {}).keys())
    if not ev_stations.issubset(station_params):
        missing = sorted(ev_stations - station_params)
        raise ValueError(f"parameters.stations missing ev_stations entries: {missing}")

    elec = set(str(k) for k in params.get("electricity_price", {}).keys())
    if not ev_stations.issubset(elec):
        missing = sorted(ev_stations - elec)
        raise ValueError(f"parameters.electricity_price missing ev_stations entries: {missing}")

    times: List[int] = list(sets.get("time", []))
    _require_hybrid_param_complete(params, hybrid_stations, "vertiport_storage", None)
    _require_hybrid_param_complete(params, hybrid_stations, "vt_aircraft_init_by_station", None)
    _require_hybrid_param_complete(params, hybrid_stations, "vertiport_cap_pax", times)
    _require_hybrid_param_complete(params, hybrid_stations, "vt_departure_capacity_total", times)
    _require_hybrid_param_complete(params, hybrid_stations, "vt_departure_capacity_fast", times)

    itineraries = data.get("itineraries", []) if isinstance(data.get("itineraries"), list) else []
    for it in itineraries:
        it_id = it.get("id", "<unknown>")
        mode = str(it.get("mode", "")).lower()

        for stop in (it.get("stations", []) or []):
            s = str(stop.get("station"))
            if s not in ev_stations:
                raise ValueError(f"Itinerary {it_id}: EV charging station {s} not in ev_stations")
        for stop in (it.get("access_stations", []) or []):
            s = str(stop.get("station"))
            if s not in ev_stations:
                raise ValueError(f"Itinerary {it_id}: access station {s} not in ev_stations")

        if is_evtol_itinerary(it) or mode.startswith("ev_to_evtol"):
            dep = str(it.get("dep_station")) if it.get("dep_station") is not None else None
            if dep is None or dep not in hybrid_stations:
                raise ValueError(f"Itinerary {it_id}: dep_station must be in hybrid_stations")
            arr = str(it.get("arr_station")) if it.get("arr_station") is not None else None
            if arr is not None and arr not in hybrid_stations:
                raise ValueError(f"Itinerary {it_id}: arr_station must be in hybrid_stations")


def load_data(data_path: str, schema_path: str) -> Dict[str, Any]:
    try:
        schema = load_yaml(schema_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Schema file not found: {schema_path}. Expected for required_paths validation."
        ) from exc

    try:
        data = load_yaml(data_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Data file not found: {data_path}") from exc

    data = _coerce_numeric_keys(data)
    data = _coerce_numeric_values(data)
    _normalize_od_structures(data)
    _harmonize_access_energy_fields(data)

    required_paths = schema.get("required_paths", [])
    try:
        require_paths(data, required_paths)
    except ValueError as exc:
        raise ValueError(f"Schema validation failed for {data_path}: {exc}") from exc

    _validate_basic_shapes(data)
    _validate_station_facility_consistency(data)
    return data
