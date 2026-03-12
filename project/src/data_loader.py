from typing import Any, Dict, List

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
    """Canonicalize multimodal access energy with explicit access-station energy as source of truth."""
    its = data.get("itineraries", [])
    for it in its if isinstance(its, list) else []:
        mode = str(it.get("mode", "")).lower()
        if not mode.startswith("ev_to_evtol"):
            continue

        explicit_by_t: Dict[int, float] = {}
        for stop in it.get("access_stations", []) or []:
            try:
                t = int(stop.get("t"))
            except (TypeError, ValueError):
                continue
            explicit_by_t[t] = explicit_by_t.get(t, 0.0) + float(stop.get("energy", 0.0) or 0.0)

        if "access_energy_kwh" not in it:
            continue
        scalar = it.get("access_energy_kwh")
        if isinstance(scalar, dict):
            aligned = {int(k): float(v) for k, v in scalar.items()}
            for t, e in explicit_by_t.items():
                aligned[t] = e
            it["access_energy_kwh"] = aligned
        elif explicit_by_t:
            it["access_energy_kwh"] = sum(explicit_by_t.values()) / max(1, len(explicit_by_t))

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


def load_data(data_path: str, schema_path: str) -> Dict[str, Any]:
    """Load scenario data and validate required paths/basic dimensions.

    Raises clear ValueError for missing paths, malformed structures and basic
    time/group dimensional inconsistencies.
    """
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
    return data
