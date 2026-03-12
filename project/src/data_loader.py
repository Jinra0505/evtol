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

    required_paths = schema.get("required_paths", [])
    try:
        require_paths(data, required_paths)
    except ValueError as exc:
        raise ValueError(f"Schema validation failed for {data_path}: {exc}") from exc

    _validate_basic_shapes(data)
    return data
