from typing import Any, Dict

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




def _add_aggregate_group_k1(data: Dict[str, Any]) -> None:
    params = data.get("parameters", {})
    demand = params.get("q")
    vot = params.get("VOT")
    lambdas = params.get("lambda")
    if not isinstance(demand, dict) or not isinstance(vot, dict) or not isinstance(lambdas, dict):
        return

    has_k1 = any(isinstance(od_groups, dict) and "k1" in od_groups for od_groups in demand.values())
    if has_k1:
        return

    groups = data.get("sets", {}).get("groups") or data.get("sets", {}).get("vot_groups")
    if not isinstance(groups, list) or not groups:
        return

    for od_key, od_groups in demand.items():
        if not isinstance(od_groups, dict):
            continue
        times = set()
        for g in groups:
            g_map = od_groups.get(g, {})
            if isinstance(g_map, dict):
                times.update(g_map.keys())
        od_groups["k1"] = {t: sum(float(od_groups.get(g, {}).get(t, 0.0)) for g in groups) for t in times}

    all_times = set()
    for g in groups:
        g_map = vot.get(g, {})
        if isinstance(g_map, dict):
            all_times.update(g_map.keys())
    if all_times:
        vot["k1"] = {t: sum(float(vot.get(g, {}).get(t, 0.0)) for g in groups) / len(groups) for t in all_times}

    lambdas["k1"] = sum(float(lambdas.get(g, 0.0)) for g in groups) / len(groups)

    set_groups = data.get("sets", {}).get("groups")
    if isinstance(set_groups, list) and "k1" not in set_groups:
        set_groups.append("k1")
    set_vot_groups = data.get("sets", {}).get("vot_groups")
    if isinstance(set_vot_groups, list) and "k1" not in set_vot_groups:
        set_vot_groups.append("k1")

def load_data(data_path: str, schema_path: str) -> Dict[str, Any]:
    schema = load_yaml(schema_path)
    data = load_yaml(data_path)
    data = _coerce_numeric_keys(data)
    data = _coerce_numeric_values(data)
    _add_aggregate_group_k1(data)
    required_paths = schema.get("required_paths", [])
    require_paths(data, required_paths)
    return data
