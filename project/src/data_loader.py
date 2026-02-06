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


def load_data(data_path: str, schema_path: str) -> Dict[str, Any]:
    schema = load_yaml(schema_path)
    data = load_yaml(data_path)
    data = _coerce_numeric_keys(data)
    required_paths = schema.get("required_paths", [])
    require_paths(data, required_paths)
    return data
