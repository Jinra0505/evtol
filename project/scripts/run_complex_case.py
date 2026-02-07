import json
import os
from pathlib import Path

from project.src.data_loader import load_data
from project.src.planner import solve_planning
from project.src.runner import run_equilibrium


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _mode_shares(results: dict) -> dict:
    shares = {}
    flows = results["f"]
    for it_id, group_map in flows.items():
        mode = "eVTOL" if "vt" in it_id else "EV"
        for group, time_map in group_map.items():
            for t, flow in time_map.items():
                key = f"{mode}-{group}-t{t}"
                shares[key] = shares.get(key, 0.0) + flow
    return shares


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "complex_case.yaml"
    schema_path = root / "data_schema.yaml"
    outputs_dir = root / "outputs"
    _ensure_dir(str(outputs_dir))

    data = load_data(str(data_path), str(schema_path))

    eq_results, residuals = run_equilibrium(data)
    best_plan, best_cost, best_results, best_diag = solve_planning(data)

    summary = {
        "equilibrium": {
            "residuals": residuals,
            "validation": eq_results["validation"],
            "solver_used": eq_results["solver_used"],
            "shadow_price": eq_results["shadow_price_power"],
            "surcharge": eq_results["surcharge_power"],
            "mode_shares": _mode_shares(eq_results),
        },
        "planning": {
            "best_plan": best_plan,
            "best_cost": best_cost,
            "breakdown": best_diag,
            "mode_shares": _mode_shares(best_results),
        },
    }

    output_path = outputs_dir / "complex_case_results.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
