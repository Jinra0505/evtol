#!/usr/bin/env python3
"""Generate a larger stress-test case from larger_case.yaml."""
from __future__ import annotations

import argparse
from copy import deepcopy
import yaml


def scale_nested_numbers(obj, factor: float):
    if isinstance(obj, dict):
        return {k: scale_nested_numbers(v, factor) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scale_nested_numbers(v, factor) for v in obj]
    if isinstance(obj, (int, float)):
        return obj * factor
    return obj


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--base', default='larger_case.yaml')
    p.add_argument('--out', default='mega_case.yaml')
    p.add_argument('--demand-scale', type=float, default=1.8)
    args = p.parse_args()

    with open(args.base, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    out = deepcopy(data)
    cfg = out.setdefault('config', {})
    cfg.update({
        'max_iter': 160,
        'max_total_iter': 800,
        'auto_extend_step': 120,
        'max_iter_auto_extend': 120,
        'patience': 12,
        'tol': 0.008,
        'shadow_price_cap_mult': 6.0,
        'shadow_price_cap_abs': 40.0,
        'vt_reliability_floor': 0.1,
        'ev_reliability_floor': 0.1,
        'vt_reliability_gamma': 1.5,
        'ev_reliability_gamma': 1.5,
        'output_full_json': True,
    })

    q = out.get('parameters', {}).get('q', {})
    out['parameters']['q'] = scale_nested_numbers(q, args.demand_scale)

    with open(args.out, 'w', encoding='utf-8') as f:
        yaml.safe_dump(out, f, sort_keys=False)
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
