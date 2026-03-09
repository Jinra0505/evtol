import argparse
import yaml


def build_case():
    times = list(range(1, 11))
    groups = ["low", "mid", "high"]
    ods = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]
    stations = [f"s{i}" for i in range(1, 6)]

    q = {}
    for k, (o, d) in enumerate(ods):
        od = f"{o}-{d}"
        q[od] = {
            "low": {t: 25 + (8 if t in [4, 5, 6] else 0) + k for t in times},
            "mid": {t: 16 + (6 if t in [4, 5, 6] else 0) + k for t in times},
            "high": {t: 9 + (4 if t in [4, 5, 6] else 0) + k for t in times},
        }

    itineraries = []
    dep_map = {f"{o}-{d}": stations[i % len(stations)] for i, (o, d) in enumerate(ods)}
    for o, d in ods:
        od = f"{o}-{d}"
        itineraries.append({
            "id": f"{od.lower().replace('-', '_')}_ev_fast",
            "od": [o, d],
            "mode": "EV",
            "road_arcs": ([{"arc": "a_in", "t": t, "frac": 1.0} for t in times] +
                          [{"arc": "a_cbd", "t": t, "frac": 1.0} for t in times] +
                          [{"arc": "a_out", "t": t, "frac": 0.8} for t in times]),
            "stations": [{"station": dep_map[od], "t": t, "energy": 1.2 + (0.7 if t in [4,5,6] else 0.0)} for t in times],
            "money": 1.3,
            "flight_time": {str(t): 0.0 for t in times},
        })
        itineraries.append({
            "id": f"{od.lower().replace('-', '_')}_ev_slow",
            "od": [o, d],
            "mode": "EV",
            "road_arcs": ([{"arc": "a_in", "t": t, "frac": 1.0} for t in times] +
                          [{"arc": "a_bypass", "t": t, "frac": 1.0} for t in times]),
            "stations": [{"station": dep_map[od], "t": t, "energy": 1.0 + (0.4 if t in [4,5,6] else 0.0)} for t in times],
            "money": 0.9,
            "flight_time": {str(t): 0.0 for t in times},
        })
        itineraries.append({
            "id": f"{od.lower().replace('-', '_')}_vt",
            "od": [o, d],
            "mode": "eVTOL",
            "dep_station": dep_map[od],
            "flight_time": {str(t): 1.0 for t in times},
            "e_per_pax": 1.5,
            "money": 4.5,
            "road_arcs": [],
            "stations": [],
        })

    return {
        "meta": {"T": len(times), "delta_t": 1.0},
        "config": {
            "max_iter": 80,
            "max_iter_auto_extend": 40,
            "min_iters": 12,
            "patience": 8,
            "surcharge_msa_alpha": 0.2,
            "tol": 0.01,
            "eps_improve": "1e-06",
            "use_generator": False,
            "K_paths": 3,
            "shared_power_solver": "highs",
            "voll_ev_per_kwh": 50,
            "voll_vt_per_kwh": 200,
            "shadow_price_scale": 1.0,
            "shadow_price_cap_mult": 10.0,
            "shadow_price_cap_abs": 50.0,
            "vt_reliability_floor": 0.05,
            "vt_reliability_gamma": 1.0,
            "ev_reliability_gamma": 1.0,
            "output_full_json": True,
            "audit_raise": True,
            "strict_audit": True,
            "representative_od": "A-B",
        },
        "sets": {
            "time": times,
            "groups": groups,
            "vot_groups": groups,
            "od_pairs": [list(x) for x in ods],
            "ods": [f"{o}-{d}" for o, d in ods],
            "arcs": ["a_in", "a_cbd", "a_out", "a_bypass"],
            "stations": stations,
            "vehicles": ["m1", "m2", "m3", "m4", "m5"],
        },
        "parameters": {
            "q": q,
            "VOT": {"low": {t: 0.45 for t in times}, "mid": {t: 0.9 for t in times}, "high": {t: 1.5 for t in times}},
            "lambda": {"low": 0.7, "mid": 0.85, "high": 1.0},
            "arcs": {
                "a_in": {"tau0": 1.1, "alpha": 0.15, "beta": 4.0, "cap": 320.0, "type": "ROAD"},
                "a_cbd": {"tau0": 3.0, "alpha": 0.15, "beta": 4.0, "cap": 200.0, "type": "CBD", "theta": 1.0},
                "a_out": {"tau0": 1.0, "alpha": 0.15, "beta": 4.0, "cap": 320.0, "type": "ROAD"},
                "a_bypass": {"tau0": 3.6, "alpha": 0.1, "beta": 3.0, "cap": 420.0, "type": "ROAD"},
            },
            "boundary_in": ["a_in"],
            "boundary_out": ["a_out", "a_bypass"],
            "n0": 35.0,
            "mfd": {"gamma": 0.35, "n_crit": 150.0, "g_max": 2.0},
            "stations": {
                s: {
                    "P_site": {t: (18 + i if t not in [4, 5, 6] else 11 + i) for t in times},
                    "cap_stall": 24.0,
                    "w0": 0.1,
                }
                for i, s in enumerate(stations)
            },
            "electricity_price": {s: {t: (0.18 + 0.01 * i if t not in [4,5,6] else 0.26) for t in times} for i, s in enumerate(stations)},
            "charging": {f"m{i}": {"E_min": 5.0, "E_max": 80.0, "E_init": 25.0, "P_max": 10.0} for i in range(1, 6)},
            "avail": {f"m{i}": {s: {t: 1 for t in times} for s in stations} for i in range(1, 6)},
            "e_fly_or_drive": {f"m{i}": {t: 2.0 for t in times} for i in range(1, 6)},
            "vertiport_storage": {s: {"B_init": 10.0, "B_min": 1.0, "B_max": 90.0, "eta_ch": 0.9} for s in stations},
            "vertiport_cap_pax": {s: {t: (120 if t in [4,5,6] else 180) for t in times} for s in stations},
        },
        "itineraries": itineraries,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synth_case.yaml")
    args = ap.parse_args()
    case = build_case()
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(case, f, sort_keys=False, allow_unicode=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
