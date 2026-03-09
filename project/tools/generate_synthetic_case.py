import argparse
import yaml


def build_case():
    times = list(range(1, 13))
    groups = ["low", "mid", "high"]
    ods = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]
    stations = [f"s{i}" for i in range(1, 6)]

    q = {}
    for oi, (o, d) in enumerate(ods):
        key = f"{o}-{d}"
        q[key] = {}
        for g, base in zip(groups, [35, 22, 12]):
            q[key][g] = {t: base + (12 if t in [4,5,6,7] else 0) + oi for t in times}

    itineraries = []
    dep = {f"{o}-{d}": stations[i % len(stations)] for i, (o, d) in enumerate(ods)}
    for o, d in ods:
        od = f"{o}-{d}"
        itineraries.append({
            "id": f"{od.lower().replace('-', '_')}_ev_fast",
            "od": [o, d],
            "mode": "EV",
            "road_arcs": ([{"arc": "a_in", "t": t, "frac": 1.0} for t in times] +
                          [{"arc": "a_cbd", "t": t, "frac": 1.0} for t in times] +
                          [{"arc": "a_out", "t": t, "frac": 0.7} for t in times]),
            "stations": [{"station": dep[od], "t": t, "energy": 1.5 + (0.5 if t in [4,5,6,7] else 0.0)} for t in times],
            "money": 1.2,
            "flight_time": {str(t): 0.0 for t in times},
        })
        itineraries.append({
            "id": f"{od.lower().replace('-', '_')}_ev_slow",
            "od": [o, d],
            "mode": "EV",
            "road_arcs": ([{"arc": "a_in", "t": t, "frac": 1.0} for t in times] +
                          [{"arc": "a_bypass", "t": t, "frac": 1.0} for t in times]),
            "stations": [{"station": dep[od], "t": t, "energy": 1.1 + (0.3 if t in [4,5,6,7] else 0.0)} for t in times],
            "money": 0.8,
            "flight_time": {str(t): 0.0 for t in times},
        })
        itineraries.append({
            "id": f"{od.lower().replace('-', '_')}_vt",
            "od": [o, d],
            "mode": "eVTOL",
            "dep_station": dep[od],
            "flight_time": {str(t): 1.0 for t in times},
            "e_per_pax": 1.55,
            "money": 4.5,
            "road_arcs": [],
            "stations": [],
        })

    case = {
        "meta": {"T": len(times), "delta_t": 1.0},
        "config": {
            "max_iter": 60,
            "min_iters": 12,
            "patience": 8,
            "surcharge_msa_alpha": 0.2,
            "surcharge_kappa": 1.0,
            "surcharge_beta": 2.0,
            "representative_od": "A-B",
            "tol": 0.01,
            "eps_improve": "1e-06",
            "use_generator": False,
            "K_paths": 3,
            "shared_power_solver": "highs",
            "voll_ev_per_kwh": 50,
            "voll_vt_per_kwh": 200,
            "shadow_price_scale": 1.0,
            "shadow_price_cap_mult": 10.0,
            "vt_reliability_floor": 0.05,
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
            "VOT": {"low": {t: 0.45 for t in times}, "mid": {t: 0.9 for t in times}, "high": {t: 1.6 for t in times}},
            "lambda": {"low": 0.7, "mid": 0.85, "high": 1.0},
            "arcs": {
                "a_in": {"tau0": 1.2, "alpha": 0.15, "beta": 4.0, "cap": 450.0, "type": "ROAD"},
                "a_cbd": {"tau0": 3.3, "alpha": 0.15, "beta": 4.0, "cap": 250.0, "type": "CBD", "theta": 1.0},
                "a_out": {"tau0": 1.1, "alpha": 0.15, "beta": 4.0, "cap": 430.0, "type": "ROAD"},
                "a_bypass": {"tau0": 3.8, "alpha": 0.1, "beta": 3.0, "cap": 550.0, "type": "ROAD"},
            },
            "boundary_in": ["a_in"],
            "boundary_out": ["a_out", "a_bypass"],
            "n0": 40.0,
            "mfd": {"gamma": 0.4, "n_crit": 180.0, "g_max": 2.2},
            "stations": {
                s: {
                    "P_site": {t: (16 + i if t not in [4,5,6,7] else 8 + i) for t in times},
                    "cap_stall": 30.0,
                    "w0": 0.1,
                }
                for i, s in enumerate(stations)
            },
            "electricity_price": {s: {t: (0.15 + 0.01 * i if t not in [4,5,6,7] else 0.26) for t in times} for i, s in enumerate(stations)},
            "charging": {f"m{i}": {"E_min": 5.0, "E_max": 70.0, "E_init": 25.0, "P_max": 10.0} for i in range(1, 6)},
            "avail": {f"m{i}": {s: {t: 1 for t in times} for s in stations} for i in range(1, 6)},
            "e_fly_or_drive": {f"m{i}": {t: 2.0 for t in times} for i in range(1, 6)},
            "vertiport_storage": {s: {"B_init": 10.0, "B_min": 1.0, "B_max": 100.0, "eta_ch": 0.9} for s in stations},
            "vertiport_cap_pax": {s: {t: (120 if t in [4,5,6,7] else 180) for t in times} for s in stations},
        },
        "itineraries": itineraries,
    }
    return case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic_case.yaml")
    args = ap.parse_args()
    case = build_case()
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(case, f, sort_keys=False, allow_unicode=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
