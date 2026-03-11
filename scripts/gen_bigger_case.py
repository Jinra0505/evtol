import yaml

times = [1,2,3,4,5,6]
groups = ["low","mid","high"]
ods = [("A","B"),("A","C"),("B","C")]
stations = ["s1","s2","s3"]

q = {}
for o,d in ods:
    k=f"{o}-{d}"
    q[k]={
        "low": {t: (40 if t in [3,4] else 22) for t in times},
        "mid": {t: (28 if t in [3,4] else 15) for t in times},
        "high": {t: (15 if t in [3,4] else 8) for t in times},
    }

vot = {
    "low": {t: 0.45 for t in times},
    "mid": {t: 0.9 for t in times},
    "high": {t: 1.6 for t in times},
}
lam = {"low": 0.7, "mid": 0.85, "high": 1.0}

arcs = {
    "a_in": {"tau0": 1.2, "alpha": 0.15, "beta": 4.0, "cap": 230.0, "type": "ROAD"},
    "a_cbd": {"tau0": 3.4, "alpha": 0.15, "beta": 4.0, "cap": 140.0, "type": "CBD", "theta": 1.0},
    "a_out": {"tau0": 1.1, "alpha": 0.15, "beta": 4.0, "cap": 230.0, "type": "ROAD"},
    "a_bypass": {"tau0": 3.8, "alpha": 0.1, "beta": 3.0, "cap": 300.0, "type": "ROAD"},
}

boundary_in = ["a_in"]
boundary_out = ["a_out", "a_bypass"]

Psite = {
    "s1": {1: 16, 2: 15, 3: 8, 4: 8, 5: 14, 6: 15},
    "s2": {1: 14, 2: 14, 3: 7, 4: 7, 5: 12, 6: 13},
    "s3": {1: 15, 2: 14, 3: 9, 4: 8, 5: 13, 6: 14},
}

prices = {
    s: {t: (0.24 if t in [3,4] else 0.16 + 0.01*idx) for t in times}
    for idx, s in enumerate(stations)
}

itineraries=[]
dep_map = {"A-B":"s1","A-C":"s2","B-C":"s3"}
for o,d in ods:
    od=f"{o}-{d}"
    # EV fast
    itineraries.append({
        "id": f"{od.lower().replace('-','_')}_ev_fast",
        "od": [o,d],
        "mode": "EV",
        "road_arcs": [{"arc":"a_in","t":t,"frac":1.0} for t in times] + [{"arc":"a_cbd","t":t,"frac":1.0} for t in times] + [{"arc":"a_out","t":t,"frac":0.7} for t in times],
        "stations": [{"station": dep_map[od], "t": t, "energy": 2.1 if t in [3,4] else 1.4} for t in times],
        "money": 1.2,
        "flight_time": {str(t): 0.0 for t in times},
    })
    # EV cheap via bypass
    itineraries.append({
        "id": f"{od.lower().replace('-','_')}_ev_slow",
        "od": [o,d],
        "mode": "EV",
        "road_arcs": [{"arc":"a_in","t":t,"frac":1.0} for t in times] + [{"arc":"a_bypass","t":t,"frac":1.0} for t in times],
        "stations": [{"station": dep_map[od], "t": t, "energy": 1.6 if t in [3,4] else 1.1} for t in times],
        "money": 0.8,
        "flight_time": {str(t): 0.0 for t in times},
    })
    # eVTOL
    itineraries.append({
        "id": f"{od.lower().replace('-','_')}_vt",
        "od": [o,d],
        "mode": "eVTOL",
        "dep_station": dep_map[od],
        "flight_time": {str(t): 1.2 for t in times},
        "e_per_pax": 1.85 if od=="A-B" else 1.65,
        "money": 4.8,
        "road_arcs": [],
        "stations": [],
    })

obj = {
    "meta": {"T": len(times), "delta_t": 1.0},
    "config": {
        "max_iter": 50,
        "min_iters": 12,
        "patience": 7,
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
        "ods": [f"{o}-{d}" for o,d in ods],
        "arcs": list(arcs.keys()),
        "stations": stations,
        "vehicles": ["m1","m2","m3"],
    },
    "parameters": {
        "q": q,
        "VOT": vot,
        "lambda": lam,
        "arcs": arcs,
        "boundary_in": boundary_in,
        "boundary_out": boundary_out,
        "n0": 30.0,
        "mfd": {"gamma": 0.45, "n_crit": 95.0, "g_max": 2.2},
        "stations": {s: {"P_site": Psite[s], "cap_stall": 24.0, "w0": 0.12 if s == "s1" else 0.1} for s in stations},
        "electricity_price": prices,
        "charging": {
            "m1": {"E_min": 5.0, "E_max": 60.0, "E_init": 20.0, "P_max": 8.0},
            "m2": {"E_min": 5.0, "E_max": 60.0, "E_init": 22.0, "P_max": 8.0},
            "m3": {"E_min": 5.0, "E_max": 60.0, "E_init": 18.0, "P_max": 8.0},
        },
        "avail": {m: {s: {t: 1 for t in times} for s in stations} for m in ["m1","m2","m3"]},
        "e_fly_or_drive": {m: {t: 2.0 for t in times} for m in ["m1","m2","m3"]},
        "vertiport_storage": {s: {"B_init": 10.0, "B_min": 1.0, "B_max": 80.0, "eta_ch": 0.9} for s in stations},
        "vertiport_cap_pax": {s: {t: (80 if t in [3,4] else 120) for t in times} for s in stations},
    },
    "itineraries": itineraries,
}

with open('bigger_case.yaml','w',encoding='utf-8') as f:
    yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)
print('wrote bigger_case.yaml')
