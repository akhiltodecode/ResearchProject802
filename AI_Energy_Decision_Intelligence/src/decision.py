
import numpy as np
import pandas as pd

CONTROLLABLE = ["machine_load_pct","spindle_speed_rpm","feed_rate_mm_min","pressure_bar"]

def recommend_operating_point(model, reference_row, n_candidates=20000, seed=42):
    rng=np.random.default_rng(seed)
    base=reference_row.copy()
    candidates=pd.DataFrame(np.repeat(base.to_numpy()[None,:], n_candidates, axis=0),
                            columns=base.index)
    # Keep production useful while searching within feasible operating bounds.
    def bounds(value, lo, hi, down, up):
        a=max(lo, value*down); b=min(hi, value*up)
        if a > b:
            a=b=float(np.clip(value, lo, hi))
        return a,b
    a,b=bounds(base["machine_load_pct"],25,100,.75,1.05)
    candidates["machine_load_pct"] = rng.uniform(a,b,n_candidates)
    a,b=bounds(base["spindle_speed_rpm"],800,5000,.75,1.05)
    candidates["spindle_speed_rpm"] = rng.uniform(a,b,n_candidates)
    a,b=bounds(base["feed_rate_mm_min"],100,1800,.80,1.10)
    candidates["feed_rate_mm_min"] = rng.uniform(a,b,n_candidates)
    a,b=bounds(base["pressure_bar"],3.5,9.5,.80,1.05)
    candidates["pressure_bar"] = rng.uniform(a,b,n_candidates)

    pred=np.asarray(model.predict(candidates), dtype=float)
    # Throughput proxy: must preserve >=90% of the reference productive intensity.
    throughput=(candidates["machine_load_pct"]*candidates["feed_rate_mm_min"])
    ref_throughput=base["machine_load_pct"]*base["feed_rate_mm_min"]
    feasible=throughput >= 0.90*ref_throughput
    if feasible.any():
        idx=np.where(feasible)[0][np.argmin(pred[feasible])]
    else:
        idx=int(np.argmin(pred))
    chosen=candidates.iloc[idx].copy()
    baseline=float(model.predict(pd.DataFrame([base]))[0])
    optimized=float(pred[idx])
    return {
        "baseline_predicted_kwh": baseline,
        "optimized_predicted_kwh": optimized,
        "saving_kwh": baseline-optimized,
        "saving_pct": 100*(baseline-optimized)/max(baseline,1e-9),
        "recommended_settings": {k: float(chosen[k]) for k in CONTROLLABLE}
    }
