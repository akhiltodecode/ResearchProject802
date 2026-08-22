
from pathlib import Path
import json, time, yaml, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from xgboost import DMatrix

from data import make_dataset, FEATURES, TARGET
from models import train_xgb, train_rf, train_mlp, metrics
from decision import recommend_operating_point

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"; OUT.mkdir(exist_ok=True)

def save_feature_importance_xgb(model, X_sample):
    booster=model.get_booster()
    booster.set_param({"device": model.get_params().get("device","cpu")})
    dm=DMatrix(X_sample, feature_names=list(X_sample.columns))
    contrib=booster.predict(dm, pred_contribs=True)[:, :-1]
    importance=np.abs(contrib).mean(axis=0)
    df=pd.DataFrame({"feature":X_sample.columns,"mean_abs_shap":importance}).sort_values("mean_abs_shap",ascending=False)
    df.to_csv(OUT/"xgb_shap_importance.csv",index=False)
    plt.figure(figsize=(8,5))
    d=df.sort_values("mean_abs_shap")
    plt.barh(d["feature"], d["mean_abs_shap"])
    plt.xlabel("Mean |SHAP contribution|")
    plt.tight_layout(); plt.savefig(OUT/"xgb_shap_importance.png",dpi=160); plt.close()

def main():
    cfg=yaml.safe_load((ROOT/"config.yaml").read_text())
    np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU fallback")
    t0=time.time()

    df=make_dataset(int(cfg["n_samples"]), int(cfg["seed"]))
    df.to_csv(OUT/"industrial_energy_dataset.csv", index=False)
    X,y=df[FEATURES],df[TARGET]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=float(cfg["test_size"]),
                                     random_state=int(cfg["seed"]))

    rows=[]

    print("\n[1/3] Training XGBoost...")
    xgb,xgb_device=train_xgb(Xtr,ytr,cfg,use_gpu=torch.cuda.is_available())
    px=xgb.predict(Xte)
    rows.append({"model":"XGBoost", "device":xgb_device, **metrics(yte,px)})
    joblib.dump(xgb,OUT/"xgboost_model.joblib")

    print("\n[2/3] Training Random Forest benchmark...")
    rf=train_rf(Xtr,ytr,cfg)
    pr=rf.predict(Xte)
    rows.append({"model":"RandomForest", "device":"cpu", **metrics(yte,pr)})
    joblib.dump(rf,OUT/"random_forest_model.joblib")

    print("\n[3/3] Training PyTorch MLP...")
    mlp,mlp_device=train_mlp(Xtr,ytr,cfg)
    pm=mlp.predict(Xte)
    rows.append({"model":"PyTorchMLP", "device":mlp_device, **metrics(yte,pm)})

    results=pd.DataFrame(rows).sort_values("RMSE")
    results.to_csv(OUT/"model_metrics.csv",index=False)
    print("\nMODEL RESULTS\n", results.to_string(index=False))

    # Always use XGBoost for interpretable decision layer.
    sample=Xte.sample(min(5000,len(Xte)),random_state=int(cfg["seed"]))
    save_feature_importance_xgb(xgb,sample)

    # Test decision intelligence over representative machine states.
    refs=Xte.sample(25,random_state=int(cfg["seed"]))
    decisions=[]
    for idx,row in refs.iterrows():
        rec=recommend_operating_point(
            xgb,row,
            n_candidates=int(cfg["decision_candidates"]),
            seed=int(cfg["seed"])+int(idx)%10000
        )
        rec["row_id"]=int(idx)
        rec["annualized_cost_saving_usd_at_8h_250d"] = rec["saving_kwh"]*8*250*float(cfg["energy_price_per_kwh"])
        decisions.append(rec)
    flat=[]
    for d in decisions:
        r={k:v for k,v in d.items() if k!="recommended_settings"}
        r.update({f"recommended_{k}":v for k,v in d["recommended_settings"].items()})
        flat.append(r)
    dd=pd.DataFrame(flat)
    dd.to_csv(OUT/"decision_recommendations.csv",index=False)

    summary={
        "title":"AI-Powered Data Analytics and Decision Intelligence for Energy-Efficient Industrial Machines",
        "runtime_minutes":round((time.time()-t0)/60,2),
        "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "best_model_by_rmse":results.iloc[0]["model"],
        "xgboost_metrics":rows[0],
        "mean_recommended_energy_saving_pct":float(dd["saving_pct"].mean()),
        "median_recommended_energy_saving_pct":float(dd["saving_pct"].median()),
        "mean_annualized_cost_saving_usd":float(dd["annualized_cost_saving_usd_at_8h_250d"].mean()),
        "note":"Synthetic-data proof-of-concept. Replace generator with measured industrial sensor data for external validity."
    }
    (OUT/"experiment_summary.json").write_text(json.dumps(summary,indent=2))
    print("\nDECISION INTELLIGENCE SUMMARY")
    print(json.dumps(summary,indent=2))
    print(f"\nDone. Results: {OUT}")

if __name__=="__main__":
    main()
