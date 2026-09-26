"""Zero-shot evaluation on held-out synthetic test splits (never seen in training)."""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_paper_experiments import run_bead, run_pp, OUT  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

N_PER_SET = 25
SETS = {
    "closed": sorted((ROOT / "datasets/closed_shapes_dataset/test").glob("*.jpg"))[:N_PER_SET],
    "curve": sorted((ROOT / "datasets/curve_dataset/test").glob("*.*"))[:N_PER_SET],
}


def main():
    bead = PPO.load(str(ROOT / "results/models/PPO_BTP_MODEL.zip"), device="cpu")
    rlpp = PPO.load(str(ROOT / "results/models/RL_PP_JOINT_MODEL.zip"), device="cpu")
    res, gallery = {}, {}
    for sname, files in SETS.items():
        res[sname] = {"bead_ppo": [], "joint": [], "fixed": []}
        for k, f in enumerate(files):
            mb, tb = run_bead(bead, str(f), 777)
            mj, tj = run_pp("joint", rlpp, str(f), 777)
            mf, _ = run_pp("fixed", None, str(f), 777)
            res[sname]["bead_ppo"].append(mb); res[sname]["joint"].append(mj); res[sname]["fixed"].append(mf)
            if k < 4:
                np.savez_compressed(OUT / f"shape_{sname}_{k}.npz", path=tb["path"], bead=tb["traj"], joint=tj["traj"],
                                    gb=mb["geom010"], gj=mj["geom010"])
        for c, rows in res[sname].items():
            g = np.array([r["geom010"] for r in rows]); e = np.array([r["rmse"] for r in rows])
            s = np.array([r["sequential"] for r in rows])
            print(f"{sname:6s} {c:9s} n={len(rows)} geom@0.10={g.mean():.4f}±{g.std():.4f} "
                  f"median={np.median(g):.4f} rmse={e.mean():.4f} seq={s.mean():.4f}", flush=True)
    (OUT / "shapes.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
