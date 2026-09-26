import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_this_dir = Path(__file__).resolve().parent
_root_dir = _this_dir.parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

from stable_baselines3 import SAC, PPO
from bead_gym_env import BeadTraceEnv
from geometry_utils import points_to_polyline_distance

def resolve_path(p, subfolders=("results/models", "datasets/test_images")):
    if not p:
        return p
    for candidate in [p, str(_root_dir / p), *[str(_root_dir / folder / Path(p).name) for folder in subfolders], *[f"{folder}/{Path(p).name}" for folder in subfolders]]:
        if Path(candidate).exists():
            return str(candidate)
    return p

def load_agent(model_path):
    model_path = resolve_path(model_path)
    try:
        return SAC.load(model_path, device="cpu")
    except Exception:
        return PPO.load(model_path, device="cpu")

p=argparse.ArgumentParser()
p.add_argument("--model",default="FINAL_BTP_MODEL_95pct.zip")
p.add_argument("--image",default="path1.jpeg")
p.add_argument("--out",default="results/plots/complete_model_trace.png")
a=p.parse_args()

model_path = resolve_path(a.model)
img_path = resolve_path(a.image)
model = load_agent(model_path)
env = BeadTraceEnv(image_path=img_path, seed=15000)
obs,_=env.reset(); c=env._env; path=np.asarray(c.path_points,float)
traj=[(c.pos_x,c.pos_y)]; reward=0.; done=trunc=False; info={}
while not done and not trunc:
    act,_=model.predict(obs,deterministic=True)
    obs,r,done,trunc,info=env.step(act);reward+=float(r);traj.append((c.pos_x,c.pos_y))
traj=np.asarray(traj,float)
dref=points_to_polyline_distance(path,traj)
dtraj=points_to_polyline_distance(traj,path,path_breaks=set(c.path_breaks))
covered=dref<=.10
geom=float(np.mean(covered));rmse=float(np.sqrt(np.mean(dtraj**2)))

fig,ax=plt.subplots(figsize=(9,8))
ax.plot(path[:,0],path[:,1],linewidth=2.2,label="Reference path")
ax.plot(traj[:,0],traj[:,1],linewidth=1.8,label="Learned SAC trajectory")
ax.scatter(path[covered,0],path[covered,1],s=13,label="Covered within 0.10")
ax.scatter(path[~covered,0],path[~covered,1],s=24,marker="x",label="Outside 0.10")
ax.scatter([traj[0,0]],[traj[0,1]],s=85,marker="o",label="Start")
ax.scatter([traj[-1,0]],[traj[-1,1]],s=105,marker="X",label="Finish")
ax.set_aspect("equal",adjustable="box");ax.grid(True,alpha=.25)
ax.set_xlabel("Arena X");ax.set_ylabel("Arena Y");ax.legend(loc="best")
ax.set_title(
    f"Complete BTP SAC Model\nSequential={info['coverage']*100:.2f}% | "
    f"Geometric@0.10={geom*100:.2f}% | Reward={reward:.2f} | RMSE={rmse:.4f}"
)
fig.tight_layout();fig.savefig(a.out,dpi=250,bbox_inches="tight");plt.close(fig)
print(f"Image -> {a.out}")
print(f"Sequential={info['coverage']:.2%}, Geometric@0.10={geom:.2%}, Reward={reward:.2f}, RMSE={rmse:.4f}")
env.close()
