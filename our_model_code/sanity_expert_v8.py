import sys
import argparse, numpy as np
from pathlib import Path

_this_dir = Path(__file__).resolve().parent
_root_dir = _this_dir.parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

from bead_gym_env import BeadTraceEnv
from geometry_utils import points_to_polyline_distance

def resolve_path(p, subfolders=("results/models", "datasets/test_images")):
    if not p:
        return p
    for candidate in [p, str(_root_dir / p), *[str(_root_dir / folder / Path(p).name) for folder in subfolders], *[f"{folder}/{Path(p).name}" for folder in subfolders]]:
        if Path(candidate).exists():
            return str(candidate)
    return p

p=argparse.ArgumentParser();p.add_argument("--image",default="path1.jpeg");p.add_argument("--episodes",type=int,default=5);a=p.parse_args()
a.image = resolve_path(a.image)
vals=[]
for ep in range(a.episodes):
    env=BeadTraceEnv(a.image,100+ep);obs,_=env.reset();c=env._env
    traj=[(c.pos_x,c.pos_y)];rw=0;errs=[];d=t=False
    while not(d or t):
        wi=min(c.waypoint_idx,len(c.path_points)-1);ti=min(wi+1,len(c.path_points)-1)
        v=c.path_points[ti]-np.array([c.pos_x,c.pos_y]);n=np.linalg.norm(v)
        act=(v/n if n>1e-8 else np.zeros(2)).astype(np.float32)
        obs,r,d,t,info=env.step(act);rw+=r;errs.append(info["tracking_error"]);traj.append((c.pos_x,c.pos_y))
    geom=np.mean(points_to_polyline_distance(c.path_points,np.asarray(traj))<=.10)
    vals.append((info["coverage"],geom,rw,np.sqrt(np.mean(np.square(errs))),len(errs)))
    print(f"Episode {ep+1}: seq={info['coverage']:.2%} geom@.10={geom:.2%} reward={rw:.2f} RMSE={vals[-1][3]:.4f} steps={len(errs)}")
    env.close()
print("Mean:",np.mean(vals,axis=0))
