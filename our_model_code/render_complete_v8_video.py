import sys
import argparse
from pathlib import Path
import cv2, numpy as np

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
p.add_argument("--out",default="results/videos/complete_model_trace.mp4")
p.add_argument("--fps",type=int,default=30)
p.add_argument("--frame_skip",type=int,default=4)
a=p.parse_args()

model_path = resolve_path(a.model)
img_path = resolve_path(a.image)
model = load_agent(model_path)
env = BeadTraceEnv(image_path=img_path, seed=16000)
obs,_=env.reset();c=env._env;path=np.asarray(c.path_points,float)
traj=[(c.pos_x,c.pos_y)];infos=[{"coverage":0.0,"tracking_error":0.0}]
reward=0.; rewards=[0.];done=trunc=False
while not done and not trunc:
    act,_=model.predict(obs,deterministic=True)
    obs,r,done,trunc,info=env.step(act);reward+=float(r)
    traj.append((c.pos_x,c.pos_y));infos.append(dict(info));rewards.append(reward)
traj=np.asarray(traj,float)
dref=points_to_polyline_distance(path,traj);geom=float(np.mean(dref<=.10))
dtraj=points_to_polyline_distance(traj,path,path_breaks=set(c.path_breaks))
rmse=float(np.sqrt(np.mean(dtraj**2)))

W=900;H=820;margin=65
xmin=min(path[:,0].min(),traj[:,0].min());xmax=max(path[:,0].max(),traj[:,0].max())
ymin=min(path[:,1].min(),traj[:,1].min());ymax=max(path[:,1].max(),traj[:,1].max())
pad=.7;xmin-=pad;xmax+=pad;ymin-=pad;ymax+=pad
scale=min((W-2*margin)/(xmax-xmin),(H-2*margin-70)/(ymax-ymin))
def pix(pt):
    x=int(margin+(pt[0]-xmin)*scale)
    y=int(H-margin-(pt[1]-ymin)*scale)
    return (x,y)

fourcc=cv2.VideoWriter_fourcc(*"mp4v")
writer=cv2.VideoWriter(a.out,fourcc,a.fps,(W,H))
if not writer.isOpened():
    raise RuntimeError("Could not open MP4 writer. OpenCV mp4v codec is unavailable.")

ids=list(range(0,len(traj),max(1,a.frame_skip)))
if ids[-1]!=len(traj)-1:ids.append(len(traj)-1)

for idx in ids:
    frame=np.full((H,W,3),248,np.uint8)
    # reference
    for j in range(len(path)-1):
        if (j+1) in set(c.path_breaks): continue
        cv2.line(frame,pix(path[j]),pix(path[j+1]),(150,150,150),2,cv2.LINE_AA)
    # trail
    if idx>0:
        pts=np.array([pix(q) for q in traj[:idx+1]],np.int32)
        cv2.polylines(frame,[pts],False,(210,95,30),3,cv2.LINE_AA)
    # bead
    cv2.circle(frame,pix(traj[idx]),8,(35,35,220),-1,cv2.LINE_AA)
    cv2.circle(frame,pix(traj[0]),6,(30,160,30),-1,cv2.LINE_AA)
    inf=infos[idx]
    cv2.putText(frame,"BTP - Learned SAC Path Tracing",(28,34),cv2.FONT_HERSHEY_SIMPLEX,.78,(25,25,25),2,cv2.LINE_AA)
    cv2.putText(frame,f"Step: {idx}/{len(traj)-1}   Sequential: {100*inf.get('coverage',0):.1f}%   Reward: {rewards[idx]:.1f}",
                (28,64),cv2.FONT_HERSHEY_SIMPLEX,.58,(40,40,40),1,cv2.LINE_AA)
    cv2.putText(frame,f"Current tracking error: {inf.get('tracking_error',0):.4f}",
                (28,89),cv2.FONT_HERSHEY_SIMPLEX,.55,(40,40,40),1,cv2.LINE_AA)
    writer.write(frame)

# Hold final result for 2 seconds
for _ in range(2*a.fps):
    frame=np.full((H,W,3),248,np.uint8)
    for j in range(len(path)-1):
        if (j+1) in set(c.path_breaks): continue
        cv2.line(frame,pix(path[j]),pix(path[j+1]),(150,150,150),2,cv2.LINE_AA)
    pts=np.array([pix(q) for q in traj],np.int32)
    cv2.polylines(frame,[pts],False,(210,95,30),3,cv2.LINE_AA)
    cv2.circle(frame,pix(traj[-1]),8,(35,35,220),-1,cv2.LINE_AA)
    cv2.putText(frame,"FINAL RESULT",(28,34),cv2.FONT_HERSHEY_SIMPLEX,.8,(25,25,25),2,cv2.LINE_AA)
    cv2.putText(frame,f"Sequential: {100*infos[-1]['coverage']:.2f}%   Geometric@0.10: {100*geom:.2f}%",
                (28,65),cv2.FONT_HERSHEY_SIMPLEX,.62,(30,30,30),2,cv2.LINE_AA)
    cv2.putText(frame,f"Reward: {reward:.2f}   Tracking RMSE: {rmse:.4f}",
                (28,92),cv2.FONT_HERSHEY_SIMPLEX,.62,(30,30,30),2,cv2.LINE_AA)
    writer.write(frame)

writer.release();env.close()
print(f"Video -> {a.out}")
print(f"Sequential={infos[-1]['coverage']:.2%}, Geometric@0.10={geom:.2%}, Reward={reward:.2f}, RMSE={rmse:.4f}")
