from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple
from pathlib import Path


class BeadEnvironment:

    ARENA     = 10.0
    BEAD_R    = 0.15
    BOUNDARY  = ARENA - BEAD_R
    MAX_STEPS = 3500

    # Physics
    ACCEL     = 10.0
    MAX_SPEED = 7.0
    FRICTION  = 7.0
    BOUNCE    = 0.15
    DT        = 0.016

    # Coverage
    COV_RADIUS = 0.18
    STRICT_RADIUS = 0.10
    ACCEL_RADIUS = 4.3

    # Reward weights
    R_APPROACH    =  0.10
    R_WAYPOINT    =  0.40
    R_PRECISION   =  0.25   # Gaussian centering reward exp(-(e_lat/0.08)^2)
    R_TANGENT     =  0.15   # Forward velocity alignment with path tangent
    R_TRACK_ERR   = -0.05
    R_OFF_CURVE   = -0.10
    R_BACKTRACK   = -0.05
    R_BOUNDARY    = -0.50
    R_SMOOTH      = -0.005
    R_STEP        = -0.001
    R_COMPLETION  = 50.0
    R_PARTIAL80   = 15.0
    R_INCOMPLETE  = -5.0

    def __init__(self, image_path: Optional[str | list | tuple] = None, seed: int = 42):
        np.random.seed(seed)
        self.seed = seed

        self.pos_x = self.pos_y = 0.0
        self.vel_x = self.vel_y = 0.0
        self.step_count = 0

        self.path_points: np.ndarray = np.empty((0, 2), dtype=float)
        self.path_breaks: set[int]   = set()
        self.covered_points: set[int] = set()

        self.waypoint_idx = 0
        self.prev_dist    = 0.0
        self.prev_action  = np.zeros(2, np.float32)
        self.hit_boundary = False
        self.last_reward_components = self._empty_reward_components()
        self.strict_covered_points: set[int] = set()
        self.no_progress_steps = 0

        self.image_paths = []
        if image_path:
            self._init_image_paths(image_path)
            if self.image_paths:
                self.load_contours(str(np.random.choice(self.image_paths)))
            else:
                self._create_default_path()
        else:
            self._create_default_path()

    def _resolve_candidate(self, p: str | Path) -> Optional[Path]:
        p = Path(p)
        if p.exists():
            return p
        root = Path(__file__).resolve().parent.parent
        candidates = [
            root / p,
            root / "datasets" / "test_images" / p.name,
            root / "datasets" / "research_paper_tracks" / p.name,
            Path("datasets/test_images") / p.name,
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def _init_image_paths(self, image_input: str | list | tuple | Path):
        self.image_paths = []
        if isinstance(image_input, (list, tuple)):
            for item in image_input:
                res = self._resolve_candidate(item)
                if res and res.is_file():
                    self.image_paths.append(res)
        elif str(image_input) in ("all_tracks", "tracks", "three_tracks"):
            for t in ["hockenheim_track.png", "montreal_track.png", "yasmarina_track.png"]:
                res = self._resolve_candidate(t)
                if res and res.is_file():
                    self.image_paths.append(res)
        else:
            res = self._resolve_candidate(image_input)
            if res:
                if res.is_dir():
                    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG'):
                        self.image_paths.extend(list(res.glob(ext)))
                elif res.is_file():
                    self.image_paths = [res]

    def _create_default_path(self):
        t = np.linspace(0, 2*np.pi, 120, endpoint=False)
        self.path_points = np.stack([4.0*np.cos(t), 4.0*np.sin(t)], 1).astype(float)
        self.path_breaks = set()

    def load_contours(self, image_path: str):
        p = Path(image_path)
        
        # Resolve directory input by selecting a random image file from within it
        if p.is_dir():
            valid_images = []
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG'):
                valid_images.extend(list(p.glob(ext)))
            if not valid_images:
                print(f"Warning: No valid images found in {image_path}")
                self._create_default_path()
                return
            image_path = str(np.random.choice(valid_images))

        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: cannot load {image_path}")
            self._create_default_path()
            return

        img  = self._resize_img(img)
        h, w = img.shape[:2]
        fg   = self._foreground_mask(img)
        ratio= cv2.countNonZero(fg) / max(1, fg.size)

        paths = self._extract_stroke_paths(fg) if ratio <= 0.20 else []
        if not paths:
            paths = self._extract_contour_paths(fg)
        if not paths:
            self._create_default_path(); return

        infos = [(self._path_length(self._to_arena(p,w,h)), self._to_arena(p,w,h))
                 for p in paths]
        infos = [(l,p) for l,p in infos if l >= 0.1]
        if not infos:
            self._create_default_path(); return

        infos.sort(key=lambda x: x[0], reverse=True)
        total_len = sum(l for l,_ in infos)
        resampled, self.path_breaks = [], set()
        for ln, pts in infos:
            n  = max(20, int(350*ln/total_len))
            rp = self._resample(pts, n)
            if resampled:
                self.path_breaks.add(sum(len(p) for p in resampled))
            resampled.append(rp)

        self.path_points = np.vstack(resampled).astype(float)

    @staticmethod
    def _resize_img(img, max_dim=1200):
        h,w=img.shape[:2]; m=max(h,w)
        if m<=max_dim: return img
        s=max_dim/m
        return cv2.resize(img,(max(1,round(w*s)),max(1,round(h*s))),interpolation=cv2.INTER_AREA)

    @staticmethod
    def _foreground_mask(img):
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        blurred=cv2.GaussianBlur(gray,(5,5),0)
        border=np.concatenate([blurred[0,:],blurred[-1,:],blurred[:,0],blurred[:,-1]])
        mode=cv2.THRESH_BINARY_INV if np.median(border)>=127 else cv2.THRESH_BINARY
        _,fg=cv2.threshold(blurred,0,255,mode+cv2.THRESH_OTSU)
        return cv2.morphologyEx(fg,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))

    def _extract_stroke_paths(self, fg):
        skel=self._zhang_suen(fg)
        n,lbl,stats,_=cv2.connectedComponentsWithStats(skel,connectivity=8)
        paths=[]
        for cid in range(1,n):
            if stats[cid,cv2.CC_STAT_AREA]<10: continue
            mask=(lbl==cid).astype(np.uint8)
            op=self._order_skeleton(mask)
            if len(op)>=2: paths.append(op)
        return paths

    @staticmethod
    def _extract_contour_paths(fg):
        cnts,_=cv2.findContours(fg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        area_img=fg.shape[0]*fg.shape[1]; paths=[]
        for c in cnts:
            if cv2.arcLength(c,True)<10 or cv2.contourArea(c)<area_img*0.0005: continue
            pts=c.reshape(-1,2).astype(float)
            if not np.allclose(pts[0],pts[-1]): pts=np.vstack([pts,pts[0]])
            paths.append(pts)
        return paths

    @staticmethod
    def _zhang_suen(fg,max_iter=200):
        img=(fg>0).astype(np.uint8); img[[0,-1],:]=0; img[:,[0,-1]]=0
        def T(a,b): return ((a==0)&(b==1)).astype(np.uint8)
        for _ in range(max_iter):
            changed=False
            for step in(0,1):
                p2=img[:-2,1:-1];p3=img[:-2,2:];p4=img[1:-1,2:];p5=img[2:,2:]
                p6=img[2:,1:-1];p7=img[2:,:-2];p8=img[1:-1,:-2];p9=img[:-2,:-2];p1=img[1:-1,1:-1]
                nc=p2+p3+p4+p5+p6+p7+p8+p9
                tc=T(p2,p3)+T(p3,p4)+T(p4,p5)+T(p5,p6)+T(p6,p7)+T(p7,p8)+T(p8,p9)+T(p9,p2)
                if step==0: rm=(p1==1)&(nc>=2)&(nc<=6)&(tc==1)&((p2*p4*p6)==0)&((p4*p6*p8)==0)
                else:       rm=(p1==1)&(nc>=2)&(nc<=6)&(tc==1)&((p2*p4*p8)==0)&((p2*p6*p8)==0)
                if np.any(rm): img[1:-1,1:-1][rm]=0; changed=True
            if not changed: break
        return img

    @staticmethod
    def _order_skeleton(mask):
        pixels=[tuple(map(int,p)) for p in np.argwhere(mask>0)]
        if len(pixels)<=1: return np.empty((0,2),float)
        pset=set(pixels)
        off=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        def nbrs(p): y,x=p; return [n for n in[(y+dy,x+dx) for dy,dx in off] if n in pset]
        degs={p:len(nbrs(p)) for p in pixels}
        eps=[p for p,d in degs.items() if d==1]
        start=min(eps or pixels,key=lambda p:(p[1],p[0]))
        path,vis,prev,cur=[start],{start},None,start
        for _ in range(len(pixels)+2):
            cands=[n for n in nbrs(cur) if n!=prev]
            unvis=[n for n in cands if n not in vis]
            if not unvis:
                if not eps and start in cands and len(path)>2: path.append(start)
                break
            def sc(n,prev=prev,cur=cur):
                if prev is None: return(n[1],n[0])
                dy0,dx0=cur[0]-prev[0],cur[1]-prev[1]; dy1,dx1=n[0]-cur[0],n[1]-cur[1]
                n0=max(1e-6,(dy0**2+dx0**2)**.5); n1=max(1e-6,(dy1**2+dx1**2)**.5)
                return(-(dy0*dy1+dx0*dx1)/(n0*n1),n1)
            nxt=min(unvis,key=sc); prev,cur=cur,nxt; path.append(cur); vis.add(cur)
        return np.array([[x,y] for y,x in path],float)

    def _to_arena(self,pts,w,h):
        s=(2*self.BOUNDARY)/max(1,w-1,h-1)
        out=np.empty_like(pts,float)
        out[:,0]=(pts[:,0]-(w-1)/2)*s; out[:,1]=((h-1)/2-pts[:,1])*s
        return out

    @staticmethod
    def _path_length(pts):
        if len(pts)<=1: return 0.0
        return float(np.sum(np.linalg.norm(np.diff(pts,axis=0),axis=1)))

    @staticmethod
    def _resample(pts, n):
        if len(pts)<=1: return pts
        d=np.sqrt(np.sum(np.diff(pts,axis=0)**2,axis=1))
        cum=np.concatenate([[0],np.cumsum(d)]); tot=cum[-1]
        if tot==0: return pts
        ts=np.linspace(0,tot,n)
        idx=np.clip(np.searchsorted(cum,ts)-1,0,len(pts)-2)
        seg=np.where(cum[idx+1]-cum[idx]>0,(ts-cum[idx])/(cum[idx+1]-cum[idx]+1e-12),0)
        return pts[idx]+seg[:,None]*(pts[idx+1]-pts[idx])

    def reset(self) -> Tuple[np.ndarray, dict]:
        if len(self.image_paths) > 1:
            self.load_contours(str(np.random.choice(self.image_paths)))

        if len(self.path_points) > 0:
            self.pos_x = float(self.path_points[0,0]) + np.random.uniform(-0.05,0.05)
            self.pos_y = float(self.path_points[0,1]) + np.random.uniform(-0.05,0.05)
        else:
            self.pos_x = self.pos_y = 0.0

        self.vel_x = self.vel_y = 0.0
        self.step_count = 0
        self.waypoint_idx = 0 
        self.covered_points.clear()
        self.strict_covered_points.clear()
        if len(self.path_points) > 0:
            self.covered_points.add(0)
            self.strict_covered_points.add(0)
        self.no_progress_steps = 0
        self.hit_boundary = False
        self.prev_action  = np.zeros(2, np.float32)
        self.last_reward_components = self._empty_reward_components()

        obs = self._observe()
        self.prev_dist = self._dist_to_waypoint()
        return obs, {}

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if np.isscalar(action) or (hasattr(action, "__len__") and len(action) == 1):
            action_arr = np.array(
                {0:(0,1),1:(0,-1),2:(1,0),3:(-1,0),4:(0,0)}.get(int(action),(0,0)),
                dtype=np.float32,
            )
        else:
            action_arr = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        self._physics(action_arr)
        self.step_count += 1

        curr_dist = self._dist_to_waypoint()
        old_wp = self.waypoint_idx

        best_wp = old_wp
        upper = min(len(self.path_points) - 1, old_wp + 4)
        for candidate in range(old_wp, upper + 1):
            if candidate in self.path_breaks:
                continue
            d = float(np.hypot(
                self.path_points[candidate, 0] - self.pos_x,
                self.path_points[candidate, 1] - self.pos_y,
            ))
            if d <= self.COV_RADIUS:
                best_wp = candidate

        self.waypoint_idx = max(old_wp, best_wp)
        advanced_count = self.waypoint_idx - old_wp
        if advanced_count > 0:
            self.covered_points.update(range(old_wp + 1, self.waypoint_idx + 1))
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1

        nearest_idx, nearest_dist = self._local_path_info()
        if nearest_dist <= self.STRICT_RADIUS:
            self.strict_covered_points.add(int(nearest_idx))

        reward, components = self._reward(
            action_arr, curr_dist, advanced_count, nearest_idx, nearest_dist
        )

        coverage = self.waypoint_idx / max(1, len(self.path_points) - 1)
        strict_coverage = len(self.strict_covered_points) / max(1, len(self.path_points))
        terminated = bool(coverage >= 0.99 and self.step_count > 100)
        stalled = self.no_progress_steps >= 400
        truncated = bool((self.step_count >= self.MAX_STEPS or stalled) and not terminated)

        if terminated:
            ep_r = self.R_COMPLETION
            reward += ep_r
            components["completion"] = ep_r
            components["total"] = reward
        elif truncated:
            if coverage >= 0.80:
                ep_r = self.R_PARTIAL80
                components["partial80"] = ep_r
            else:
                ep_r = self.R_INCOMPLETE
                components["incomplete"] = ep_r
            reward += ep_r
            components["total"] = reward

        self.prev_dist = self._dist_to_waypoint()
        self.prev_action = action_arr

        info = {
            "coverage": float(coverage),
            "strict_coverage": float(strict_coverage),
            "tracking_error": float(nearest_dist),
            "on_canvas": not self.hit_boundary,
            "distance": float(self.prev_dist),
            "waypoint_idx": int(self.waypoint_idx),
            "advanced_count": int(advanced_count),
            "reward_components": components,
            "success": bool(coverage >= 0.80),
            "stalled": bool(stalled),
            "no_progress_steps": int(self.no_progress_steps),
        }
        return self._observe(), float(reward), terminated, truncated, info

    def _observe(self) -> np.ndarray:
        if len(self.path_points) == 0:
            return np.zeros(16, np.float32)

        N  = self.ARENA
        wi = min(self.waypoint_idx, len(self.path_points)-1)

        wx = (self.path_points[wi,0] - self.pos_x) / N
        wy = (self.path_points[wi,1] - self.pos_y) / N

        wi2= min(wi+1, len(self.path_points)-1)
        wx2= (self.path_points[wi2,0] - self.pos_x) / N
        wy2= (self.path_points[wi2,1] - self.pos_y) / N

        wi3= min(wi+5, len(self.path_points)-1)
        lax= self.path_points[wi3,0] - self.pos_x
        lay= self.path_points[wi3,1] - self.pos_y
        ln = max(1e-6, np.hypot(lax,lay))

        ni, d_near = self._local_path_info()
        on_curve = 1.0 if d_near < self.COV_RADIUS else 0.0

        if 1 <= ni <= len(self.path_points)-2:
            v1=self.path_points[ni]-self.path_points[ni-1]
            v2=self.path_points[ni+1]-self.path_points[ni]
            curv=float(v1[0]*v2[1] - v1[1]*v2[0])/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8)
        else:
            curv=0.0

        spd = np.hypot(self.vel_x, self.vel_y)
        return np.array([
            wx, wy,                                
            wx2, wy2,                              
            self.pos_x/N, self.pos_y/N,            
            self.vel_x/self.MAX_SPEED,             
            self.vel_y/self.MAX_SPEED,             
            float(np.clip(d_near/N, 0.0, 1.0)),    
            on_curve,                              
            float(np.clip(self.vel_y/(spd+1e-8),-1,1)), 
            float(np.clip(self.vel_x/(spd+1e-8),-1,1)), 
            float(np.clip(curv,-1,1)),             
            lax/ln, lay/ln,                        
            float(self.waypoint_idx / max(1, len(self.path_points)-1)),  
        ], np.float32)

    def _dist_to_waypoint(self) -> float:
        if len(self.path_points)==0: return 0.0
        wi=min(self.waypoint_idx,len(self.path_points)-1)
        return float(np.hypot(self.path_points[wi,0]-self.pos_x,
                               self.path_points[wi,1]-self.pos_y))

    def _local_path_info(self, window: int = 20):
        if len(self.path_points) == 0:
            return -1, self.ARENA * 2.0
        lo = max(0, self.waypoint_idx - window)
        hi = min(len(self.path_points), self.waypoint_idx + window + 1)
        pts = self.path_points[lo:hi]
        d = np.hypot(pts[:, 0] - self.pos_x, pts[:, 1] - self.pos_y)
        j = int(np.argmin(d))
        return lo + j, float(d[j])

    def _reward(self, action, curr_dist, advanced_count=0,
                nearest_idx=None, nearest_dist=None):
        c = self._empty_reward_components()
        if len(self.path_points) == 0:
            return 0.0, c

        # 1. Approach potential toward current target waypoint
        delta_norm = (self.prev_dist - curr_dist) / max(self.COV_RADIUS, 1e-6)
        c["approach"] = self.R_APPROACH * float(np.clip(delta_norm, -1.0, 1.0))

        # 2. Discrete progress reward for newly covered waypoints
        c["waypoint"] = self.R_WAYPOINT * float(max(0, advanced_count))

        if nearest_idx is None or nearest_dist is None:
            nearest_idx, nearest_dist = self._local_path_info()

        # 3. High-precision centerline bonus: exp(-(e_lat / 0.08)^2)
        precision_bonus = float(np.exp(- (nearest_dist / 0.08) ** 2))
        c["precision"] = self.R_PRECISION * precision_bonus

        # 4. Tangent velocity alignment (forward momentum along track contour)
        if 0 <= nearest_idx < len(self.path_points) - 1:
            t_vec = self.path_points[nearest_idx + 1] - self.path_points[nearest_idx]
        elif len(self.path_points) > 1:
            t_vec = self.path_points[-1] - self.path_points[-2]
        else:
            t_vec = np.array([1.0, 0.0])
        t_norm = np.linalg.norm(t_vec)
        if t_norm > 1e-6:
            t_unit = t_vec / t_norm
            vel_proj = (self.vel_x * t_unit[0] + self.vel_y * t_unit[1]) / self.MAX_SPEED
            c["tangent"] = self.R_TANGENT * float(np.clip(vel_proj, -1.0, 1.0))

        # 5. Tracking error penalty
        err_norm = float(np.clip(nearest_dist / max(self.STRICT_RADIUS, 1e-6), 0.0, 3.0))
        c["tracking_error"] = self.R_TRACK_ERR * err_norm

        # 6. Off curve / backtrack / boundary penalties
        if nearest_dist > self.COV_RADIUS:
            c["off_curve"] = self.R_OFF_CURVE
        if nearest_idx < self.waypoint_idx - 5 and nearest_dist <= self.COV_RADIUS:
            c["backtrack"] = self.R_BACKTRACK
        if self.hit_boundary:
            c["boundary"] = self.R_BOUNDARY

        # 7. Action smoothness & step cost
        jerk = float(np.linalg.norm(action - self.prev_action))
        c["smooth"] = self.R_SMOOTH * min(jerk, 1.0)
        c["step_cost"] = self.R_STEP

        total = float(sum(c.values()))
        c["total"] = total
        self.last_reward_components = c
        return total, c

    def _physics(self, action: np.ndarray):
        ax, ay = float(action[0])*self.ACCEL, float(action[1])*self.ACCEL
        self.hit_boundary = False

        self.vel_x = (self.vel_x + ax*self.DT) * (1 - self.FRICTION*self.DT)
        self.vel_y = (self.vel_y + ay*self.DT) * (1 - self.FRICTION*self.DT)

        spd = np.hypot(self.vel_x, self.vel_y)
        if spd > self.MAX_SPEED:
            self.vel_x *= self.MAX_SPEED/spd
            self.vel_y *= self.MAX_SPEED/spd

        self.pos_x += self.vel_x*self.DT
        self.pos_y += self.vel_y*self.DT

        for attr,v_attr in (("pos_x","vel_x"),("pos_y","vel_y")):
            p=getattr(self,attr); v=getattr(self,v_attr)
            if p>self.BOUNDARY:
                setattr(self,attr,self.BOUNDARY); setattr(self,v_attr,-abs(v)*self.BOUNCE); self.hit_boundary=True
            elif p<-self.BOUNDARY:
                setattr(self,attr,-self.BOUNDARY); setattr(self,v_attr,abs(v)*self.BOUNCE);  self.hit_boundary=True

    def get_coverage(self): return self.waypoint_idx / max(1, len(self.path_points) - 1)

    @staticmethod
    def _empty_reward_components():
        return {k:0.0 for k in
                ["approach","waypoint","precision","tangent","tracking_error",
                 "off_curve","backtrack","boundary","smooth","step_cost",
                 "completion","partial80","incomplete","total"]}

    def _get_episode_reward(self):
        cov=self.get_coverage(); suc=cov>=0.80
        if cov >= 0.99:
            return self.R_COMPLETION, True
        if cov >= 0.80:
            return self.R_PARTIAL80, True
        return self.R_INCOMPLETE, False

    def _find_closest_path_point(self):
        if len(self.path_points)==0: return 0.,0.,self.ARENA*2,-1
        d=np.hypot(self.path_points[:,0]-self.pos_x,self.path_points[:,1]-self.pos_y)
        ni=int(np.argmin(d))
        return float(self.path_points[ni,0]),float(self.path_points[ni,1]),float(d[ni]),ni

    def calculate_trace_reward(self,prev_pos,curr_pos,path_index):
        self.pos_x,self.pos_y=curr_pos
        d=self._dist_to_waypoint()
        r,_=self._reward(self.prev_action,d)
        self.prev_dist=d
        return r
