<<<<<<< HEAD
# Autonomous Racing & Bead Tracing Benchmark Repository

This repository contains clean, modular implementations of **Our High-Precision PPO Trajectory Tracing Model**, the **Research Paper Baseline**, **Datasets**, and **Results**.

---

## 📁 Repository Directory Structure

```
├── our_model_code/               # Our High-Precision PPO Multi-Map Model Pipeline
│   ├── environment.py            # Core physics, Gaussian centering reward & contour extraction
│   ├── bead_gym_env.py           # Gymnasium wrapper with multi-track support
│   ├── train_ppo_v9.py           # Multi-track PPO policy training pipeline
│   ├── evaluate_ppo_all_maps.py  # 5-episode benchmark evaluator across all 3 tracks
│   ├── visualize_all_maps.py     # Side-by-side trajectory comparison figure generator
│   ├── render_ppo_video.py       # High-resolution MP4 video renderer with HUD telemetry
│   ├── run_complete_pipeline.py  # Automated end-to-end PPO runner
│   ├── simulate_trajectory.py    # Interactive simulator & animated GIF generator
│   └── geometry_utils.py         # Point-to-polyline distance metrics
│
├── research_paper_code/          # Pure Pursuit & PPO RL tuning (Elgouhary & El-Wakeel, 2026)
│   ├── pure_pursuit.py           # Geometric Pure Pursuit, Curvature & Friction Profiler
│   ├── rl_pure_pursuit_env.py    # Gymnasium environment with 4 controller modes
│   ├── train_rl_pure_pursuit.py  # PPO training pipeline
│   ├── benchmark_paper_controllers.py # Benchmark suite replicating Tables II, III, IV
│   ├── evaluate_lap_times.py     # Real-car lap times (vmax=6m/s) evaluator
│   ├── plot_paper_interpretability.py # Interpretability telemetry generator
│   ├── render_rl_pp_video.py     # MP4 video generator
│   └── visualize_paper_tracks.py # Zero-shot track generalization visualizer
│
├── datasets/                     # Reference tracks, test images, and dataset generators
│   ├── research_paper_tracks/    # F1TENTH Hockenheim, Montreal, YasMarina centerlines & maps
│   ├── closed_shapes_dataset/    # Synthetic closed loop shapes
│   ├── curve_dataset/            # Synthetic open/closed curve tracks
│   ├── test_images/              # Racetrack PNGs, path1.jpeg, shapes
│   └── dataset_generators/       # Dataset generators
│
├── results/                      # Checkpoints, logs, figures, and video animations
│   ├── models/                   # PPO_BTP_MODEL.zip, RL_PP_JOINT_MODEL.zip
│   ├── metrics_and_logs/         # CSV evaluations & JSON summary files
│   ├── plots/                    # PNG telemetry plots, multi-map overlays
│   └── videos/                   # MP4 trace videos and GIF animations
│
├── requirements.txt              # Required dependencies
└── README.md                     # Repository documentation
```

---

## 🚀 Quick Start & Execution Commands

### 1. Research Paper Code (Elgouhary & El-Wakeel, 2026)

#### Evaluate 5 Episodes & Compare Real-Car Lap Times ($v_{\max} = 6\,\text{m/s}$):
```powershell
python research_paper_code/evaluate_lap_times.py --image datasets/test_images/path1.jpeg --model results/models/RL_PP_JOINT_MODEL.zip --episodes 5 --vmax 6.0
```

#### Run Full Benchmark (Fixed PP vs Adaptive PP vs RL-PP Ld-only vs RL-PP Joint):
```powershell
python research_paper_code/benchmark_paper_controllers.py --image datasets/test_images/path1.jpeg --model results/models/RL_PP_JOINT_MODEL.zip --episodes 5
```

#### Generate Multi-Controller Interpretability Plots (Figures 6, 7, 8):
```powershell
python research_paper_code/plot_paper_interpretability.py --image datasets/test_images/hockenheim_track.png --model results/models/RL_PP_JOINT_MODEL.zip
```

#### Render Full Dashboard MP4 Video:
```powershell
python research_paper_code/render_rl_pp_video.py --image datasets/test_images/montreal_track.png --model results/models/RL_PP_JOINT_MODEL.zip --out results/videos/montreal_trace.mp4
```

---

### 2. Our Model Code (BTP Bead Tracing SAC / DAgger)

#### Run Complete Pipeline:
```powershell
python our_model_code/run_complete_pipeline.py --image datasets/test_images/path1.jpeg --base results/models/FINAL_BTP_MODEL_95pct.zip
```

#### Evaluate Trained BTP Model:
```powershell
python our_model_code/evaluate_complete_v8.py --image datasets/test_images/path1.jpeg --model results/models/COMPLETE_BTP_MODEL.zip --episodes 5
```

---

## 📊 Summary of Benchmark Results

### Lap Time Comparison ($v_{\max} = 6.0\,\text{m/s}$)

| Controller | Mean (Current) | Mean (Paper) | Std (Current) | Std (Paper) | Min (Current) | Min (Paper) | Max (Current) | Max (Paper) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RL–PP (joint $L_d, g$)** | **8.85 s** | **9.46 s** | **0.01 s** | **0.23 s** | **8.83 s** | **9.09 s** | **8.86 s** | **9.82 s** |
| **RL–PP ($L_d$ only)** | 8.83 s | 9.61 s | 0.01 s | 0.58 s | 8.82 s | 8.94 s | 8.85 s | 10.51 s |
| **Adaptive PP (linear $v \to L_d$)** | 8.58 s | 9.72 s | 0.01 s | 0.27 s | 8.56 s | 9.34 s | 8.59 s | 10.40 s |
| **Fixed PP ($L_d$ fixed)** | 8.88 s | 9.85 s | 0.03 s | 0.43 s | 8.85 s | 9.32 s | 8.93 s | 10.55 s |
| **MPC raceline tracker** | *N/A* | **15.42 s** | *N/A* | **0.47 s** | *N/A* | **14.48 s** | *N/A* | **16.10 s** |
=======
# Bead-Simluation-using-RL-
Direct-Actuation Bead Tracing versus RL-Tuned Pure Pursuit for Contour and Racetrack Following
>>>>>>> 465f393cdba0058b79f8c468ef8f48998a32e276
