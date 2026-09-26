import sys
from pathlib import Path

_this_dir = Path(__file__).resolve().parent
_root_dir = _this_dir.parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))
if str(_root_dir / "research_paper_code") not in sys.path:
    sys.path.insert(0, str(_root_dir / "research_paper_code"))

import benchmark_paper_controllers

def resolve_path(p, subfolders=("results/models", "datasets/test_images", "datasets/closed_shapes_dataset/test")):
    if not p:
        return p
    for candidate in [p, str(_root_dir / p), *[str(_root_dir / folder / Path(p).name) for folder in subfolders], *[f"{folder}/{Path(p).name}" for folder in subfolders]]:
        if Path(candidate).exists():
            return str(candidate)
    return p

test_images = [
    "path1.jpeg",
    "test_shape.jpg",
    "circle.jpg",
    "triangle.png",
    "test_shape_hollow_star.jpg",
    "closed_shapes_dataset/test/shape_test_0000.jpg",
    "closed_shapes_dataset/test/shape_test_0001.jpg",
    "closed_shapes_dataset/test/shape_test_0002.jpg",
]

print("=" * 80)
print("ZERO-SHOT GENERALIZATION EVALUATION (Section IV.B of Research Paper)")
print("Model: RL_PP_JOINT_MODEL.zip (Trained with PPO on path1.jpeg, tested on UNSEEN shapes)")
print("=" * 80)
print(f"{'Image Path':<38} | {'Sequential':<12} | {'Geom@0.10':<12} | {'RMSE':<10} | {'Steps'}")
print("-" * 80)

for img_raw in test_images:
    img = resolve_path(img_raw)
    if not Path(img).exists():
        continue
    try:
        model_p = resolve_path("RL_PP_JOINT_MODEL.zip")
        res = benchmark_paper_controllers.run_controller_episodes(
            "joint", img, model_path=model_p, episodes=1
        )
        r = res[0]
        print(f"{Path(img).name:<38} | {r['sequential']:<12.2%} | {r['geom010']:<12.2%} | {r['rmse']:<10.4f} | {r['steps']}")
    except Exception as e:
        print(f"{Path(img).name:<38} | Error: {e}")
print("-" * 80)
