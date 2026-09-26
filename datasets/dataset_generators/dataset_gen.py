import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, RegularPolygon
from pathlib import Path

def create_directories(base_path):
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(Path(base_path) / split, exist_ok=True)
    return splits

def generate_closed_shape(filepath):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off') # Hide axes so the agent only sees the shape

    # Choose a random closed shape type
    shape_type = np.random.choice(['rectangle', 'ellipse', 'polygon'])
    lw = np.random.uniform(2.0, 5.0) # Randomize line thickness
    
    # Keep the shape roughly centered but with slight offsets
    center_x = np.random.uniform(0.4, 0.6)
    center_y = np.random.uniform(0.4, 0.6)

    if shape_type == 'rectangle':
        w = np.random.uniform(0.3, 0.7)
        # 30% chance to force it to be a perfect square
        h = w if np.random.rand() < 0.3 else np.random.uniform(0.3, 0.7) 
        x = center_x - w / 2
        y = center_y - h / 2
        angle = np.random.uniform(0, 360)
        patch = Rectangle((x, y), w, h, angle=angle, rotation_point='center', 
                          edgecolor='black', facecolor='none', linewidth=lw)
    
    elif shape_type == 'ellipse':
        w = np.random.uniform(0.3, 0.7)
        # 30% chance to force it to be a perfect circle
        h = w if np.random.rand() < 0.3 else np.random.uniform(0.3, 0.7)
        angle = np.random.uniform(0, 360)
        patch = Ellipse((center_x, center_y), w, h, angle=angle, 
                        edgecolor='black', facecolor='none', linewidth=lw)
        
    elif shape_type == 'polygon':
        # Generates triangles, squares, pentagons, or hexagons
        num_vertices = np.random.randint(3, 7) 
        radius = np.random.uniform(0.2, 0.4)
        angle = np.random.uniform(0, 2 * np.pi)
        patch = RegularPolygon((center_x, center_y), numVertices=num_vertices, 
                               radius=radius, orientation=angle,
                               edgecolor='black', facecolor='none', linewidth=lw)

    ax.add_patch(patch)
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def generate_dataset(base_path="closed_shapes_dataset", total_images=1000):
    splits = create_directories(base_path)
    
    # 70% Train, 15% Val, 15% Test
    train_count = int(total_images * 0.7)
    val_count = int(total_images * 0.15)
    test_count = total_images - train_count - val_count
    counts = {'train': train_count, 'val': val_count, 'test': test_count}
    
    print(f"Generating {total_images} images across {splits}...")
    
    for split in splits:
        for i in range(counts[split]):
            filename = Path(base_path) / split / f"shape_{split}_{i:04d}.jpg"
            generate_closed_shape(filename)
        print(f"Finished {split}: {counts[split]} images.")

if __name__ == "__main__":
    generate_dataset("closed_shapes_dataset", total_images=1000)