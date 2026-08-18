import os
import random
import subprocess
import csv
import re
from typing import Optional
import click
import dnnlib
import numpy as np
import torch
import PIL.Image
import cv2
from tqdm import tqdm
from scipy.interpolate import Rbf
from scipy.ndimage import map_coordinates

import legacy

def num_range(s: str):
    """Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints."""
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(',')
    return [int(x) for x in vals if x.strip() != '']

# Multiple Impression Generation

def apply_random_brightness_darkness(img, alpha=1.0, beta=0):
    _, mask = cv2.threshold(img, 180, 1, cv2.THRESH_BINARY_INV)
    mask_bool = mask.astype(bool)
    img_float = img.astype(np.float32)
    img_float[mask_bool] *= alpha
    img_float[mask_bool] += beta
    np.clip(img_float, 0, 255, out=img_float)
    img_adjusted = img_float.astype(np.uint8)
    img_adjusted[~mask_bool] = 255
    return img_adjusted

def apply_nonlinear_deformation(img, num_points=10, deformation_scale=0.01):
    rows, cols = img.shape
    grid_x, grid_y = np.mgrid[0:cols, 0:rows]
    points_x = np.random.rand(num_points) * cols
    points_y = np.random.rand(num_points) * rows
    dx = np.random.randn(num_points) * cols * deformation_scale
    dy = np.random.randn(num_points) * rows * deformation_scale
    rbf_x = Rbf(points_x, points_y, dx, function='multiquadric', smooth=0.1)
    rbf_y = Rbf(points_x, points_y, dy, function='multiquadric', smooth=0.1)
    return map_coordinates(img, [grid_y + rbf_y(grid_x, grid_y), grid_x + rbf_x(grid_x, grid_y)], order=1, mode='reflect')

def apply_transformations(img, max_trans=10, max_angle=30):
    tx, ty = np.random.randint(-max_trans, max_trans, 2)
    angle = np.random.uniform(-max_angle, max_angle)
    center = (img.shape[1] // 2, img.shape[0] // 2)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    mat[0, 2] += tx
    mat[1, 2] += ty
    return cv2.warpAffine(img, mat, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=255)

def straighten_image(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

def generate_impressions(img, num_variations=5):
    return [
        straighten_image(
            apply_random_brightness_darkness(
                apply_nonlinear_deformation(
                    apply_transformations(img)
                ),
                alpha=random.uniform(0.7, 1.3),
                beta=random.randint(-30, 30)
            )
        )
        for _ in range(num_variations)
    ]

@click.command()
@click.option('--network', 'network_pkl', required=True, help='Network pickle filename')
@click.option('--outdir', required=True, type=str, help='Output directory for images')
@click.option('--class', 'class_idx', type=int, help='Class label for conditional model')
@click.option('--num-images', type=int, default=50, help='Number of unique fingerprint to generate')
@click.option('--num-impressions', type=int, default=3, help='Number of impressions per image')
@click.option('--name', type=str, required=False, help='Spoof type name (e.g., gel, sil)')
@click.option('--noise-mode', type=click.Choice(['const', 'random', 'none']), default='random', show_default=True)
@click.option('--seeds', type=num_range, help="List of random seeds:'a,b,c' or 'a-c'")
@click.option('--trunc', 'truncation_psi', type=float, default=1.0, show_default=True, help='Truncation psi value')
def generate_images(network_pkl, outdir, class_idx, num_images, num_impressions, name, noise_mode, seeds, truncation_psi):
    print(f'Loading network from: {network_pkl}')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    os.makedirs(outdir, exist_ok=True)

    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('--class is required for conditional models.')
        label[:, class_idx] = 1

    csv_path = os.path.join(outdir, "metadata.csv")
    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_name", "seed", "impression_index", "truncation_psi", "noise_mode"])

        generated = 0
        used_seeds = set()
        seed_iter = iter(seeds) if seeds is not None else None

        with tqdm(total=num_images, desc="Accepted Images") as pbar:
            while generated < num_images:
                if seed_iter is not None:
                    try:
                        seed = next(seed_iter)
                    except StopIteration:
                        print("Ran out of provided seeds before reaching num-images; stopping.")
                        break
                    if seed in used_seeds:
                        # If a duplicate is in the provided list, skip it.
                        continue
                else:
                    seed = random.randint(0, 100000)
                    if seed in used_seeds:
                        continue
                used_seeds.add(seed)

                rnd = np.random.RandomState(seed)
                z = rnd.randn(1, G.z_dim)
                z_tensor = torch.from_numpy(z).to(device)

                img = G(z_tensor, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                base_img = img[0].cpu().numpy()
                base_gray = cv2.cvtColor(base_img, cv2.COLOR_RGB2GRAY)
                impressions = generate_impressions(base_gray, num_impressions)
                for idx, imp in enumerate(impressions):
                    filename = f'seed{generated:04d}_{idx+1}.png'
                    imp_rgb = cv2.cvtColor(imp, cv2.COLOR_GRAY2RGB)
                    imp_path = os.path.join(outdir, filename)
                    cv2.imwrite(imp_path, imp_rgb)
                    writer.writerow([filename, seed, idx+1, f"{truncation_psi:.3f}", noise_mode])

                generated += 1
                pbar.update(1)

    if name and name.lower() != "live":
        print(f'Running test.py with name: {name}')
        subprocess.run(["python3", "test.py", "--dataroot", outdir, "--name", name, "--model", "test", "--crop_size", "512","--load_size", "512", "--no_dropout", "--results_dir", outdir])
        for file in os.listdir(outdir):
            if file.endswith('.png') and file.startswith('seed'):
                try:
                    os.remove(os.path.join(outdir, file))
                except Exception as e:
                    print(f"Could not remove file {file}: {e}")

if __name__ == "__main__":
    generate_images()

