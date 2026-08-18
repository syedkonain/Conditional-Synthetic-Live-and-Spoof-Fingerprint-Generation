import os
import random
import csv
import subprocess
from typing import Optional, Union, List
import re
import click
import dnnlib
import numpy as np
import torch
import PIL.Image
import cv2
from scipy.interpolate import Rbf
from scipy.ndimage import map_coordinates
from tqdm import tqdm
import legacy


def parse_seeds(s: Union[str, List[int]]) -> List[int]:
    # If it's already a list of ints, just return it
    if isinstance(s, list):
        return s

    ranges: List[int] = []
    range_re = re.compile(r'^(\d+)-(\d+)$')

    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        m = range_re.match(part)
        if m:
            start = int(m.group(1))
            end = int(m.group(2))
            ranges.extend(range(start, end + 1))
        else:
            # Single number
            ranges.append(int(part))

    return ranges


def make_transform(translate, angle):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

@click.command()
@click.option('--network', 'network_pkl', help='Path to StyleGAN3 network pickle file', required=True)
@click.option('--outdir', help='Output directory for images', type=str, required=True)
@click.option('--class', 'class_idx', type=int, help='Class label for conditional models')
@click.option('--num-images', type=int, default=50, help='Number of unique fingerprints to generate')
@click.option('--num-impressions', type=int, default=3, help='Number of impressions per image')
@click.option('--name', type=str, required=False, help='Spoof type name (e.g., sil, gel). Runs test.py if provided.')
@click.option('--noise-mode', type=click.Choice(['const', 'random', 'none']), default='random', show_default=True)
@click.option('--seeds', type=parse_seeds, required=False, help="Seed list or ranges (e.g. '1,5,10-20')")
@click.option('--translate', type=str, default='0,0', help='XY translation for StyleGAN3 (e.g., "0.3,0.1")')
@click.option('--rotate', type=float, default=0, help='Rotation for StyleGAN3 in degrees')
@click.option('--truncation-psi', type=float, default=1.0, show_default=True, help='Truncation psi controls tradeoff between variety and fidelity')
def generate_images(network_pkl, outdir, class_idx, num_images, num_impressions, name, noise_mode, seeds, translate, rotate, truncation_psi):
    print(f'Loading network from: {network_pkl}')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    os.makedirs(outdir, exist_ok=True)

    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify --class for conditional model.')
        label[:, class_idx] = 1

    seed_list = seeds if seeds else None
    translate = tuple(map(float, translate.split(',')))

    metadata_path = os.path.join(outdir, "metadata.csv")
    with open(metadata_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["seed", "impression_index", "filename", "truncation_psi", "noise_mode"])

        generated = 0
        used_seeds = set()

        with tqdm(total=num_images, desc="Accepted images") as pbar:
            while generated < num_images:
                if seed_list:
                    if len(seed_list) == 0:
                        break
                    seed = seed_list.pop(0)
                else:
                    seed = random.randint(0, 100000)
                if seed in used_seeds:
                    continue
                used_seeds.add(seed)

                rnd = np.random.RandomState(seed)
                z = rnd.randn(1, G.z_dim)
                z_tensor = torch.from_numpy(z).to(device)

                if hasattr(G.synthesis, 'input'):
                    m = make_transform(translate, rotate)
                    m = np.linalg.inv(m)
                    G.synthesis.input.transform.copy_(torch.from_numpy(m).to(device))

                img = G(z_tensor, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                base_img = img[0].cpu().numpy()
                base_img_gray = cv2.cvtColor(base_img, cv2.COLOR_RGB2GRAY)
                impressions = generate_impressions(base_img_gray, num_impressions)
                for j, imp in enumerate(impressions):
                    imp_rgb = cv2.cvtColor(imp, cv2.COLOR_GRAY2RGB)
                    filename = f'seed{generated:04d}_{j+1}.png'
                    imp_path = os.path.join(outdir, filename)
                    cv2.imwrite(imp_path, imp_rgb)
                    writer.writerow([seed, j + 1, filename, f"{truncation_psi:.3f}", noise_mode])

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

if __name__ == "__main__":
    generate_images()

