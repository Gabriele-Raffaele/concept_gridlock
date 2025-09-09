import os
import argparse
from PIL import Image
import numpy as np

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x 

def add_gaussian_noise_to_image_array(img_arr, sigma, seed=None):
    """
    img_arr: numpy array float32 in range [0,1], shape HxWxC
    sigma: std dev in the same scale (0..1)
    returns noisy image in [0,1] float32
    """
    if seed is not None:
        # Use a per-call RNG to avoid global state side-effects
        rng = np.random.RandomState(seed)
        noise = rng.normal(loc=0.0, scale=sigma, size=img_arr.shape).astype(np.float32)
    else:
        noise = np.random.normal(loc=0.0, scale=sigma, size=img_arr.shape).astype(np.float32)
    noisy = img_arr + noise
    noisy = np.clip(noisy, 0.0, 1.0)
    return noisy

def process_folder(input_dir, output_dir, sigmas, exts=('.jpg', '.jpeg', '.png'),
                   seed=None, overwrite=False):
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)


    exts = tuple(e.lower() for e in exts)


    files = []
    for root, _, filenames in os.walk(input_dir):
        for fn in filenames:
            if fn.lower().endswith(exts):
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, input_dir)
                files.append((full, rel))

    if not files:
        print("No one found in", input_dir, "with extensions", exts)
        return

    print(f"Found {len(files)} images. Creating output in {output_dir}")

    for sigma in sigmas:
        sigma_str = f"{sigma:.2f}".replace('.', '_')  # es. 0.03 -> "0_03"
        out_root = os.path.join(output_dir, f"gauss{sigma_str}")
        for src_full, rel in tqdm(files, desc=f"σ={sigma}"):
            dest_full = os.path.join(out_root, rel)
            dest_dir = os.path.dirname(dest_full)
            os.makedirs(dest_dir, exist_ok=True)

            if os.path.exists(dest_full) and not overwrite:
                # skip
                continue

          
            try:
                with Image.open(src_full) as im:
                    im_conv = im.convert('RGB')
                    arr = np.asarray(im_conv).astype(np.float32) / 255.0  # H,W,C in [0,1]
            except Exception as e:
                print(f"Error opening {src_full}: {e}")
                continue

            img_seed = None
            if seed is not None:
                img_seed = int((hash(rel) ^ int(seed)) & 0xffffffff)

            noisy = add_gaussian_noise_to_image_array(arr, sigma, seed=img_seed)

            noisy_u8 = (noisy * 255.0).round().astype(np.uint8)
            noisy_im = Image.fromarray(noisy_u8)

            try:
                noisy_im.save(dest_full, quality=95)
            except Exception as e:
                print(f"Error saving {dest_full}: {e}")

    print("Done. Output in:", output_dir)
    print("Folders created per sigma:", [f"gauss{f'{s:.2f}'.replace('.', '_')}" for s in sigmas])

def parse_args():
    p = argparse.ArgumentParser(description="Add Gaussian noise to frames recursively.")
    p.add_argument('--input_dir', required=True, help='Directory containing folders with frames')
    p.add_argument('--output_dir', required=True, help='Base output directory')
    p.add_argument('--sigmas', nargs='+', type=float, default=[0.03, 0.08, 0.10],
                   help='List of sigma (float) in the range [0,1], e.g. 0.03 0.08 0.10')
    p.add_argument('--exts', nargs='*', default=['.jpg', '.jpeg', '.png'],
                   help='Extensions of image files to process')
    p.add_argument('--seed', type=int, default=None, help='Seed for reproducibility (optional)')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    process_folder(args.input_dir, args.output_dir, args.sigmas, exts=args.exts,
                   seed=args.seed, overwrite=args.overwrite)
