#!/usr/bin/env python3
"""
add_gaussian_noise.py

Esempio:
python add_gaussian_noise.py \
    --input_dir frames_noise \
    --output_dir frames_noise_noisy \
    --sigmas 0.03 0.08 0.10 \
    --exts .jpg .jpeg .png (OPTIONAL)\
    --seed 42 (OPTIONAL)
"""

import os
import argparse
from PIL import Image
import numpy as np

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x  # fallback semplice

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

    # Normalizza estensioni
    exts = tuple(e.lower() for e in exts)

    # raccogli tutti i file immagine
    files = []
    for root, _, filenames in os.walk(input_dir):
        for fn in filenames:
            if fn.lower().endswith(exts):
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, input_dir)
                files.append((full, rel))

    if not files:
        print("Nessun file trovato in", input_dir, "con estensioni", exts)
        return

    print(f"Trovati {len(files)} immagini. Creazione output in {output_dir}")

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

            # apri immagine
            try:
                with Image.open(src_full) as im:
                    # conserva modalità (ma convertiamo in RGB per sicurezza)
                    # se vuoi preservare canale alpha, si può aggiungere supporto
                    im_conv = im.convert('RGB')
                    arr = np.asarray(im_conv).astype(np.float32) / 255.0  # H,W,C in [0,1]
            except Exception as e:
                print(f"Errore aprendo {src_full}: {e}")
                continue

            # opzionale: deriva un seed per immagine per riproducibilità (seed + hash)
            img_seed = None
            if seed is not None:
                # mix seed con path per ottenere variabilità per immagine ma riproducibile
                img_seed = int((hash(rel) ^ int(seed)) & 0xffffffff)

            noisy = add_gaussian_noise_to_image_array(arr, sigma, seed=img_seed)

            # back to uint8
            noisy_u8 = (noisy * 255.0).round().astype(np.uint8)
            noisy_im = Image.fromarray(noisy_u8)

            # salva mantenendo estensione originale (ma salvando in RGB)
            try:
                noisy_im.save(dest_full, quality=95)
            except Exception as e:
                print(f"Errore salvando {dest_full}: {e}")

    print("Fatto. Output in:", output_dir)
    print("Cartelle create per sigma:", [f"gauss{f'{s:.2f}'.replace('.', '_')}" for s in sigmas])

def parse_args():
    p = argparse.ArgumentParser(description="Add Gaussian noise to frames recursively.")
    p.add_argument('--input_dir', required=True, help='Directory contenente cartelle con frame')
    p.add_argument('--output_dir', required=True, help='Directory base di output')
    p.add_argument('--sigmas', nargs='+', type=float, default=[0.03, 0.08, 0.10],
                   help='Lista di sigma (float) nello spazio [0,1], es. 0.03 0.08 0.10')
    p.add_argument('--exts', nargs='*', default=['.jpg', '.jpeg', '.png'],
                   help='Estensioni dei file immagine da processare')
    p.add_argument('--seed', type=int, default=None, help='Seed per riproducibilità (opzionale)')
    p.add_argument('--overwrite', action='store_true', help='Sovrascrivi file esistenti')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    process_folder(args.input_dir, args.output_dir, args.sigmas, exts=args.exts,
                   seed=args.seed, overwrite=args.overwrite)
