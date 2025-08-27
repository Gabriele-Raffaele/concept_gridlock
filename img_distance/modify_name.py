import os
folder = "/Users/gabriele/Desktop/Magistrale/Explainable_and_trustworthy_AI/progetti/concept_gridlock/kaggle/working/road-save/comma/cache/resnet50I3D512-Pkinetics-b4s8x1x1-commat3-h3x3x3/batch_concepts-30-08"

# Controllo cartella
if not os.path.exists(folder):
    print(f"❌ La cartella '{folder}' non esiste!")
else:
    print(f"📂 Scansiono cartella: {folder}")

    for filename in os.listdir(folder):
        print(f"🔎 Trovato file: {filename}")  # debug

        if "_240frames" in filename:
            new_name = filename.replace("_240frames", "")
            old_path = os.path.join(folder, filename)
            new_path = os.path.join(folder, new_name)

            os.rename(old_path, new_path)
            print(f"✅ Rinomimato: {filename} → {new_name}")