import os

# 📂 cartella da modificare
cartella = "/Users/gabriele/Desktop/Magistrale/Explainable_and_trustworthy_AI/progetti/venv/concept_gridlock/concepts_with_noise"
count = 0
for filename in os.listdir(cartella):
    if "_240frames" in filename:
        count += 1
        nuovo_nome = filename.replace("_240frames", "")
        vecchio_path = os.path.join(cartella, filename)
        nuovo_path = os.path.join(cartella, nuovo_nome)
        os.rename(vecchio_path, nuovo_path)
        print(f"✅ Rinomato: {filename} → {nuovo_nome}")

print(f"🔍 Totale file rinominati: {count}")