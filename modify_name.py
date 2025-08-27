import os
input_file = "/Users/gabriele/Desktop/Magistrale/Explainable_and_trustworthy_AI/progetti/concept_gridlock/kept_seqs.txt"

if not os.path.exists(input_file):
    print(f"❌ Il file '{input_file}' non esiste!")
else:
    print(f"📄 Elaboro file: {input_file}")

    with open(input_file, "r") as f:
        lines = f.readlines()

    modified_lines = []
    for line in lines:
        line_stripped = line.rstrip("\n")
        if "|" in line_stripped:
            new_line = line_stripped.replace("|", "_")
            print(f"✅ Modificata riga: {line_stripped} → {new_line}")
            modified_lines.append(new_line + "\n")
        else:
            print(f"🔎 Riga senza modifica: {line_stripped}")
            modified_lines.append(line)

    output_file = os.path.join(os.path.dirname(input_file), "kept_seqs_modified.txt")
    with open(output_file, "w") as f:
        f.writelines(modified_lines)

    print(f"📁 File modificato salvato come: {output_file}")