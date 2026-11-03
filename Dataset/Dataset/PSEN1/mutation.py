import os

REF_FILE = "psen1_ref.fa"
VARIANT_FILE = "psen1_healthy_details.txt"
OUTPUT_DIR = "output_safe_healthy"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- READ REFERENCE ----------
with open(REF_FILE) as f:
    header = f.readline().strip()
    seq = "".join(line.strip() for line in f)

# Extract genomic start
range_part = header.split(":")[1].split()[0]
genomic_start = int(range_part.split("-")[0])

print("Reference length:", len(seq))
print("Genomic start:", genomic_start)


# ---------- MUTATION FUNCTION ----------
def apply_variant(sequence, var_type, start, end, ref, alt, idx):
    rel_start = start - genomic_start
    rel_end = end - genomic_start

    if rel_start < 0 or rel_end >= len(sequence):
        print(f"[SKIP] Variant {start}-{end} outside reference")
        return None

    if var_type == "snv":
        if ref != "." and sequence[rel_start] != ref:
            print(f"[WARNING] REF mismatch at {start}")

        mutated = sequence[:rel_start] + alt + sequence[rel_start + 1 :]

    elif var_type == "del":
        print(f"[INFO] Deleting {rel_end-rel_start+1} bases at {start}-{end}")
        mutated = sequence[:rel_start] + sequence[rel_end + 1 :]

    elif var_type == "ins":
        mutated = sequence[:rel_start] + alt + sequence[rel_start:]

    elif var_type == "dup":
        dup_seq = sequence[rel_start:rel_end+1]
        mutated = sequence[:rel_end+1] + dup_seq + sequence[rel_end+1:]

    elif var_type == "delins":
        mutated = sequence[:rel_start] + alt + sequence[rel_end+1:]

    else:
        print(f"[SKIP] Unknown variant type {var_type}")
        return None

    return mutated



# ---------- PROCESS VARIANTS ----------
with open(VARIANT_FILE) as f:
    for i, line in enumerate(f, 1):
        line = line.strip()

        # skip empty lines
        if not line:
            continue

        parts = line.split(",")

        # skip malformed rows
        if len(parts) < 5:
            print("[SKIP malformed line]", line)
            continue

        var_type = parts[0]
        start = int(parts[1])
        end = int(parts[2])
        ref = parts[3]
        alt = parts[4]


        mutated_seq = apply_variant(seq, var_type, start, end, ref, alt, i)

        if mutated_seq is None:
            continue

        new_header = f"{header}_var{i}"

        output_path = os.path.join(OUTPUT_DIR, f"psen1_safe_{i}.fasta")

        with open(output_path, "w") as out:
            out.write(new_header + "\n")
            for j in range(0, len(mutated_seq), 80):
                out.write(mutated_seq[j:j+80] + "\n")

print("\nSafe mutation complete.")
