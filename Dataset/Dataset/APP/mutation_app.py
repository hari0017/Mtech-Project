import os

REF_FILE = "app_ref.fa"
VARIANT_FILE = "app_unhealthy_details.txt"   # change to unhealthy later
OUTPUT_DIR = "output_safe_unhealthy"

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
def apply_variant(sequence, var_type, start, end, ref, alt):

    rel_start = start - genomic_start
    rel_end = end - genomic_start

    # bounds check
    if rel_start < 0 or rel_end >= len(sequence):
        print(f"[SKIP] Variant outside reference: {start}-{end}")
        return None

    if var_type == "snv":
        if ref != "." and sequence[rel_start] != ref:
            print(f"[WARNING] REF mismatch at {start}")

        mutated = sequence[:rel_start] + alt + sequence[rel_start+1:]

    elif var_type == "del":
        print(f"[INFO] Deleting {rel_end-rel_start+1} bases at {start}-{end}")
        mutated = sequence[:rel_start] + sequence[rel_end+1:]

    elif var_type == "ins":
        mutated = sequence[:rel_start] + alt + sequence[rel_start:]

    elif var_type == "dup":
        dup_seq = sequence[rel_start:rel_end+1]
        mutated = sequence[:rel_end+1] + dup_seq + sequence[rel_end+1:]

    elif var_type == "delins":
        mutated = sequence[:rel_start] + alt + sequence[rel_end+1:]

    elif var_type == "inv":
        inv_seq = sequence[rel_start:rel_end+1][::-1]
        mutated = sequence[:rel_start] + inv_seq + sequence[rel_end+1:]

    else:
        print(f"[SKIP] Unknown variant type {var_type}")
        return None

    return mutated


# ---------- APPLY MUTATIONS ----------
with open(VARIANT_FILE) as f:
    for i, line in enumerate(f, 1):

        parts = line.strip().split(",")
        if len(parts) < 5:
            continue

        var_type, start, end, ref, alt = parts
        start = int(start)
        end = int(end)

        mutated = apply_variant(seq, var_type, start, end, ref, alt)

        if mutated is None:
            continue

        new_header = f"{header}_var{i}"
        out_path = os.path.join(OUTPUT_DIR, f"app_safe_{i}.fasta")

        with open(out_path, "w") as out:
            out.write(new_header+"\n")
            for j in range(0,len(mutated),80):
                out.write(mutated[j:j+80]+"\n")

print("\nSafe APP mutation complete")
