import pandas as pd
from pathlib import Path

OUT_DIR = Path("data/genes/prepped")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def prep_mapping(mapping_path, yorf_path, out_name, expected_missing, expected_duplicates):
    id_mapping = pd.read_csv(mapping_path, sep="\t", header=0)  # .gz is decompressed automatically
    yorf_set = set(pd.read_csv(yorf_path, header=0, names=["Feature name"])["Feature name"].astype(str).str.strip())

    id_mapping["Gene Names"] = (
        id_mapping["Gene Names"]
        .fillna("")
        .str.split(r"[;\s]+", regex=True)
        .apply(lambda names: " ".join(n for n in names if n in yorf_set))
    )

    from_set = set(id_mapping["From"])

    # YORF ids that share the same protein
    # these will be missing from our data under the 'From' column but will exist in 'Gene Names' column (hence expected_missing)
    shared = {}
    for from_id, entry, names in zip(id_mapping["From"], id_mapping["Entry"], id_mapping["Gene Names"].str.split()):
        for gene_id in names:
            if gene_id not in from_set:
                shared.setdefault(entry, {"from_id": from_id, "gene_ids": []})["gene_ids"].append(gene_id)

    for entry, info in sorted(shared.items()):
        ids = info["gene_ids"] + [info["from_id"]]
        listed = " and ".join(ids) if len(ids) == 2 else ", ".join(ids[:-1]) + f", and {ids[-1]}"
        print(f"for protein {entry}, {listed} are listed")

    missing_from = {gene_id for info in shared.values() for gene_id in info["gene_ids"]}
    assert missing_from == expected_missing, (
        f"Found {sorted(missing_from)} != {sorted(expected_missing)}"
    )

    # YORF ids that return multiple proteins
    dup = id_mapping[id_mapping["From"].duplicated(keep=False)]
    if len(dup):
        cols = [c for c in ["From", "Entry", "Gene Names"] if c in dup.columns]
        print(f"{dup['From'].nunique()} IDs returned multiple entries:")
        print(dup[cols].to_string())
    assert set(dup["From"]) == expected_duplicates, (
        f"Found {sorted(set(dup['From']))} != {sorted(expected_duplicates)}"
    )

    # return the mapping
    out = id_mapping[["From", "Entry", "Entry Name", "Gene Names", "Sequence"]]
    out.to_csv(OUT_DIR / out_name, sep="\t", index=False)
    return out


prep_mapping(
    "data/genes/yorf_filtered_idmapping_2026_10_29.tsv.gz", # comes from uniprot using yorf_ids_filtered mapping Gene_names to Uniprot Swiss prot, should contain the columns ['From', 'Entry', 'Entry Name', 'Gene Names', 'Organism', 'Sequence']
    "data/genes/prepped/yorf_ids_filtered.csv",  # comes from scripts/prep_data.py
    "mapping_from_yorf_filtered_to_uniprot.tsv", # output file
    expected_missing={"YNL030W", "YKR059W", "YNL031C"}, # these genes encode a protein identical to a paralog, so UniProt returns one entry for the pair
    expected_duplicates={"YNL015W", "YNL067W-B"}, # these genes return X different proteins each
)