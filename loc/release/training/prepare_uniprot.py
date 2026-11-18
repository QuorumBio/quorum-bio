"""Build a homology-partitioned subcellular-localization + signal-peptide dataset from UniProtKB/Swiss-Prot (CC BY 4.0).
Steps: download reviewed entries (TSV) -> keep experimentally supported locations (ECO:0000269) mapped to 10 classes -> mmseqs2 clustering at 30% identity ->
cluster-level 80/10/10 split -> remove val/test sequences with any >=30% identity hit to train -> write jsonl + provenance.json.
Windows/Linux, needs internet (UniProt REST, GitHub release of MMseqs2)."""
import gzip, hashlib, io, json, os, random, re, subprocess, sys, urllib.request, zipfile
from pathlib import Path
root = Path(__file__).resolve().parents[1]; raw = root / "data" / "raw"; out = root / "data"; raw.mkdir(parents=True, exist_ok=True)
URL = ("https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=tsv&query=%28reviewed%3Atrue%29"
       "&fields=accession%2Csequence%2Clength%2Cft_signal%2Ccc_subcellular_location%2Corganism_id")
tsv = raw / "sprot.tsv.gz"; prov = {"sources": {}}
if not tsv.exists():
    req = urllib.request.Request(URL, headers={"User-Agent": "quorumbio-research/0.1"})
    with urllib.request.urlopen(req) as r, open(tsv, "wb") as f:
        prov["sources"]["uniprot_release"] = r.headers.get("X-UniProt-Release"); prov["sources"]["uniprot_release_date"] = r.headers.get("X-UniProt-Release-Date")
        while True:
            b = r.read(1 << 20)
            if not b: break
            f.write(b)
prov["sources"]["uniprot_url"] = URL; prov["sources"]["uniprot_license"] = "CC BY 4.0"
prov["sources"]["sprot_tsv_sha256"] = hashlib.sha256(tsv.read_bytes()).hexdigest()
CLASSES = ["Nucleus", "Cytoplasm", "Extracellular", "Mitochondrion", "Cell membrane", "Endoplasmic reticulum", "Plastid", "Golgi apparatus", "Lysosome/Vacuole", "Peroxisome"]
RULES = [("Nucleus", r"^nucle(us|oplasm)\b"), ("Cytoplasm", r"^cytoplasm\b|^cytosol"), ("Extracellular", r"^secreted"),
         ("Mitochondrion", r"^mitochondri"), ("Cell membrane", r"^cell membrane"), ("Endoplasmic reticulum", r"^endoplasmic reticulum"),
         ("Plastid", r"^(plastid|chloroplast)"), ("Golgi apparatus", r"^golgi"), ("Lysosome/Vacuole", r"^(lysosom|vacuol)"), ("Peroxisome", r"^peroxisom")]
ENT = re.compile(r"([A-Za-z][^.;{}]*?)\s*\{([^}]*)\}")
def locs(text):
    found = set()
    if not text: return []
    body = text.replace("SUBCELLULAR LOCATION:", "")
    body = re.sub(r"\[[^\]]*\]:", "", body)  # drop isoform headers
    for name, ev in ENT.findall(body):
        if "ECO:0000269" not in ev: continue
        n = name.strip().lstrip(",").strip().lower()
        n = re.sub(r"^note=.*", "", n)
        for c, pat in RULES:
            if re.search(pat, n): found.add(c)
    return [c for c in CLASSES if c in found]
recs = []; n_all = 0
with gzip.open(tsv, "rt", encoding="utf-8") as f:
    header = f.readline().rstrip("\n").split("\t")
    for line in f:
        n_all += 1
        p = line.rstrip("\n").split("\t")
        if len(p) < 6: p += [""] * (6 - len(p))
        acc, seq, length, sig, loc, org = p[:6]
        L = len(seq)
        if not (30 <= L <= 4000) or not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", seq): continue
        lab = locs(loc)
        if not lab: continue
        m = re.search(r"SIGNAL\s+(\d+)\.\.(\d+)", sig or "")
        recs.append({"id": acc, "sequence": seq, "locations": lab, "signal_peptide": bool(m), "sp_end": int(m.group(2)) if m else None, "organism_id": org})
prov["counts"] = {"swissprot_entries": n_all, "kept_with_experimental_location": len(recs)}
print(prov["counts"], flush=True)
# MMseqs2
mm_dir = root / "tools"; mm_dir.mkdir(exist_ok=True)
exe = None
for p in mm_dir.rglob("mmseqs*"):
    if p.is_file() and p.name in ("mmseqs", "mmseqs.exe"): exe = p; break
if exe is None:
    u = "https://github.com/soedinglab/MMseqs2/releases/latest/download/mmseqs-win64.zip" if os.name == "nt" else "https://github.com/soedinglab/MMseqs2/releases/latest/download/mmseqs-linux-avx2.tar.gz"
    z = mm_dir / "mmseqs_dl"; urllib.request.urlretrieve(u, z)
    prov["sources"]["mmseqs_url"] = u; prov["sources"]["mmseqs_sha256"] = hashlib.sha256(z.read_bytes()).hexdigest()
    zipfile.ZipFile(z).extractall(mm_dir)
    exe = next(p for p in mm_dir.rglob("mmseqs*") if p.is_file() and p.name in ("mmseqs", "mmseqs.exe"))
print("mmseqs:", exe, flush=True)
if os.name == "nt":  # the MMseqs2 workflows are shell scripts; the Windows build needs BusyBox applets (sh, bash, awk...) installed next to mmseqs.exe
    bindir = exe.parent
    if not (bindir / "bash").exists():
        for mode in (["--install", str(bindir)], ["--install", "-s", str(bindir)]):
            subprocess.run([str(bindir / "busybox.exe")] + mode, capture_output=True)
            if (bindir / "bash").exists(): break
    print("busybox applets present:", (bindir / "bash").exists(), flush=True)
def run(args): 
    r = subprocess.run([str(exe)] + args, capture_output=True, text=True); 
    if r.returncode: print(r.stdout[-2000:], r.stderr[-2000:]); raise SystemExit("mmseqs failed")
    return r
work = root / "data" / "mm"; work.mkdir(exist_ok=True)
fa = work / "all.fasta"; fa.write_text("".join(">%s\n%s\n" % (r["id"], r["sequence"]) for r in recs))
# manual steps instead of easy-cluster (its final result2flat step segfaults in the Windows build)
run(["createdb", str(fa), str(work / "db")])
run(["cluster", str(work / "db"), str(work / "clu"), str(work / "tmp"), "--min-seq-id", "0.3", "-c", "0.8", "--cov-mode", "0", "--threads", "4"])
run(["createtsv", str(work / "db"), str(work / "db"), str(work / "clu"), str(work / "clu_cluster.tsv")])
cluster = {}
for l in open(work / "clu_cluster.tsv"):
    rep, mem = l.rstrip("\n").split("\t"); cluster[mem] = rep
reps = sorted(set(cluster.values())); random.Random(13).shuffle(reps)
n = len(reps); part = {}
for i, r in enumerate(reps): part[r] = "train" if i < 0.8 * n else ("dev" if i < 0.9 * n else "test")
split = {"train": [], "dev": [], "test": []}
for r in recs: split[part[cluster[r["id"]]]].append(r)
prov["clusters"] = n; prov["pre_filter_split"] = {k: len(v) for k, v in split.items()}
# remove dev/test sequences with >=30% identity hit to train
def fasta(path, rs): path.write_text("".join(">%s\n%s\n" % (r["id"], r["sequence"]) for r in rs))
fasta(work / "train.fasta", split["train"])
for k in ("dev", "test"):
    fasta(work / f"{k}.fasta", split[k])
    run(["createdb", str(work / f"{k}.fasta"), str(work / f"{k}_db")])
    if k == "dev": run(["createdb", str(work / "train.fasta"), str(work / "train_db")])
    run(["search", str(work / f"{k}_db"), str(work / "train_db"), str(work / f"{k}_res"), str(work / "tmp2"), "--min-seq-id", "0.3", "-c", "0.5", "--cov-mode", "2", "-s", "7.5", "--threads", "4"])
    run(["convertalis", str(work / f"{k}_db"), str(work / "train_db"), str(work / f"{k}_res"), str(work / f"{k}_hits.m8")])
    bad = {l.split("\t")[0] for l in open(work / f"{k}_hits.m8")}
    split[k] = [r for r in split[k] if r["id"] not in bad]
    prov[f"removed_{k}_similar_to_train"] = len(bad)
prov["final_split"] = {k: len(v) for k, v in split.items()}
prov["class_counts"] = {k: {c: sum(c in r["locations"] for r in v) for c in CLASSES} for k, v in split.items()}
prov["signal_peptide_positive"] = {k: sum(r["signal_peptide"] for r in v) for k, v in split.items()}
prov["notes"] = ("Locations kept only when annotated with ECO:0000269 (experimental). Signal peptide label = UniProt SIGNAL feature of any evidence. "
                 "Splits are cluster-level at 30% identity (mmseqs2 -c 0.8), then dev/test sequences with any >=30% identity hit (coverage 0.5, sensitivity 7.5) to train were removed. "
                 "This is a homology-reduced split, not identical to the DeepLoc/SignalP partitions, so numbers are not directly comparable to published ones.")
for k, v in split.items():
    with open(out / f"{k}.jsonl", "w") as f:
        for r in v: f.write(json.dumps(r) + "\n")
(out / "provenance.json").write_text(json.dumps(prov, indent=2))
print(json.dumps(prov, indent=1)); print("PREP_LOC_DONE")
