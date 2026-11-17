"""Download BC5CDR and NCBI-disease from official sources, convert to sentence-level word/BIO jsonl.
Writes ner/data/<corpus>/{train,dev,test}.jsonl and ner/data/provenance.json (urls, sha256, counts)."""
import io, json, re, hashlib, sys, zipfile, urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data"
SOURCES = {
    "bc5cdr": {
        "url": "https://github.com/JHnlp/BioCreative-V-CDR-Corpus/raw/master/CDR_Data.zip",
        "files": {"train": "TrainingSet.PubTator.txt", "dev": "DevelopmentSet.PubTator.txt", "test": "TestSet.PubTator.txt"},
        "license": "Public Domain Mark 1.0 (per BigBio metadata; verify)",
    },
    "ncbi_disease": {
        "urls": {
            "train": "https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBItrainset_corpus.zip",
            "dev": "https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBIdevelopset_corpus.zip",
            "test": "https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBItestset_corpus.zip",
        },
        "license": "CC0 1.0 per BigBio loader; NCBI public domain notice (verify)",
    },
}
TOKEN_RE = re.compile(r"\w+|[^\w\s]")
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")

def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "quorumbio-research/0.1"})
    data = urllib.request.urlopen(req, timeout=120).read()
    return data, hashlib.sha256(data).hexdigest()

def parse_pubtator(text):
    docs, cur = [], None
    for line in text.splitlines():
        if not line.strip():
            if cur: docs.append(cur); cur = None
            continue
        if "|t|" in line or "|a|" in line:
            pmid, kind, body = line.split("|", 2)
            cur = cur or {"pmid": pmid, "t": "", "a": "", "ents": []}
            cur[kind] = body
        else:
            f = line.split("\t")
            if cur and len(f) >= 5 and f[1].isdigit():
                label = "Chemical" if f[4].lower() == "chemical" else "Disease"
                cur["ents"].append((int(f[1]), int(f[2]), f[3], label))
    if cur: docs.append(cur)
    return docs

def to_examples(docs, stats):
    ex = []
    for d in docs:
        text = d["t"] + " " + d["a"]
        for s, e, surf, lab in d["ents"]:
            stats["ents"] += 1
            if text[s:e] != surf: stats["offset_mismatch"] += 1
        # sentence spans
        pos, spans = 0, []
        for m in SENT_RE.finditer(text):
            spans.append((pos, m.start())); pos = m.end()
        spans.append((pos, len(text)))
        for a, b in spans:
            toks = [(m.group(), a + m.start(), a + m.end()) for m in TOKEN_RE.finditer(text[a:b])]
            if not toks: continue
            tags = ["O"] * len(toks)
            for s, e, surf, lab in d["ents"]:
                first = True
                for i, (w, ts, te) in enumerate(toks):
                    if te > s and ts < e:
                        tags[i] = ("B-" if first else "I-") + lab
                        first = False
            ex.append({"pmid": d["pmid"], "text": text[a:b], "offsets": [(t[1]-a, t[2]-a) for t in toks], "tokens": [t[0] for t in toks], "tags": tags})
    return ex

def write(name, split, ex):
    p = OUT / name; p.mkdir(parents=True, exist_ok=True)
    with open(p / f"{split}.jsonl", "w", encoding="utf-8") as f:
        for e in ex: f.write(json.dumps(e) + "\n")

def main():
    prov = {}
    # BC5CDR
    s = SOURCES["bc5cdr"]
    data, sha = fetch(s["url"])
    z = zipfile.ZipFile(io.BytesIO(data))
    prov["bc5cdr"] = {"url": s["url"], "sha256": sha, "license_noted": s["license"], "splits": {}}
    for split, suffix in s["files"].items():
        name = [n for n in z.namelist() if n.endswith(suffix)][0]
        stats = {"ents": 0, "offset_mismatch": 0}
        docs = parse_pubtator(z.read(name).decode("utf-8"))
        ex = to_examples(docs, stats); write("bc5cdr", split, ex)
        prov["bc5cdr"]["splits"][split] = {"file": name, "docs": len(docs), "sentences": len(ex), **stats}
    # NCBI disease
    s = SOURCES["ncbi_disease"]
    prov["ncbi_disease"] = {"license_noted": s["license"], "splits": {}}
    for split, url in s["urls"].items():
        data, sha = fetch(url)
        z = zipfile.ZipFile(io.BytesIO(data))
        name = [n for n in z.namelist() if n.lower().endswith(".txt")][0]
        stats = {"ents": 0, "offset_mismatch": 0}
        docs = parse_pubtator(z.read(name).decode("utf-8"))
        ex = to_examples(docs, stats); write("ncbi_disease", split, ex)
        prov["ncbi_disease"]["splits"][split] = {"url": url, "sha256": sha, "file": name, "docs": len(docs), "sentences": len(ex), **stats}
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(json.dumps(prov, indent=2))

if __name__ == "__main__":
    main()
