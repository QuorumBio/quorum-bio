"""NER v3 data: six entity types from four corpora, de-duplicated across corpora so no test abstract is seen in training.
Inputs:
  ner/data/{bc5cdr,ncbi_disease}/*.jsonl   (from prepare_data.py)
  ner/data/raw3/biored/BIORED.zip          (ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/; US Government work, public domain notice)
  ner/data/raw3/nlmgene/NLM-Gene-Corpus.zip (zenodo.org/records/5089049; CC0)
Output: ner/data3/<corpus>/{train,dev,test}.jsonl (same word/BIO format as v2, plus "corpus") and ner/data3/meta.json.

Label mapping
  BioRED: ChemicalEntity->Chemical, DiseaseOrPhenotypicFeature->Disease, GeneOrGeneProduct->Gene,
          OrganismTaxon->Species, SequenceVariant->Variant, CellLine->CellLine
  NLM-Gene: Gene, GENERIF, STARGENE -> Gene (GENERIF/STARGENE only flag importance; all are gene mentions).
            Domain, Other -> tag "X" (ignored in the loss): the guidelines say these are not to be annotated as genes
            (protein domains; gene products used as drugs or tools), so they are neither positives nor clean negatives.
Each corpus only annotates some types (TYPES below). During training, "O" in a corpus means "none of that corpus's types".

De-duplication (priority test > dev > train, by PMID, across all corpora):
  test sets are kept exactly as published;
  dev(c)   = dev(c)   minus every PMID in any test set;
  train(c) = train(c) minus every PMID in any test or dev set.
NLM-Gene has no dev split: 50 PMIDs are drawn from its (de-duplicated) training list with random.Random(13), before training."""
import hashlib, io, json, random, re, zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parents[1]
RAW = root / "data/raw3"; OUT = root / "data3"
TOKEN_RE = re.compile(r"\w+|[^\w\s]")
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")
TYPES = {"bc5cdr": ["Chemical", "Disease"], "ncbi_disease": ["Disease"], "nlmgene": ["Gene"],
         "biored": ["Chemical", "Disease", "Gene", "Species", "Variant", "CellLine"]}
BIORED_MAP = {"ChemicalEntity": "Chemical", "DiseaseOrPhenotypicFeature": "Disease", "GeneOrGeneProduct": "Gene",
              "OrganismTaxon": "Species", "SequenceVariant": "Variant", "CellLine": "CellLine"}
NLMG_MAP = {"Gene": "Gene", "GENERIF": "Gene", "STARGENE": "Gene", "Domain": "X", "Other": "X"}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def sentences(pmid, text, ents, stats):
    """ents: (start, end, label); label "X" = ignore. Longer spans win on token conflicts."""
    ents = sorted(ents, key=lambda e: (e[0], -(e[1] - e[0])))
    pos, spans, out = 0, [], []
    for m in SENT_RE.finditer(text): spans.append((pos, m.start())); pos = m.end()
    spans.append((pos, len(text)))
    for a, b in spans:
        toks = [(m.group(), a + m.start(), a + m.end()) for m in TOKEN_RE.finditer(text[a:b])]
        if not toks: continue
        tags = ["O"] * len(toks)
        for s, e, lab in ents:
            idx = [i for i, (_, ts, te) in enumerate(toks) if te > s and ts < e]
            if not idx: continue
            if any(tags[i] != "O" for i in idx): stats["token_conflicts_skipped"] += 1; continue
            for k, i in enumerate(idx): tags[i] = "X" if lab == "X" else ("B-" if k == 0 else "I-") + lab
        out.append({"pmid": pmid, "text": text[a:b], "offsets": [(t[1] - a, t[2] - a) for t in toks], "tokens": [t[0] for t in toks], "tags": tags})
    return out

def biored(prov):
    z = zipfile.ZipFile(RAW / "biored/BIORED.zip"); prov["biored"] = {"zip_sha256": sha(RAW / "biored/BIORED.zip"), "splits": {}}
    docs = {}
    for split, f in [("train", "Train"), ("dev", "Dev"), ("test", "Test")]:
        name = [n for n in z.namelist() if n.endswith(f"{f}.PubTator")][0]; cur = None; ds = []
        for line in z.read(name).decode("utf-8").splitlines() + [""]:
            if not line.strip():
                if cur: ds.append(cur); cur = None
                continue
            if "|t|" in line or "|a|" in line:
                pmid, kind, body = line.split("|", 2); cur = cur or {"pmid": pmid, "t": "", "a": "", "ents": []}; cur[kind] = body
            else:
                x = line.split("\t")
                if cur and len(x) >= 6 and x[1].isdigit(): cur["ents"].append((int(x[1]), int(x[2]), x[3], BIORED_MAP[x[4]]))
        docs[split] = ds; prov["biored"]["splits"][split] = {"file": name, "docs": len(ds)}
    return docs

def nlmgene(prov):
    zp = RAW / "nlmgene/NLM-Gene-Corpus.zip"; z = zipfile.ZipFile(zp); prov["nlmgene"] = {"zip_sha256": sha(zp), "splits": {}}
    lists = {s: [l.strip() for l in z.read(n).decode().splitlines() if l.strip()] for s, n in
             [(s, [n for n in z.namelist() if n.endswith(f"Pmidlist.{f}.txt")][0]) for s, f in [("train", "Train"), ("test", "Test")]]}
    by = {}
    for n in z.namelist():
        if not n.endswith(".BioC.XML"): continue
        r = ET.fromstring(z.read(n)); d = r.find("document"); pmid = d.findtext("id")
        parts, ents = [], []
        for p in d.findall("passage"):
            off = int(p.findtext("offset")); txt = p.findtext("text") or ""; parts.append((off, txt))
            for an in p.findall("annotation"):
                typ = [i.text for i in an.findall("infon") if i.get("key") == "type"][0]
                for loc in an.findall("location"):
                    s = int(loc.get("offset")); ents.append((s, s + int(loc.get("length")), an.findtext("text"), NLMG_MAP[typ]))
        text = ""
        for off, txt in sorted(parts):
            text += " " * (off - len(text)) + txt
        by[pmid] = {"pmid": pmid, "text": text, "ents": ents}
    return {s: [by[p] for p in v] for s, v in lists.items()}

def main():
    prov = {}; raw = {}
    for c in ["bc5cdr", "ncbi_disease"]:
        raw[c] = {s: [json.loads(l) for l in open(root / f"data/{c}/{s}.jsonl", encoding="utf-8")] for s in ["train", "dev", "test"]}
    B, N = biored(prov), nlmgene(prov)
    pm = {}
    for c in raw:
        for s, v in raw[c].items(): pm[(c, s)] = {x["pmid"] for x in v}
    for s, v in B.items(): pm[("biored", s)] = {d["pmid"] for d in v}
    for s, v in N.items(): pm[("nlmgene", s)] = {d["pmid"] for d in v}
    test_all = set().union(*[v for (c, s), v in pm.items() if s == "test"])
    dev_all = set().union(*[v for (c, s), v in pm.items() if s == "dev"])
    # NLM-Gene dev carved from its de-duplicated train list (dev must also avoid other corpora's train? no: train is filtered against dev instead)
    ng_train = [p for p in sorted(pm[("nlmgene", "train")]) if p not in test_all and p not in dev_all]
    ng_dev = set(random.Random(13).sample(ng_train, 50))
    pm[("nlmgene", "dev")] = ng_dev; pm[("nlmgene", "train")] = set(ng_train) - ng_dev
    dev_all |= ng_dev
    keep = {}
    for (c, s), v in pm.items():
        keep[(c, s)] = v if s == "test" else (v - test_all if s == "dev" else v - test_all - dev_all)
    meta = {"types": TYPES, "doc": __doc__, "provenance": prov, "splits": {}}
    stats = Counter()
    def emit(c, s, exs):
        p = OUT / c; p.mkdir(parents=True, exist_ok=True)
        with open(p / f"{s}.jsonl", "w", encoding="utf-8") as f:
            for e in exs: e["corpus"] = c; f.write(json.dumps(e) + "\n")
        tc = Counter(t[2:] for e in exs for t in e["tags"] if t.startswith("B-"))
        meta["splits"][f"{c}/{s}"] = {"docs_published": len(pm[(c, s)]) if not (c == "nlmgene" and s != "test") else None,
                                      "docs_kept": len({e["pmid"] for e in exs}), "sentences": len(exs), "entities": dict(tc),
                                      "ignored_X_tokens": sum(t == "X" for e in exs for t in e["tags"])}
    for c in raw:
        for s in ["train", "dev", "test"]:
            emit(c, s, [x for x in raw[c][s] if x["pmid"] in keep[(c, s)]])
    mm = Counter()
    for s, ds in B.items():
        ex = []
        for d in ds:
            if d["pmid"] not in keep[("biored", s)]: continue
            text = d["t"] + " " + d["a"]
            for st, en, surf, _ in d["ents"]: mm["biored_ents"] += 1; mm["biored_offset_mismatch"] += text[st:en] != surf
            ex += sentences(d["pmid"], text, [(st, en, lab) for st, en, _, lab in d["ents"]], stats)
        emit("biored", s, ex)
    allN = {d["pmid"]: d for v in N.values() for d in v}
    for s in ["train", "dev", "test"]:
        ex = []
        for p in sorted(keep[("nlmgene", s)]):
            d = allN[p]
            for st, en, surf, _ in d["ents"]: mm["nlmgene_ents"] += 1; mm["nlmgene_offset_mismatch"] += d["text"][st:en] != surf
            ex += sentences(p, d["text"], [(st, en, lab) for st, en, _, lab in d["ents"]], stats)
        emit("nlmgene", s, ex)
    meta["checks"] = {**mm, **stats}
    meta["nlmgene_dev_pmids"] = sorted(ng_dev)
    removed = {f"{c}/{s}": len(pm[(c, s)] - keep[(c, s)]) for (c, s) in pm if s != "test"}
    meta["removed_for_cross_corpus_overlap"] = removed
    # final leak check
    tr = set().union(*[keep[k] for k in keep if k[1] == "train"]); dv = set().union(*[keep[k] for k in keep if k[1] == "dev"])
    assert not (tr & test_all) and not (dv & test_all) and not (tr & dv)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k not in ("doc", "nlmgene_dev_pmids")}, indent=1))

if __name__ == "__main__":
    main()
