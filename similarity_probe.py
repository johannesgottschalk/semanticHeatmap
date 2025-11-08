#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-Click: DeepSeek (Chat) -> JSON -> lokale Ähnlichkeit -> Heatmap/CSV/NPY
- Generierung: DeepSeek über OpenAI-kompatible API
- Ähnlichkeit: lokal (TF-IDF, optional SBERT)
"""

import argparse, json, os, sys, time
from datetime import datetime

# ---- Third-party ----
import numpy as np
import matplotlib.pyplot as plt

try:
    from openai import OpenAI
except ImportError:
    print("Fehlt: openai-Paket. Bitte installieren mit: pip install openai", file=sys.stderr)
    sys.exit(1)

# TF-IDF & Cosine (lokal, schnell & kostenlos)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

# Optional: SBERT für bessere Semantik (auch lokal)
_HAS_SBERT = False
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBERT = True
except Exception:
    _HAS_SBERT = False


def pinfo(msg: str):
    print(f"[info] {msg}")

def perr(msg: str):
    print(f"[error] {msg}", file=sys.stderr)

def ensure_key_and_client(provider: str, base_url: str | None):
    """
    Erstellt OpenAI-Client passend zum Provider.
    - provider 'deepseek': nutzt DEEPSEEK_API_KEY (oder OPENAI_API_KEY) + base_url (default DS)
    - provider 'openai'  : klassisch OPENAI_API_KEY + Default-Base-URL
    """
    if provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            perr("Kein DEEPSEEK_API_KEY (oder OPENAI_API_KEY) gesetzt.")
            perr("PowerShell:  $env:DEEPSEEK_API_KEY = \"sk-...\"")
            sys.exit(2)
        base = base_url or "https://api.deepseek.com"
        client = OpenAI(api_key=api_key, base_url=base)
        pinfo(f"Provider: DeepSeek | base_url={base}")
        return client
    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            perr("Kein OPENAI_API_KEY gesetzt.")
            perr("PowerShell:  $env:OPENAI_API_KEY = \"sk-...\"")
            sys.exit(2)
        client = OpenAI(api_key=api_key)  # Standard OpenAI-Base URL
        pinfo("Provider: OpenAI (Standard)")
        return client
    else:
        perr(f"Unbekannter Provider: {provider}")
        sys.exit(2)


def collect_runs(client: OpenAI, prompt: str, n: int, model: str, temperature: float, outfile: str, pause: float = 0.15):
    """
    Ruft n-mal das Chat-API auf (DeepSeek/OpenAI, identische Schnittstelle).
    Schreibt JSON. Gibt Daten als Dict zurück.
    """
    runs = []
    for i in range(n):
        pinfo(f"Run {i+1}/{n} ...")
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        text = resp.choices[0].message.content
        runs.append({
            "i": i,
            "model": model,
            "temperature": temperature,
            "created_iso": datetime.utcfromtimestamp(getattr(resp, "created", int(time.time()))).isoformat()+"Z",
            "usage": getattr(resp, "usage", None).model_dump() if getattr(resp, "usage", None) else None,
            "response_text": text
        })
        time.sleep(pause)  # Schonend bzgl. Rate Limits

    blob = {"prompt": prompt, "runs": runs}
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(blob, f, ensure_ascii=False, indent=2)
    pinfo(f"{len(runs)} Antworten -> {outfile}")
    return blob


def load_runs(infile: str):
    with open(infile, "r", encoding="utf-8") as f:
        return json.load(f)


def build_similarity_tfidf(texts: list[str]) -> np.ndarray:
    if not _HAS_SKLEARN:
        perr("scikit-learn fehlt. Installiere: pip install scikit-learn")
        sys.exit(3)
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X = vec.fit_transform(texts)      # sparse tf-idf
    S = cosine_similarity(X)          # (n x n)
    return S


def build_similarity_sbert(texts: list[str]) -> np.ndarray:
    if not _HAS_SBERT:
        perr("sentence-transformers fehlt. Installiere: pip install sentence-transformers")
        sys.exit(3)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    E = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    S = (E @ E.T).astype(float)       # Kosinus wegen Normalisierung
    return S


def save_heatmap_and_tables(S: np.ndarray, labels: list[str], title: str, heatmap_png: str, sim_csv: str, sim_npy: str):
    # Heatmap
    plt.figure(figsize=(8, 6), dpi=200)
    im = plt.imshow(S, interpolation="nearest")
    plt.title(title)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(heatmap_png)
    pinfo(f"Heatmap -> {heatmap_png}")

    # CSV
    import csv
    with open(sim_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + labels)
        for i, row in enumerate(S):
            w.writerow([labels[i]] + [f"{v:.6f}" for v in row])
    pinfo(f"CSV -> {sim_csv}")

    # NPY
    np.save(sim_npy, S)
    pinfo(f"NPY -> {sim_npy}")


def main():
    ap = argparse.ArgumentParser(description="DeepSeek Similarity Probe (Chat -> JSON -> lokale Ähnlichkeit -> Heatmap)")
    ap.add_argument("--provider", default="deepseek", choices=["deepseek", "openai"], help="API-Provider (default: deepseek)")
    ap.add_argument("--base-url", default=None, help="Custom base_url (z.B. für DeepSeek).")
    ap.add_argument("--model", default="deepseek-chat", help="Chat-Modell (DeepSeek: deepseek-chat | OpenAI: z.B. gpt-4o-mini)")
    ap.add_argument("--prompt", help="Prompt für alle Läufe")
    ap.add_argument("--n", type=int, default=10, help="Anzahl der Läufe (default: 10)")
    ap.add_argument("--temperature", type=float, default=1.0, help="Sampling-Temperatur (default: 1.0)")

    ap.add_argument("--outfile", default="runs.json", help="JSON-Ausgabe (default: runs.json)")
    ap.add_argument("--infile", default=None, help="Vorhandenes runs.json statt neuer Sammlung verwenden")
    ap.add_argument("--skip-collect", action="store_true", help="Nur aus bestehendem JSON die Heatmap bauen (kein Sammeln)")

    ap.add_argument("--sim-method", default="tfidf", choices=["tfidf", "sbert"], help="Ähnlichkeitsmethode lokal (default: tfidf)")
    ap.add_argument("--heatmap", default="similarity_heatmap.png", help="Pfad Heatmap (default: similarity_heatmap.png)")
    ap.add_argument("--sim-csv", default="similarity_matrix.csv", help="Pfad CSV (default: similarity_matrix.csv)")
    ap.add_argument("--sim-npy", default="similarity_matrix.npy", help="Pfad NPY (default: similarity_matrix.npy)")
    args = ap.parse_args()

    # Client
    client = ensure_key_and_client(args.provider, args.base_url)

    # Datenquelle
    if args.infile or args.skip_collect:
        src = args.infile if args.infile else args.outfile
        if not os.path.exists(src):
            perr(f"JSON nicht gefunden: {src}")
            sys.exit(4)
        data = load_runs(src)
    else:
        if not args.prompt:
            perr("Bitte --prompt angeben (oder --infile/--skip-collect verwenden).")
            sys.exit(5)
        data = collect_runs(client, args.prompt, args.n, args.model, args.temperature, args.outfile)

    # Texte/Labels
    texts  = [r["response_text"] for r in data.get("runs", [])]
    labels = [f"Run {r['i']}" for r in data.get("runs", [])]
    if not texts:
        perr("Keine Runs im JSON gefunden.")
        sys.exit(6)

    # Ähnlichkeit
    if args.sim_method == "tfidf":
        title = "Response Similarity (cosine, TF-IDF)"
        S = build_similarity_tfidf(texts)
    else:
        title = "Response Similarity (cosine, SBERT all-MiniLM-L6-v2)"
        S = build_similarity_sbert(texts)

    # Outputs
    save_heatmap_and_tables(S, labels, title, args.heatmap, args.sim_csv, args.sim_npy)
    pinfo("Fertig ✅")


if __name__ == "__main__":
    main()
