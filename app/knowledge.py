"""Knowledge bases and hybrid retrieval (FR-601-lite/602/603/605/607).

Documents are chunked with overlap; each chunk gets a deterministic hashed
bag-of-words embedding (works fully offline, swappable for a real embedding
model). Retrieval fuses vector cosine ranking with BM25 keyword ranking via
Reciprocal Rank Fusion, and returns citations (document name + chunk id).
All queries are scoped by tenant_id — a tenant can never retrieve another
tenant's chunks.
"""
import hashlib
import math
import re
from collections import Counter

from . import db

EMBED_DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "do", "does",
    "did", "have", "has", "had", "how", "what", "when", "where", "who", "why",
    "we", "i", "you", "he", "she", "it", "they", "to", "of", "in", "on", "at",
    "for", "with", "by", "from", "and", "or", "not", "this", "that", "there",
    "their", "our", "your", "his", "her", "its", "as", "if", "then", "than",
    "many", "much", "can", "could", "should", "would", "will", "shall",
}


def _stem(tok: str) -> str:
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower())
            if t not in _STOPWORDS]


def embed(text: str) -> list[float]:
    vec = [0.0] * EMBED_DIM
    for tok in tokenize(text):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % EMBED_DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def chunk_text(text: str, size: int = 500, overlap: int = 80) -> list[str]:
    """Paragraph-aware chunking with a sliding-window fallback."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 1 <= size:
            buf = f"{buf}\n{p}" if buf else p
        else:
            if buf:
                chunks.append(buf)
            if len(p) > size:
                step = max(1, size - overlap)
                for i in range(0, len(p), step):
                    chunks.append(p[i:i + size])
                buf = ""
            else:
                buf = p
    if buf:
        chunks.append(buf)
    return chunks or ([text] if text.strip() else [])


# ------------------------------------------------------------------- CRUD --

def create_kb(tenant_id: str, name: str) -> dict:
    kb_id = db.new_id("kb")
    db.x("INSERT INTO knowledge_bases (id, tenant_id, name, created_at)"
         " VALUES (?,?,?,?)", (kb_id, tenant_id, name, db.now_ts()))
    return {"id": kb_id, "tenant_id": tenant_id, "name": name}


def get_kb(tenant_id: str, kb_id: str) -> dict | None:
    row = db.q1("SELECT * FROM knowledge_bases WHERE id=? AND tenant_id=?",
                (kb_id, tenant_id))
    return dict(row) if row else None


def find_kb_by_name(tenant_id: str, name: str) -> dict | None:
    row = db.q1("SELECT * FROM knowledge_bases WHERE tenant_id=? AND name=?",
                (tenant_id, name))
    return dict(row) if row else None


def add_document(tenant_id: str, kb_id: str, name: str, content: str) -> dict:
    if get_kb(tenant_id, kb_id) is None:
        raise ValueError(f"knowledge base '{kb_id}' not found for tenant")
    doc_id = db.new_id("doc")
    db.x("INSERT INTO documents (id, kb_id, tenant_id, name, content_hash, created_at)"
         " VALUES (?,?,?,?,?,?)",
         (doc_id, kb_id, tenant_id, name,
          hashlib.sha256(content.encode()).hexdigest(), db.now_ts()))
    parts = chunk_text(content)
    for i, part in enumerate(parts):
        db.x("INSERT INTO chunks (id, doc_id, kb_id, tenant_id, seq, content, vector)"
             " VALUES (?,?,?,?,?,?,?)",
             (db.new_id("chk"), doc_id, kb_id, tenant_id, i, part,
              db.jdump(embed(part))))
    return {"id": doc_id, "kb_id": kb_id, "name": name, "chunks": len(parts)}


# --------------------------------------------------------------- retrieval --

def _bm25_scores(query_tokens: list[str], corpus: list[dict]) -> dict[str, float]:
    """Classic BM25 over the KB chunk corpus."""
    n = len(corpus)
    if n == 0:
        return {}
    doc_freq: Counter = Counter()
    counters: list[Counter] = []
    lengths: list[int] = []
    for c in corpus:
        toks = tokenize(c["content"])
        counters.append(Counter(toks))
        lengths.append(len(toks))
        for t in set(toks):
            doc_freq[t] += 1
    avgdl = (sum(lengths) / n) if n else 1.0
    k1, b = 1.5, 0.75
    scores: dict[str, float] = {}
    for c, counter, dl in zip(corpus, counters, lengths):
        score = 0.0
        for t in set(query_tokens):
            if t not in counter:
                continue
            idf = math.log(1 + (n - doc_freq[t] + 0.5) / (doc_freq[t] + 0.5))
            tf = counter[t]
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
        if score > 0:
            scores[c["id"]] = score
    return scores


def search(tenant_id: str, kb_ids: list[str], query: str, top_k: int = 5,
           min_vector_score: float = 0.0) -> list[dict]:
    """Hybrid retrieval with RRF fusion. kb_ids may be ids or names.
    Chunks whose vector cosine is below min_vector_score are dropped before
    ranking, so off-topic queries can legitimately return nothing."""
    resolved: list[str] = []
    for kid in kb_ids:
        if get_kb(tenant_id, kid):
            resolved.append(kid)
        else:
            by_name = find_kb_by_name(tenant_id, kid)
            if by_name:
                resolved.append(by_name["id"])
    if not resolved or not query.strip():
        return []
    placeholders = ",".join("?" * len(resolved))
    rows = db.q(
        f"SELECT c.*, d.name AS doc_name FROM chunks c"
        f" JOIN documents d ON d.id = c.doc_id"
        f" WHERE c.tenant_id=? AND c.kb_id IN ({placeholders})",
        (tenant_id, *resolved))
    corpus = [dict(r) for r in rows]
    if not corpus:
        return []

    qvec = embed(query)
    raw_scores = {c["id"]: cosine(qvec, db.jload(c["vector"], [])) for c in corpus}
    if min_vector_score > 0:
        corpus = [c for c in corpus if raw_scores.get(c["id"], 0) >= min_vector_score]
        if not corpus:
            return []
    vec_scores = {c["id"]: raw_scores[c["id"]] for c in corpus}
    bm25 = _bm25_scores(tokenize(query), corpus)

    def ranks(scores: dict[str, float]) -> dict[str, int]:
        ordered = sorted(scores.items(), key=lambda kv: -kv[1])
        return {cid: i + 1 for i, (cid, s) in enumerate(ordered) if s > 0}

    vec_ranks, bm_ranks = ranks(vec_scores), ranks(bm25)
    K = 60  # RRF constant
    fused: dict[str, float] = {}
    for cid in set(vec_ranks) | set(bm_ranks):
        fused[cid] = (1 / (K + vec_ranks[cid]) if cid in vec_ranks else 0) + \
                     (1 / (K + bm_ranks[cid]) if cid in bm_ranks else 0)
    by_chunk = {c["id"]: c for c in corpus}
    out = []
    for cid, score in sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]:
        c = by_chunk[cid]
        out.append({
            "chunk_id": cid, "doc_id": c["doc_id"], "doc_name": c["doc_name"],
            "seq": c["seq"], "content": c["content"],
            "score": round(score, 6),
            "citation": f"[{c['doc_name']}#{c['seq']}]",
        })
    return out
