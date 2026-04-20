"""
memory/chroma_bank.py — ChromaDB-backed Error Memory Bank for TrialNet

Replaces flat JSONL with a queryable vector store.
Before answering, the chatbot retrieves the top-N most similar past mistakes
and injects them into the system prompt so the model avoids repeating them.
"""

from __future__ import annotations
import json
import os
import uuid
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_store")
COLLECTION_NAME = "trialnet_mistakes"


def _get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    ef = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


class ChromaMemoryBank:
    def __init__(self):
        self.collection = _get_collection()

    def add_mistake(self, prompt: str, bad_generation: str, human_correction: str):
        """Store one mistake-correction pair."""
        doc_id = str(uuid.uuid4())
        self.collection.add(
            ids=[doc_id],
            documents=[prompt],
            metadatas=[{
                "bad_generation": bad_generation[:500],
                "human_correction": human_correction[:500],
                "prompt": prompt[:500],
            }],
        )
        return doc_id

    def query_similar(self, prompt: str, n: int = 3) -> list[dict]:
        """Return top-N past mistakes similar to this prompt."""
        count = self.collection.count()
        if count == 0:
            return []
        n = min(n, count)
        results = self.collection.query(
            query_texts=[prompt],
            n_results=n,
            include=["metadatas", "distances"],
        )
        mistakes = []
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            if dist < 0.6:  # only inject genuinely similar mistakes
                mistakes.append(meta)
        return mistakes

    def count(self) -> int:
        return self.collection.count()

    def export_dpo_pairs(self, out_path: str):
        """Write all mistakes as chosen/rejected JSONL for DPO/SFT training."""
        results = self.collection.get(include=["metadatas"])
        if not results["metadatas"]:
            print("No mistakes in memory bank yet.")
            return 0
        written = 0
        with open(out_path, "w") as f:
            for meta in results["metadatas"]:
                if meta.get("human_correction") and meta.get("bad_generation"):
                    pair = {
                        "prompt": meta["prompt"],
                        "chosen": meta["human_correction"],
                        "rejected": meta["bad_generation"],
                    }
                    f.write(json.dumps(pair) + "\n")
                    written += 1
        return written

    def migrate_jsonl(self, jsonl_path: str):
        """One-time import of legacy mac_mistakes_memory.jsonl into ChromaDB."""
        if not os.path.exists(jsonl_path):
            return 0
        migrated = 0
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                self.add_mistake(
                    prompt=data.get("prompt", ""),
                    bad_generation=data.get("bad_generation", ""),
                    human_correction=data.get("human_correction", ""),
                )
                migrated += 1
        return migrated

    def build_system_injection(self, prompt: str) -> str:
        """Return a system-prompt block listing past mistakes relevant to this query."""
        mistakes = self.query_similar(prompt, n=3)
        if not mistakes:
            return ""
        lines = ["Past mistakes to avoid:"]
        for i, m in enumerate(mistakes, 1):
            lines.append(
                f"  [{i}] Asked: {m['prompt'][:80]!r} | "
                f"Wrong: {m['bad_generation'][:60]!r} | "
                f"Correct: {m['human_correction'][:80]!r}"
            )
        return "\n".join(lines)
