"""Read-only inspection commands for corpus chunks, persisted vectors, and retrieval."""

from __future__ import annotations

import argparse
import json

from src.ingestion.chunking import chunk_sections
from src.ingestion.indexer import load_persisted_index
from src.ingestion.loaders import load_corpus
from src.pipeline import ClinicalGuidelinesPipeline
from src.utils.config import load_settings, repository_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Clinical Guidelines Assistant artefacts")
    commands = parser.add_subparsers(dest="command", required=True)
    chunks_command = commands.add_parser("chunks", help="Print clause chunks before indexing")
    chunks_command.add_argument("--limit", type=int, default=5)
    index_command = commands.add_parser("index", help="Print persisted FAISS index manifest and sample chunks")
    index_command.add_argument("--limit", type=int, default=3)
    retrieval_command = commands.add_parser("retrieve", help="Show transform/fusion/rerank results")
    retrieval_command.add_argument("query")
    args = parser.parse_args()
    settings = load_settings()

    if args.command == "chunks":
        sections = load_corpus(repository_path(settings["project"]["corpus_dir"]))
        chunks = chunk_sections(
            sections,
            int(settings["ingestion"]["max_chunk_characters"]),
            int(settings["ingestion"]["chunk_overlap_characters"]),
        )
        print(json.dumps([chunk.to_dict() for chunk in chunks[: args.limit]], indent=2))
    elif args.command == "index":
        index, chunks, manifest = load_persisted_index(repository_path(settings["project"]["index_dir"]))
        vector_limit = min(args.limit, index.ntotal)
        sample_vectors = [index.reconstruct(position).tolist() for position in range(vector_limit)]
        print(
            json.dumps(
                {"manifest": manifest, "sample_chunks": chunks[: args.limit], "sample_vectors": sample_vectors},
                indent=2,
            )
        )
    else:
        plan, fused, reranked = ClinicalGuidelinesPipeline().retrieve(args.query)
        print(
            json.dumps(
                {
                    "query_plan": plan.__dict__,
                    "fused": [item.__dict__ for item in fused],
                    "reranked": [item.__dict__ for item in reranked],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
