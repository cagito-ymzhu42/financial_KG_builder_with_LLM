"""Command-line configuration without a separate configuration framework."""

import argparse
import json
import os
from pathlib import Path

from .crawlers import crawl_sources
from .extraction import chatGPT_to_summary_relation
from .graph import Neo4jHandler
from .pipeline import run_pipeline


ROOT = Path(__file__).resolve().parent.parent


def main(argv=None):
    parser = argparse.ArgumentParser(description="Crawl, extract, store and cluster financial relationships.")
    parser.add_argument("--sources", type=Path, default=ROOT / "data_Sources.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--delay", type=float, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-neo4j", action="store_true", help="Export files without writing a database.")
    parser.add_argument("--example", action="store_true", help="Run real article excerpts with hand-written example responses; no network or database.")
    args = parser.parse_args(argv)
    if args.clusters < 1 or args.delay < 0 or (args.limit is not None and args.limit < 1):
        parser.error("clusters and limit must be positive; delay must be nonnegative")

    client, graph = None, None
    try:
        if args.example:
            records = json.loads((ROOT / "examples" / "articles.json").read_text(encoding="utf-8"))
            if args.limit is not None:
                records = records[:args.limit]
            answers = {record["Content"]: record["ExampleResponse"] for record in records}
            articles = [{key: value for key, value in record.items() if key != "ExampleResponse"} for record in records]
            extract = answers.__getitem__
            print("Offline example: real repository excerpts + hand-written responses (not live LLM output).")
        else:
            from openai import OpenAI

            if not os.getenv("OPENAI_API_KEY"):
                parser.error("Set OPENAI_API_KEY, or use --example for the offline example")
            if not args.skip_neo4j and not os.getenv("NEO4J_PASSWORD"):
                parser.error("Set NEO4J_PASSWORD, or use --skip-neo4j")
            client = OpenAI()
            articles = crawl_sources(args.sources, delay=args.delay, limit=args.limit)

            def extract(text):
                return chatGPT_to_summary_relation(text, client, args.model)

            if not args.skip_neo4j:
                graph = Neo4jHandler(
                    os.getenv("NEO4J_URI", "bolt://localhost:8888"),
                    os.getenv("NEO4J_USER", "financial"),
                    os.environ["NEO4J_PASSWORD"],
                )
        results, relations = run_pipeline(articles, extract, args.output_dir, graph, args.clusters)
        print(f"Processed {len(results)} articles and {len(relations)} relationships -> {args.output_dir.resolve()}")
    finally:
        if graph is not None:
            graph.close()
        if client is not None:
            client.close()
