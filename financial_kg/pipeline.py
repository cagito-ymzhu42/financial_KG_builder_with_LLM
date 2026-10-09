"""Small orchestration functions; external services stay in their own modules."""

import json
from pathlib import Path

from .clustering import save_clusters
from .extraction import parse_answer
from .graph import store_relations_in_neo4j


def run_pipeline(articles, extract, output_dir, neo4j_handler=None, n_clusters=10):
    """extract receives article text and returns the original text response."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results, relations = [], []
    with (output_dir / "all_output.jsonl").open("w", encoding="utf-8") as stream:
        for article in articles:
            if not article["Content"].strip():
                raise ValueError(f"Empty article: {article['Entity']}")
            answer = extract(article["Content"])
            summary, article_relations = parse_answer(answer)
            result = dict(article, Summary=summary, Relationships=article_relations, RawAnswer=answer)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            results.append(result)
            relations.extend(article_relations)
    with (output_dir / "relations.json").open("w", encoding="utf-8") as stream:
        json.dump(relations, stream, ensure_ascii=False, indent=2)
    if neo4j_handler is not None:
        store_relations_in_neo4j(relations, neo4j_handler)
    save_clusters(relations, output_dir, n_clusters)
    return results, relations
