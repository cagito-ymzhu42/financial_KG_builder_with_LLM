import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from bs4 import BeautifulSoup

from financial_kg.clustering import cluster_relations
from financial_kg.crawlers import crawl_sources, fetch_soup, investopedia_extract
from financial_kg.extraction import chatGPT_to_summary_relation, parse_answer, parse_relationships
from financial_kg.graph import Neo4jHandler
from financial_kg.pipeline import run_pipeline


ROOT = Path(__file__).resolve().parent.parent


class ExtractionTests(unittest.TestCase):
    def test_hyphenated_financial_concepts(self):
        self.assertEqual(
            parse_relationships(["1. (short-term_debt)-[is-a]->(liability)"]),
            [("short-term_debt", "is-a", "liability")],
        )

    def test_bad_relation_does_not_discard_good_relation(self):
        with self.assertWarns(UserWarning):
            result = parse_relationships(["(bond)->(debt)", "(bond)-[is_a]->(debt)"])
        self.assertEqual(result, [("bond", "is_a", "debt")])

    def test_heading_free_response_keeps_first_lines(self):
        summary, relations = parse_answer("Currency is money.\n\n(currency)-[acts_as]->(money)")
        self.assertEqual(summary, "Currency is money.")
        self.assertEqual(relations, [("currency", "acts_as", "money")])

    def test_inline_and_markdown_headings(self):
        summary, relations = parse_answer(
            "**Summary:** Currency is money.\n\n### Relationships:\n1. (currency)-[acts_as]->(money)"
        )
        self.assertEqual(summary, "Currency is money.")
        self.assertEqual(len(relations), 1)

    def test_modern_sdk_request_and_response(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="  response  "))]
        )
        self.assertEqual(chatGPT_to_summary_relation("article", client), "response")
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "gpt-3.5-turbo")
        self.assertIn("article", request["messages"][1]["content"])


class CrawlerTests(unittest.TestCase):
    def test_investopedia_fallback_returns_content_dict(self):
        soup = BeautifulSoup("<title>Loan</title><div class='content'>A loan is debt.</div>", "lxml")
        with patch("financial_kg.crawlers.fetch_soup", return_value=soup):
            article = investopedia_extract("https://example.com/loan.asp", "loan")
        self.assertEqual(article["Content"], "A loan is debt.")
        self.assertEqual(article["Entity"], "loan")

    def test_http_failure_is_not_parsed_as_article(self):
        response = Mock()
        response.raise_for_status.side_effect = RuntimeError("HTTP failure")
        with patch("financial_kg.crawlers.requests.get", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "HTTP failure"):
                fetch_soup("https://example.com")

    def test_investopedia_only_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sources.csv"
            path.write_text("\ufeffsource,URL\ninvestopedia,https://example.com/loan.asp\n", encoding="utf-8")
            with patch("financial_kg.crawlers.investopedia_extract", return_value={"Entity": "loan", "Content": "Debt"}):
                articles = list(crawl_sources(path, delay=0))
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]["Source"], "investopedia")


class PipelineTests(unittest.TestCase):
    def test_real_excerpt_example_exports_and_passes_all_relations_to_graph(self):
        records = json.loads((ROOT / "examples/articles.json").read_text(encoding="utf-8"))
        answers = {record["Content"]: record["ExampleResponse"] for record in records}
        graph = Mock()
        with tempfile.TemporaryDirectory() as temp:
            results, relations = run_pipeline(records, answers.__getitem__, temp, graph)
            self.assertEqual(len(results), 3)
            self.assertEqual(len(relations), 9)
            self.assertEqual(graph.create_relationship.call_count, 9)
            for call, relation in zip(graph.create_relationship.call_args_list, relations):
                self.assertEqual(call.args, relation)
            with (Path(temp) / "cluster_keywords.csv").open() as stream:
                keywords = list(csv.DictReader(stream))
            with (Path(temp) / "relation_clusters.csv").open() as stream:
                assignments = list(csv.DictReader(stream))
            self.assertIn("keyword", keywords[0])
            self.assertIn("relations", assignments[0])
            self.assertEqual(len(assignments), 9)
            saved = [json.loads(line) for line in (Path(temp) / "all_output.jsonl").read_text().splitlines()]
            self.assertTrue(all(item["Summary"] and item["URL"] for item in saved))

    def test_clustering_preserves_profit_and_supports_small_input(self):
        keywords, assignments = cluster_relations([("profit", "part_of", "income")])
        self.assertEqual(len(keywords), 1)
        self.assertEqual(assignments.iloc[0]["relations"], "profit part_of income")

    def test_empty_relations_still_have_export_columns(self):
        keywords, assignments = cluster_relations([])
        self.assertEqual(list(keywords.columns), ["cluster_id", "keyword"])
        self.assertEqual(list(assignments.columns), ["cluster_id", "relations"])

    def test_neo4j_preserves_parameterized_graph_model(self):
        with patch("financial_kg.graph.GraphDatabase.driver") as factory:
            graph = Neo4jHandler("bolt://localhost:8888", "financial", "test-password")
            session = factory.return_value.session.return_value.__enter__.return_value
            graph.create_relationship("borrower's debt", "is_a", "liability")
            query = session.run.call_args.args[0]
            self.assertNotIn("borrower's debt", query)
            self.assertIn("r:RELATION", query)
            self.assertEqual(session.run.call_args.kwargs["entity1"], "borrower's debt")
            graph.close()
            factory.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
