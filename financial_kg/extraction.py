"""Chat Completions and parsing of the original textual relation format."""

import re
import warnings


RELATION = re.compile(
    r"^\s*(?:(?:\d+[.)]|[-*])\s*)?"
    r"\(([^()]+)\)\s*-\s*\[([^\[\]]+)\]\s*->\s*\(([^()]+)\)\s*$"
)
HEADING = re.compile(
    r"^\s*(?:\d+[.)]\s*)?(summary|relationship pairs|relationships|relations)\s*:?\s*",
    re.IGNORECASE,
)


def parse_relationships(lines):
    relations = []
    for line in lines:
        if not line.strip():
            continue
        match = RELATION.fullmatch(line)
        if match and all(part.strip() for part in match.groups()):
            relations.append(tuple(part.strip() for part in match.groups()))
        else:
            warnings.warn(f"Skipped malformed relationship: {line}", stacklevel=2)
    return relations


def parse_answer(answer):
    """Keep the first summary/relation line, whether headings are present or not."""
    summary_lines, relation_lines = [], []
    in_relations = False
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```"):
            continue
        # Accept common Markdown headings without modifying entity names.
        heading_line = line.lstrip("# ").replace("**", "")
        heading = HEADING.match(heading_line)
        if heading and (
            heading_line[heading.end():] == "" or ":" in heading.group(0)
        ):
            in_relations = heading.group(1).lower() != "summary"
            line = heading_line[heading.end():].strip()
            if not line:
                continue
        if "->" in line or in_relations:
            in_relations = True
            relation_lines.append(line)
        else:
            summary_lines.append(line)
    return "\n".join(summary_lines), parse_relationships(relation_lines)


def chatGPT_to_summary_relation(article, client, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert in financial investigation and a proficient reader of financial and investigative articles.",
            },
            {
                "role": "user",
                "content": (
                    "Read the article and summarize it in 128 words or fewer.\n"
                    "Then extract relationship pairs between financial concepts mentioned in it.\n"
                    "Use these two headings: Summary: and Relationships:.\n"
                    "Under Relationships, write one pair per line, strictly in this format:\n"
                    "(currency)-[acts_as]->(medium_of_exchange)\n"
                    f"Here is the article:\n{article}"
                ),
            },
        ],
        max_tokens=1000,
        temperature=0.8,
    )
    return (response.choices[0].message.content or "").strip()
