"""CountVectorizer -> TF-IDF -> KMeans, as in the original script."""

from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def cluster_relations(relations, n_clusters=10):
    if n_clusters < 1:
        raise ValueError("n_clusters must be positive")
    texts = [" ".join(relation) for relation in relations]
    if not texts:
        return (
            pd.DataFrame(columns=["cluster_id", "keyword"]),
            pd.DataFrame(columns=["cluster_id", "relations"]),
        )
    vectorizer = CountVectorizer()
    tfidf = TfidfTransformer().fit_transform(vectorizer.fit_transform(texts))
    # The bundled example has fewer relations than the original dataset.
    count = min(n_clusters, len(set(texts)))
    model = KMeans(n_clusters=count, random_state=42, n_init=10).fit(tfidf)
    words = vectorizer.get_feature_names_out()
    order = model.cluster_centers_.argsort()[:, ::-1]
    keywords = pd.DataFrame({
        "cluster_id": range(count),
        "keyword": [words[row[0]] for row in order],
    })
    assignments = pd.DataFrame({"cluster_id": model.labels_, "relations": texts})
    return keywords, assignments.sort_values("cluster_id").reset_index(drop=True)


def save_clusters(relations, output_dir, n_clusters=10):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keywords, assignments = cluster_relations(relations, n_clusters)
    keywords.to_csv(output_dir / "cluster_keywords.csv", index=False)
    assignments.to_csv(output_dir / "relation_clusters.csv", index=False)
