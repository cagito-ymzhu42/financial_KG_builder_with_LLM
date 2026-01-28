# Financial Knowledge Graph Builder with LLM

An automated pipeline for constructing a financial knowledge graph by integrating web crawling, topic clustering, and the OpenAI Large Language Model (LLM) API for entity-relation extraction.

## 🌟 Overview

This project automates the transition from raw financial data to a structured knowledge representation. It scrapes financial data sources, performs topic modeling to categorize content, and leverages LLMs to identify key financial entities and their semantic relationships.

## 🚀 Key Features

* **Automated Data Acquisition**: Integrated crawler logic to fetch raw financial text from specified `data_Sources.csv`.
* **Topic Clustering**: Implements clustering algorithms (visualized in `cluster_topic.png`) to group financial news and reports into thematic nodes.
* **LLM-Powered Extraction**: Uses OpenAI's API to perform Named Entity Recognition (NER) and Relation Extraction (RE) from unstructured financial text.
* **Visualized Relationships**: Generates relationship graphs (visualized in `relation_result.png`) to represent connections between companies, markets, and economic indicators.
* **Structured Output**: Consolidates results into a unified format for further analysis or integration into graph databases like Neo4j.

## 📁 Project Structure

```text
.
├── financial_KG_builder_with_LLM_v1.py  # Main execution script
├── data_Sources.csv                    # Input file containing data URLs or text sources
├── all_output.txt                      # Consolidated extraction results
├── cluster_topic.png                  # Visualization of the topic clustering results
├── relation_result.png                # Visualization of the extracted knowledge graph
└── README.md                           # Project documentation

```

## 🛠️ Getting Started

### 1. Prerequisites

* Python 3.8+
* An OpenAI API Key

### 2. Installation

```bash
git clone https://github.com/cagito-ymzhu42/financial_KG_builder_with_LLM.git
cd financial_KG_builder_with_LLM
pip install openai pandas matplotlib scikit-learn  # Add other dependencies as needed

```

### 3. Usage

1. Add your financial data sources or raw text to `data_Sources.csv`.
2. Configure your OpenAI API key in the environment or directly within the script.
3. Run the builder:
```bash
python financial_KG_builder_with_LLM_v1.py

```



## 📊 Sample Outputs

* **Topic Clusters**: View `cluster_topic.png` to see how the script categorizes different financial themes.
* **Knowledge Graph**: Check `relation_result.png` for a visual representation of the entities and their extracted relationships.
