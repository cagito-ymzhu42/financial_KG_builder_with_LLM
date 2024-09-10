# -*- coding: utf-8 -*-

import requests
import time
import codecs
import json
import urllib
import urllib3
from bs4 import BeautifulSoup
import pandas as pd

import openai

from sklearn.cluster import KMeans
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from neo4j import GraphDatabase

import re

def parse_relationships(relation_list):
  relations = []
  for line in relation_list:
    if line:
      # (entity1)-[relation]->(entity2)
      line = re.sub(r'^[\d\-. ]+', '', line)
      parts = line.split("->")
      if len(parts) == 2:
        tmp = parts[0].split("-")
        entity1  = tmp[0].strip("()")
        relation = tmp[1].strip("[]")
        entity2 = parts[1].strip("()")
        relations.append((entity1, relation, entity2))
  return relations

def store_relations_in_neo4j(relations, neo4j_handler):
    for entity1, relation, entity2 in relations:
        neo4j_handler.create_relationship(entity1, relation, entity2)

class Neo4jHandler:
  def __init__(self, uri, user, password):
    self.driver = GraphDatabase.driver(uri, auth=(user, password))

  def close(self):
    self.driver.close()

  def create_relationship(self, entity1, relation, entity2):
    with self.driver.session() as session:
        session.run(
            "MERGE (a:Entity {name: $entity1}) "
            "MERGE (b:Entity {name: $entity2}) "
            "MERGE (a)-[r:RELATION {type: $relation}]->(b)",
            entity1=entity1, relation=relation, entity2=entity2
        )

  def query(self, cypher_query, parameters=None):
    with self.driver.session() as session:
        result = session.run(cypher_query, parameters)
        return [record.data() for record in result]


def wiki_extract(url, keyword):
  result_dict = {}
  result_dict['Entity'] = keyword

  url = url.replace('&apos;', '%27')  # '
  headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
  html = requests.get(url, timeout=15, verify=False, headers=headers).text
  soup = BeautifulSoup(html, 'lxml')
  main_content_div = soup.find('div', {'class': 'mw-parser-output'})

  main_content = ""
  if main_content_div:
      for paragraph in main_content_div.find_all('p'):
          main_content += paragraph.get_text().replace("\n", "").strip()+"\n"

  result_dict['Content'] = main_content

  related_links = soup.find_all('ul')

  related_entities = [link.get_text().replace("\n", "").strip() for link in related_links]
  related_entities = [e for e in related_entities if len(e)>4 and len(e) <15]

  result_dict['Related_Entities'] = related_entities

  return result_dict

def investopedia_extract(url, keyword):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}

    result_dict = {}
    result_dict['Entity'] = keyword

    url = url.replace('&apos;', '%27')  # '
    html = requests.get(url, timeout=15, verify=False, headers=headers).text
    soup = BeautifulSoup(html, 'lxml')
    title = soup.find('title')

    if title:
        title_content = title.get_text(strip=True)
    else:
        title_content = ""
    article = soup.find('article')
    content = ""
    if article:
        content = article.get_text(strip=True)

    else:
      content_div = soup.find('div', {'class': 'content'})
      if content_div:
          return content_div.get_text(strip=True)

      paragraphs = soup.find_all('p')
      if paragraphs:
          content = ' '.join([p.get_text(strip=True) for p in paragraphs])

    result_dict["Title"] = title_content
    result_dict["Article"] = content

    return result_dict

def chatGPT_to_summary_relation(article):
  messages = [
      {"role": "system", "content": "You are an expert in financial investigation and a proficient reader of financial and investigative articles."},
      {"role": "user", "content": f"1. Please carefully read the provided article and summarize it in 128 words or fewer.\
      2. Additionally, identify and extract relationship pairs between financial concepts mentioned in the article. \
      Output these pairs in a format compatible for use in Neo4j, ensuring the relationships between the entities are clear and structured.the format is strictly like (currency)-[acts_as]->(medium_of_exchange)\
      Here is the article:{article}"
      }
  ]

  response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo", 
      messages=messages,
      max_tokens=1000,
      temperature=0.8
  )

  answer = response['choices'][0]['message']['content'].strip()
  return answer


if __name__ == "__main__":
    df = pd.read_csv("/data_Sources.csv")
    df.head()

    wiki_list = []
    investopedia_list = []
    for i in range(len(df)):
        if df["source"][i] == "wiki":
          url = df["URL"][i]
          entity = df["URL"][i].split("/")[-1].strip()
          entity = urllib.parse.unquote(entity)
          wiki_list.append((url, entity))
        elif df["source"][i] == "investopedia":
          url = df["URL"][i]
          entity = df["URL"][i].split("/")[-1].replace(".asp","").strip()
          entity = urllib.parse.unquote(entity)
          investopedia_list.append((url, entity))

    print(wiki_list[:5])
    print(investopedia_list[:5])


    count = 0
    article_length = 0
    wiki_bag = []
    fpath = 'wiki_output.txt'
    fp = codecs.open(fpath, 'w', encoding='utf-8')
    for k in wiki_list:
        print(k)
        dict_tmp = wiki_extract(k[0], k[1]) # 具体抽取部分
        wiki_bag.append(dict_tmp)
        json.dump(dict_tmp, fp, ensure_ascii=False)
        fp.write('\n')
        time.sleep(5)
        count += 1
        article_length += len(dict_tmp["Content"])
    fp.close()

    wiki_bag[0]['Content']

    print(article_length/count)


    investopedia_bag = []
    fpath = 'investopedia_output.txt'
    fp = codecs.open(fpath, 'w', encoding='utf-8')
    for k in investopedia_list:
        print(k)
        dict_tmp = investopedia_extract(k[0], k[1])
        investopedia_bag.append(dict_tmp)
        json.dump(dict_tmp, fp, ensure_ascii=False)
        fp.write('\n')
        time.sleep(5)
    fp.close()


    openai.api_base = ""
    openai.api_key = ""

    GPT_answers = []
    all_results = []
    all_summary_list = []
    all_relationships_list = []
    for dic in wiki_bag:
      print(dic["Entity"])
      dic["Soure"] = "wiki"
      GPT_answer = chatGPT_to_summary_relation(dic['Content'])
      GPT_answers.append(GPT_answer)
      parts = GPT_answer.split("\n\n")
      summary = "\n".join(parts[0].split("\n")[1:])
      dic["Summary"] = summary
      all_summary_list.append(summary)
      try:
        relationships_list = parts[1].split("\n")[1:]
        all_relationships_list += relationships_list
      except:
        print("Problem in Summary and Relation Extraction: ", dic["Entity"])
        continue
      all_results.append(dic)

    for dic in investopedia_bag:
      print(dic["Entity"])
      dic["Soure"] = "investopedia"
      GPT_answer = chatGPT_to_summary_relation(dic['Article'])
      GPT_answers.append(GPT_answer)
      parts = GPT_answer.split("\n\n")
      # 提取 Summary 部分
      summary = "\n".join(parts[0].split("\n")[1:])
      dic["Summary"] = summary
      all_summary_list.append(summary)
      try:
        # 提取 Relationship Pairs 部分
        relationships_list = parts[1].split("\n")[1:]
        all_relationships_list += relationships_list
      except:
        print("Problem in Summary and Relation Extraction: ", dic["Entity"])
        continue
      all_results.append(dic)





    neo4j_handler = Neo4jHandler("bolt://localhost:8888", "financial", "1234567")


    relations_neo4j = parse_relationships(all_relationships_list)
    relations_neo4j[:5]

    store_relations_in_neo4j(relations_neo4j, neo4j_handler)


    relations_neo4j_text = [" ".join(rela).replace("of", "") for rela in relations_neo4j]

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(relations_neo4j_text))
    word = vectorizer.get_feature_names_out()

    clu_10, n_clusters = KMeans(n_clusters=10).fit(tfidf), 10

    key_choice = 1
    order_centroid = clu_10.cluster_centers_.argsort()[:, ::-1]
    cluster_top1 = []
    for i in range(n_clusters):
        clusters=[]
        for ind in order_centroid[i, :key_choice]:
            clusters.append(word[ind])
        cluster_top1.append(clusters)

    cluster_top1_df = pd.DataFrame(cluster_top1)

    cluster_top1_df.to_csv("10_cluster_result.csv", index=False)

    relation_cluster = pd.DataFrame()
    relation_cluster['cluster_id'] = clu_10.labels_
    relation_cluster["relations"] = relations_neo4j_text

    relation_cluster = relation_cluster.sort_values(by = 'cluster_id',axis = 0,ascending = True)
    relation_cluster = relation_cluster.reset_index(drop=True)
    relation_cluster.to_csv("10_cluster_result.csv", index=False)