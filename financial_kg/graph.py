"""Keep the original Entity / RELATION graph model."""

from neo4j import GraphDatabase


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
                entity1=entity1, relation=relation, entity2=entity2,
            ).consume()

    def query(self, cypher_query, parameters=None):
        with self.driver.session() as session:
            return [record.data() for record in session.run(cypher_query, parameters)]


def store_relations_in_neo4j(relations, neo4j_handler):
    for entity1, relation, entity2 in relations:
        neo4j_handler.create_relationship(entity1, relation, entity2)
