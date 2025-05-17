import os
import sys
import json
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer


def create_index(client, index_name: str, dimension: int = 768) -> None:
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)

    client.indices.create(
        index=index_name,
        body={
            "settings": {
                "index": {
                    "knn": True
                }
            },
            "mappings": {
                "properties": {
                    "name": {"type": "text"},
                    "story": {"type": "text"},
                    "attributes": {"type": "keyword"},
                    "name_vector": {
                        "type": "knn_vector",
                        "dimension": dimension
                    },
                    "story_vector": {
                        "type": "knn_vector",
                        "dimension": dimension
                    },
                    "attributes_vector": {
                        "type": "knn_vector",
                        "dimension": dimension
                    }
                }
            }
        }
    )


def main(query_text) -> None:
    with open("folktales.json", "r", encoding="utf-8") as f:
        folktales = json.load(f)

    host = "localhost"
    port = 9200
    auth = (
        "admin",
        os.getenv("OPENSEARCH_INITIAL_ADMIN_PASSWORD"),
    )

    client = OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
    )

    index_name = "japanese-folktales-vector"

    model = SentenceTransformer("sonoisa/sentence-bert-base-ja-mean-tokens")

    create_index(client, index_name)

    print('\n🔄 ベクトルを登録中...\n')

    for folktale in folktales:
        name_vector = model.encode(folktale["name"]).tolist()
        story_vector = model.encode(folktale["story"]).tolist()
        attributes_text = " ".join(folktale["attributes"])
        attributes_vector = model.encode(attributes_text).tolist()

        doc = {
            **folktale,
            "name_vector": name_vector,
            "story_vector": story_vector,
            "attributes_vector": attributes_vector
        }
        response = client.index(index=index_name, body=doc)
        print(f"✅ id: {response['_id']} - {folktale['name']}")

    query_vector = model.encode(query_text).tolist()

    query = {
        "size": 5,
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": """
                        double nameSim = cosineSimilarity(params.query_vector, doc['name_vector']);
                        double storySim = cosineSimilarity(params.query_vector, doc['story_vector']);
                        double attrSim = cosineSimilarity(params.query_vector, doc['attributes_vector']);
                        return (0.2 * nameSim + 0.6 * storySim + 0.2 * attrSim) + 1.0;
                    """,
                    "params": {
                        "query_vector": query_vector
                    }
                }
            }
        }
    }

    search_response = client.search(index=index_name, body=query)

    print(f"\n🔍 検索クエリ: {query_text}\n")
    print("🔎 検索結果:")
    for hit in search_response["hits"]["hits"]:
        print(f"* {hit['_source']['name']} (score: {hit['_score']:.4f})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❗検索語を指定してください。")
        print("Usage: python main.py <クエリ>")
        sys.exit(1)

    keyword = " ".join(sys.argv[1:])
    main(query_text=keyword)
