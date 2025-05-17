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
                    }
                }
            }
        }
    )


def main() -> None:
    # JSONファイルから昔話データを読み込む
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

    print('\nベクトルを格納 ... \n')

    for folktale in folktales:
        name_vector = model.encode(folktale["name"]).tolist()
        story_vector = model.encode(folktale["story"]).tolist()
        doc = {
            **folktale,
            "name_vector": name_vector,
            "story_vector": story_vector
        }
        response = client.index(index=index_name, body=doc)
        print(f"id: {response['_id']}, name: {folktale['name']}")

    # 検索キーワード（タイトル or 本文か不明）
    query_text = "桃が流れてくる話"
    query_vector = model.encode(query_text).tolist()

    # script_score を使って両ベクトルに重みづけしてスコア化
    query = {
        "size": 3,
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": """
                        double nameSim = cosineSimilarity(params.query_vector, doc['name_vector']);
                        double storySim = cosineSimilarity(params.query_vector, doc['story_vector']);
                        return (0.3 * nameSim + 0.7 * storySim) + 1.0;
                    """,
                    "params": {
                        "query_vector": query_vector
                    }
                }
            }
        }
    }

    search_response = client.search(index=index_name, body=query)

    print("\n検索結果:")
    for hit in search_response["hits"]["hits"]:
        print(f"* {hit['_source']['name']} (score: {hit['_score']:.4f})")


if __name__ == "__main__":
    args = sys.argv
    keyword = None
    if len(args) > 1:
        keyword = args[1]
    else:
        print("キーワードを指定してください。")
        print("Usage: python main.py <keyword>")
        sys.exit(1)
    main()
