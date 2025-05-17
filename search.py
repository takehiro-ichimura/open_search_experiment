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


def index_documents(client, index_name: str, folktales: list, model) -> None:
    print("\n🔄 ベクトルを登録中...\n")

    for folktale in folktales:
        name_vector = model.encode(folktale["name"]).tolist()

        story_text = folktale["story"]
        attributes_text = " ".join(folktale["attributes"])
        story_and_attributes_vector = model.encode(
            f"{story_text} {attributes_text}").tolist()

        doc = {
            "name": folktale["name"],
            "story": folktale["story"],
            "attributes": folktale["attributes"],
            "name_vector": name_vector,
            "story_vector": story_and_attributes_vector
        }

        response = client.index(index=index_name, body=doc)
        print(f"✅ id: {response['_id']} - {folktale['name']}")


def search_vector(client, index_name: str, model, query_text: str, top_k=5):
    query_vector = model.encode(query_text).tolist()

    query = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": """
                        double nameSim = cosineSimilarity(params.query_vector, doc['name_vector']);
                        double storySim = cosineSimilarity(params.query_vector, doc['story_vector']);
                        return (0.2 * nameSim + 0.8 * storySim) + 1.0;
                    """,
                    "params": {
                        "query_vector": query_vector
                    }
                }
            }
        }
    }

    response = client.search(index=index_name, body=query)
    return response["hits"]["hits"]


def search_more_like_this(client, index_name: str, query_text: str, top_k=5):
    query = {
        "size": top_k,
        "query": {
            "more_like_this": {
                "fields": ["name", "story", "attributes"],
                "like": query_text,
                "min_term_freq": 1,
                "max_query_terms": 12
            }
        }
    }

    response = client.search(index=index_name, body=query)
    return response["hits"]["hits"]


def main(mode: str, query_text: str = None):
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

    if mode == "index":
        with open("folktales.json", "r", encoding="utf-8") as f:
            folktales = json.load(f)
        create_index(client, index_name)
        index_documents(client, index_name, folktales, model)
        print("\n✅ インデックス作成とドキュメント登録が完了しました。")

    elif mode == "search_vector":
        if not query_text:
            print("検索キーワードを指定してください。")
            return
        results = search_vector(client, index_name, model, query_text)
        print(f"\n🔍 ベクトル検索クエリ: {query_text}\n")
        for hit in results:
            print(f"* {hit['_source']['name']} (score: {hit['_score']:.4f})")

    elif mode == "search_mlt":
        if not query_text:
            print("検索キーワードを指定してください。")
            return
        results = search_more_like_this(client, index_name, query_text)
        print(f"\n🔍 More Like This検索クエリ: {query_text}\n")
        for hit in results:
            print(f"* {hit['_source']['name']} (score: {hit['_score']:.4f})")

    else:
        print("モードを指定してください。")
        print("Usage: python main.py <mode> [query]")
        print("mode:")
        print("  index          インデックス作成＆ドキュメント登録")
        print("  search_vector  ベクトル検索")
        print("  search_mlt     More Like This検索")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("モードを指定してください。")
        print("Usage: python main.py <mode> [query]")
        sys.exit(1)

    mode = sys.argv[1]
    query = sys.argv[2] if len(sys.argv) > 2 else None
    main(mode, query)
