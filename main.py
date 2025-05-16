import os
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
import gensim.downloader as api
import numpy as np


def create_index(client, index_name: str, dim_story: int = 768, dim_title: int = 300) -> None:
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
                    "title_vector": {
                        "type": "knn_vector",
                        "dimension": dim_title
                    },
                    "story_vector": {
                        "type": "knn_vector",
                        "dimension": dim_story
                    }
                }
            }
        }
    )


def average_word2vec(text, model, dim=300):
    tokens = text.split()
    vectors = [model[word] for word in tokens if word in model]
    if not vectors:
        return np.zeros(dim).tolist()
    return np.mean(vectors, axis=0).tolist()


def main() -> None:
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

    index_name = "japanese-folktales-hybrid"

    story_model = SentenceTransformer(
        "sonoisa/sentence-bert-base-ja-mean-tokens")
    word_model = api.load("word2vec-google-news-300")

    folktales = [
        {
            "name": "桃太郎",
            "story": "昔々...（省略）",
            "attributes": ["正義感", "チームワーク", "勇気"],
        },
        # ... 他の物語も同様に定義 ...
    ]

    create_index(client, index_name)

    print('\nベクトルを格納 ...\n')
    for folktale in folktales:
        title_vector = average_word2vec(folktale["name"], word_model)
        story_vector = story_model.encode(folktale["story"]).tolist()
        doc = {
            **folktale,
            "title_vector": title_vector,
            "story_vector": story_vector
        }
        response = client.index(index=index_name, body=doc)
        print(f"id: {response['_id']}, name: {folktale['name']}")

    # クエリ
    query_title = "桃太郎"
    query_story = "桃が川から流れてくる話"

    title_vec = average_word2vec(query_title, word_model)
    story_vec = story_model.encode(query_story).tolist()

    query = {
        "size": 3,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": """
                        cosineSimilarity(params.query_title, 'title_vector') * 0.3 +
                        cosineSimilarity(params.query_story, 'story_vector') * 0.7
                    """,
                    "params": {
                        "query_title": title_vec,
                        "query_story": story_vec
                    }
                }
            }
        }
    }

    response = client.search(index=index_name, body=query)
    print("\nハイブリッド検索結果:")
    for hit in response["hits"]["hits"]:
        print(f"* {hit['_source']['name']} (score: {hit['_score']:.4f})")


if __name__ == "__main__":
    main()
