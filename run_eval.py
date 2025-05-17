import csv
from search import search_vector, search_more_like_this
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
import os

# OpenSearchクライアント初期化


def get_client():
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
    return client

# 検索結果を簡易的に抽出（名前のみを取得）


def extract_names(results):
    return [hit["_source"]["name"] for hit in results]


def evaluate_queries(queries, expected, top_k=5):
    client = get_client()
    model = SentenceTransformer("sonoisa/sentence-bert-base-ja-mean-tokens")
    index_name = "japanese-folktales-vector"

    # CSVのカラム名
    fieldnames = ["query", "expected", "vector_results", "mlt_results"]

    with open("evaluation_results.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for q in queries:
            vec_res = search_vector(client, index_name, model, q, top_k=top_k)
            mlt_res = search_more_like_this(client, index_name, q, top_k=top_k)

            vec_names = extract_names(vec_res)
            mlt_names = extract_names(mlt_res)

            writer.writerow({
                "query": q,
                "expected": expected.get(q, ""),
                "vector_results": ";".join(vec_names),
                "mlt_results": ";".join(mlt_names),
            })
            print(f"Query: {q}")
            print(f" Expected: {expected.get(q, '')}")
            print(f" Vector Search: {vec_names}")
            print(f" MoreLikeThis: {mlt_names}\n")


if __name__ == "__main__":
    queries = [
        "カチカチ山",          # 典型的な昔話名
        "タヌキの話",          # タヌキが主役の話
        "桃太郎の鬼退治",      # 特定エピソード
        "浦島太郎の竜宮城",    # 関連エピソードキーワード
        "善悪の教訓",          # 教訓がテーマの昔話全般
        "動物が出る昔話",      # 動物が登場する物語群
        "日本の伝統的な物語",  # 広いカテゴリ
    ]

    expected = {
        # クエリと完全一致するタイトルを期待値に（単一）
        "カチカチ山": ["カチカチ山"],
        # タヌキが主役の話として、カチカチ山以外に「たぬきの話」があれば期待値に追加
        "タヌキの話": ["カチカチ山", "たぬきの話"],
        # 桃太郎の有名な話なので単一期待値
        "桃太郎の鬼退治": ["桃太郎"],
        # 浦島太郎に関連した話は単一期待値で良いかも
        "浦島太郎の竜宮城": ["浦島太郎"],
        # 善悪の教訓テーマなら、複数昔話が該当する想定（例示）
        "善悪の教訓": ["カチカチ山", "桃太郎", "舌切り雀"],
        # 動物が登場する昔話の期待値複数
        "動物が出る昔話": ["カチカチ山", "たぬきの話", "舌切り雀"],
        # 日本の伝統的な物語は全体的に多く該当する想定
        "日本の伝統的な物語": ["カチカチ山", "桃太郎", "浦島太郎", "舌切り雀", "金太郎"],
    }

evaluate_queries(queries, expected)
