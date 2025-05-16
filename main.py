import os
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

    folktales = [
        {
            "name": "桃太郎",
            "story": "昔々、あるところにおじいさんとおばあさんが住んでいました。ある日、おばあさんが川で洗濯をしていると、大きな桃がゆっくりと流れてきました。おばあさんがその桃を拾い上げ家に持ち帰ると、桃が割れて中から元気な男の子が現れました。その子は桃太郎と名付けられ、すくすくと育ちました。やがて、村に悪い鬼が現れ人々を困らせるようになったため、桃太郎は犬、猿、キジの三匹の仲間を家来にして、鬼ヶ島へ鬼退治に出発しました。激しい戦いの末、桃太郎たちは鬼を見事に倒し、村に平和を取り戻しました。",
            "attributes": ["正義感", "チームワーク", "勇気"],
        },
        {
            "name": "浦島太郎",
            "story": "浦島太郎は海辺で子供たちにいじめられていた亀を助けました。その亀は実は竜宮城の使いで、後日浦島太郎を竜宮城へ招待します。竜宮城では美しい乙姫に迎えられ、豪華な宴や楽しい時間を過ごしました。しかし、地上の家族や村のことが気になり帰ることにしました。地上に戻ると何十年も経っていて、見知らぬ景色と変わり果てた村に驚きます。乙姫からもらった玉手箱を開けると、たちまち老人になってしまいました。",
            "attributes": ["弱いものを守る", "約束", "玉手箱"],
        },
        {
            "name": "かぐや姫",
            "story": "竹取の翁が山で光る竹を見つけました。竹を割ると中から美しい女の赤ちゃんが現れ、翁は彼女をかぐや姫と名付け育てました。かぐや姫は成長すると多くの求婚者が現れましたが、かぐや姫は難題を出してふるいにかけました。やがて月の世界の使者が迎えに来て、かぐや姫は月へ帰って行きました。人々はその不思議な物語を今も語り継いでいます。",
            "attributes": ["知的", "お金と権力", "結婚"],
        },
        {
            "name": "一寸法師",
            "story": "一寸法師は小さな体で生まれましたが、勇敢な心を持っていました。お椀を舟に、針を剣にして冒険の旅に出ます。旅の途中で助けた人々に恩返しされ、打ち出の小槌をもらいます。最後にはその小槌の力で鬼を倒し、一人前の武士になります。",
            "attributes": ["お椀の舟", "機転", "打ち出の小槌"],
        },
        {
            "name": "金太郎",
            "story": "金太郎は山の中で動物たちと仲良く育ち、並外れた力持ちでした。毎日山で遊び、動物たちと力比べをして楽しみました。やがてその力が認められ、お偉いさんの家来となって立派に働きました。",
            "attributes": ["強い", "急ぐな休むな", "まさかり"],
        }
    ]

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
    main()
