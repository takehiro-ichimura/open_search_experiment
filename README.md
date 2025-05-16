# OpenSearch 実験

このプロジェクトは、OpenSearch を使用してテキストとベクトルベースのクエリを組み合わせたハイブリッド検索を実現するデモです。OpenSearch、Sentence Transformers、Word2Vec を活用して日本の昔話をインデックス化し、検索します。

## 特徴

- **ハイブリッド検索**: テキストとベクトル埋め込みのコサイン類似度を組み合わせた検索結果を提供。
- **OpenSearch 統合**: インデックス作成とクエリに OpenSearch を使用。
- **事前学習モデル**: `sentence-transformers` と `gensim` を使用したベクトル埋め込み。

## 必要条件

- Docker および Docker Compose がインストールされていること。
- Python 3.8 以上がインストールされていること。
- `requirements.txt` に記載された Python パッケージ。

## セットアップ

1. リポジトリをクローンします:

   ```bash
   git clone https://github.com/your-repo/open_search_experiment.git
   cd open_search_experiment
   ```

2. OpenSearch クラスターを起動します:

   ```bash
   docker-compose up -d
   ```

3. Python の依存関係をインストールします:

   ```bash
   pip install -r requirements.txt
   ```

4. OpenSearch の管理者パスワードを環境変数として設定します:
   ```bash
   export OPENSEARCH_INITIAL_ADMIN_PASSWORD=your_password
   ```

## 使用方法

1. メインスクリプトを実行して昔話をインデックス化および検索します:

   ```bash
   python main.py
   ```

2. スクリプトの処理内容:
   - OpenSearch インデックスを作成。
   - サンプルの日本昔話をベクトル埋め込みとともにインデックス化。
   - ハイブリッド検索を実行し、結果を表示。

## 設定

- **インデックス名**: スクリプト内で `japanese-folktales-hybrid` に設定されています。
- **使用モデル**:
  - Sentence Transformer: `sonoisa/sentence-bert-base-ja-mean-tokens`
  - Word2Vec: `word2vec-google-news-300`

## OpenSearch ダッシュボード

[http://localhost:5601](http://localhost:5601) にアクセスしてデータを可視化および管理できます。

## ライセンス

このプロジェクトは MIT ライセンスの下で提供されています。詳細は LICENSE ファイルをご覧ください。
