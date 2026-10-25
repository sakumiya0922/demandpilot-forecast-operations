# ローカル実行手順

Docker Composeと[UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)の元Excelを使用する。元データはGitHubに含めていない。以下のコマンドはプロジェクト直下で実行する。

## 1. 環境準備

既存のDBや実行結果と分離するため、新しいComposeプロジェクト名と保存先を指定する。

```sh
export COMPOSE_PROJECT_NAME=demandpilot_demo
export RUN_ROOT=./02_model/results/demo
export STORAGE_DIR=$RUN_ROOT/storage
export MLFLOW_DIR=$RUN_ROOT/mlflow
export RESULTS_DIR=$RUN_ROOT/runs
mkdir -p "$STORAGE_DIR" "$MLFLOW_DIR" "$RESULTS_DIR"
docker compose --profile tools build pipeline
```

新規DBの初回起動時に`schema.sql`から7テーブルを作成する。ポートが競合する場合は`DASHBOARD_PORT`と`MLFLOW_PORT`を変更する。

## 2. 日次CSVの準備

`RAW_EXCEL`には元Excelの絶対パスを指定する。

```sh
export RAW_EXCEL='/absolute/path/Online Retail.xlsx'
docker compose --profile tools run --rm \
  -v "$RAW_EXCEL:/input/online-retail.xlsx:ro" \
  pipeline python -m preparation.prepare_csv /input/online-retail.xlsx \
  --output-dir /app/01_data/storage/source_daily

docker compose --profile tools run --rm \
  -v "$RAW_EXCEL:/input/online-retail.xlsx:ro" \
  pipeline python -m preparation.prepare_csv /input/online-retail.xlsx \
  --existing-dir /app/01_data/storage/source_daily \
  --calendar-start 2011-08-01 --calendar-end 2011-10-02 \
  --output-dir /app/01_data/storage/daily
```

元Excelで取引0件と確認できた日だけ空CSVを作成し、`empty_days.json`に根拠とハッシュを保存する。取引のある日のCSV欠落や、既存の出力先への上書きはエラーとする。

## 3. 初期学習・評価

```sh
export STORAGE_MODE=ro
docker compose --profile tools run --rm -e DEMANDPILOT_INITIAL_DB=1 \
  pipeline python -m weekly.replay --mode initial
```

2010年12月〜2011年7月のデータ登録、100品番選定、6月の4週検証、7月の独立4週テスト、運用開始モデルの学習を実行する。初期処理は実績が空のDBで実行する。

`STORAGE_MODE=ro`は日次CSVの保存領域をコンテナから読み取り専用でマウントする設定。`DEMANDPILOT_INITIAL_DB=1`は初期登録を許可する実行フラグで、処理開始時にDBが空であることも確認する。

終了後、出力の`handoff.registered_model_version`を控える。`summary.json`にも記録される。

## 4. 9週間の運用再現

手順3で取得したモデルバージョンを入力する。

```sh
read -r INITIAL_MODEL_VERSION
export INITIAL_MODEL_VERSION
docker compose --profile tools run --rm -e DEMANDPILOT_OPERATIONS_DB=1 \
  pipeline python -m weekly.replay --mode operations \
  --initial-model-version "$INITIAL_MODEL_VERSION"
```

2011年8月1日〜10月2日の日次登録と9週分の予測・評価を再現し、10月3日に最終評価を行う。`DEMANDPILOT_OPERATIONS_DB=1`は運用再現を許可する実行フラグ。開始時には初期データが登録済みか、異なる運用履歴が混在していないかも確認する。途中で停止した場合は同じコマンドで再開できる。入力CSVが変更されている場合は停止する。

## 5. テスト・画面確認

```sh
docker compose --profile tools run --rm pipeline pytest -q -p no:cacheprovider 04_tests
docker compose --profile tools build dashboard mlflow
docker compose --profile tools up -d dashboard mlflow
docker compose ps
```

`docker compose ps`で公開ポートを確認し、StreamlitとMLflowをブラウザで開く。テストには隔離スキーマと一時領域を使用する。

停止は`docker compose stop`、再開は同じ環境変数を設定してから以下を実行する。

```sh
docker compose --profile tools up -d db mlflow dashboard
```

通常の停止・再開でVolumeを削除する必要はない。既存データを保持するため、`down -v`や`prune`は実行しない。

## エラー時

CSV欠落、ハッシュ不一致、未来実績の混入などが検出された場合は処理を停止する。データを削除してやり直す前に、入力とDBの状態を確認する。

処理設計は[ARCHITECTURE.md](ARCHITECTURE.md)、評価結果は[EVALUATION.md](EVALUATION.md)を参照。
