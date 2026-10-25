# ローカル実行手順

## 前提と保存先

Docker Desktop等でDocker Composeを利用できる環境と、[UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)から取得した元Excelが必要です。元データはGitに含まれません。プロジェクト直下から実行します。

初回は未使用のプロジェクト名と保存先を用意します。既存のDB・成果物を上書きするために使わないでください。

```sh
export COMPOSE_PROJECT_NAME=demandpilot_demo
export RUN_ROOT=./02_model/results/demo
export STORAGE_DIR=$RUN_ROOT/storage
export MLFLOW_DIR=$RUN_ROOT/mlflow
export RESULTS_DIR=$RUN_ROOT/runs
mkdir -p "$STORAGE_DIR" "$MLFLOW_DIR" "$RESULTS_DIR"
docker compose --profile tools build pipeline
```

`schema.sql`は新規Volumeでの初回起動時に7テーブルを作成します。旧9テーブルDBへ破壊的な移行はしません。必要ならポート競合を避けるため、`DASHBOARD_PORT`と`MLFLOW_PORT`も指定します。通常の停止・再起動にVolume削除は不要です。

## 1. 日次CSVと0件日の準備

元Excelの絶対パスを`RAW_EXCEL`へ設定してください。以下の代入値は実ファイルへ置き換えます。

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

元Excelと`source_daily`は変更せず、日次入力の検証用コピーを作ります。ファイルのない日が元Excelで取引0件の場合だけヘッダーのみCSVを作成します。取引が存在する欠落、対象期間を確認できない場合、出力先が既に存在する場合は停止します。`empty_days.json`に元Excelと各空CSVのハッシュ・日付・件数0を保存します。

初期期間のCSV一覧の完全性は元Excelから分割したファイル一式が前提です。共通空白週の扱いは[設計](ARCHITECTURE.md)を参照してください。

## 2. 初期登録・学習・検証・独立テスト

```sh
export STORAGE_MODE=ro
docker compose --profile tools run --rm -e DEMANDPILOT_INITIAL_DB=1 \
  pipeline python -m weekly.replay --mode initial
```

2010-12-01〜2011-07-31を登録し、100品番選定、5月末までの初期学習、6月4Fold、7月独立4週、運用開始モデルの順に実行します。DBの実績が空でない場合は拒否します。初期一括処理の途中停止ではデータを削除しません。別の新規DBで最初から確認するか、原因と状態を調査してください。

出力にある`handoff.registered_model_version`を控えます。モデルIDや登録バージョンは新しい実験環境ごとに変わるため、固定値を使いません。保存済みの`summary.json`にも記録されます。

## 3. 9週の運用

`INITIAL_MODEL_VERSION`へ初期処理が返した登録バージョンを設定して実行します。未指定の場合は停止します。

```sh
# 例の数字ではなく、手順2の実際の登録バージョンを設定する
read -r INITIAL_MODEL_VERSION
export INITIAL_MODEL_VERSION
docker compose --profile tools run --rm -e DEMANDPILOT_OPERATIONS_DB=1 \
  pipeline python -m weekly.replay --mode operations \
  --initial-model-version "$INITIAL_MODEL_VERSION"
```

同じコマンドで中断から再開できます。完了後は保存した同じ結果を返します。処理済みCSVの変更や異なる運用キーの混在は拒否します。新しい比較実験には別のComposeプロジェクトと保存先を使ってください。

## 4. テスト・画面

```sh
docker compose --profile tools run --rm pipeline pytest -q -p no:cacheprovider 04_tests
docker compose --profile tools build dashboard mlflow
docker compose --profile tools up -d dashboard mlflow
docker compose ps
```

テストのDBは個別の隔離スキーマ、MLflow・CSVは一時領域を使います。失敗試験で業務テーブルを消しません。接続URLは`docker compose ps`の公開ポートで確認します。画面はDBとMLflowを読み取り専用で参照します。

停止は`docker compose stop`、再開は同じ環境変数を設定した上で`docker compose --profile tools up -d db mlflow dashboard`です。DBのVolumeとホスト上の保存先は維持します。`down -v`やpruneを通常手順に含めません。

## エラー時

CSV欠落・ハッシュ不一致・未取り込み・未来実績・モデルと品番一覧の不一致は、補完せず停止します。既存データを消して解消しないでください。処理の詳細・保存先は[設計](ARCHITECTURE.md)、確認済みの件数と結果は[評価](EVALUATION.md)を参照してください。
