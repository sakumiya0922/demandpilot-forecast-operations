# DemandPilot｜ローカル実行手順

プロジェクト直下（`~/Projects/demandpilot-reorganized`）から実行する。既存の`demandpilot-v2`とは別のDockerプロジェクトとして管理する。

## 1. 環境起動・テスト

```bash
docker compose -p demandpilot_reorganized --profile tools build
docker compose -p demandpilot_reorganized --profile tools up -d db mlflow dashboard
docker compose -p demandpilot_reorganized --profile tools run --rm pipeline pytest -q 04_tests
```

- MLflow：http://localhost:5002
- Streamlit：http://localhost:8503
- 起動ポートはMac上の`compose.yaml`に合わせる。競合時は既存コンテナを不用意に停止しない。

## 2. データ準備

PostgreSQLに取引・日次販売データを投入したあと、学習用CSVを作る。以下は**取り込み済みDBがある場合**のコマンド。

```bash
docker compose -p demandpilot_reorganized --profile tools run --rm pipeline \
  python -m training.training_data \
  --start 2010-12-01 --end 2011-12-09 \
  --output 01_data/storage/training/weekly_training_full.csv
```

既存のCSVは上書きしない実装なので、再生成する際は出力名を変える。日次CSVの取り込み方法は`01_data/daily/daily_batch.py`を参照。

## 3. 新しい方式：4週検証→4週テスト→週次運用の再現

```bash
docker compose -p demandpilot_reorganized --profile tools run --rm pipeline \
  python -m training.weekly_operations
```

- 実験：`demandpilot-weekly-operations`
- 記録：候補別TimeSeriesSplit、テスト、週別のMAE・WAPE、再学習モデルv1/v2
- `02_model/mlflow/`にMLflowのSQLiteと成果物を記録する。元v2のMLflow DBは移さず残す。
- レポートは`python -m training.overview_report`で生成する（必要な学習履歴が存在する場合）。

## 4. 別経路：DBへの個別予測と画面表示

`02_model/weekly/forecast.py`は、指定された**旧方式の学習Runが保存した`conditions.json`**等を参照し、予測結果をDBの`weekly_forecasts`へ保存する。新方式`weekly_operations.py`のRun IDをそのまま指定できる構成にはなっていない。

```bash
docker compose -p demandpilot_reorganized --profile tools run --rm pipeline \
  python -m weekly.forecast --help
```

**重要：** 新方式の運用シミュレーション結果は現状DB/Streamlitへ自動反映されない。実際の画面確認には、DBに保存済みの予測データが必要。

## 5. 注意

- `docker compose down -v`は実行しない。DBボリュームが削除される。
- `02_model/mlflow/`は過去実験の再現に必要なので削除しない。
- `05_docs/step04_training.md`は旧学習方式の記録。現行の実行手順ではない。
