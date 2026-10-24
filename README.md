# DemandPilot｜需要予測を題材にしたデータ基盤・モデル運用の学習

**目的：** PostgreSQLによるデータ処理・保存、Docker Composeでの実行環境、LightGBMの学習・時系列検証、MLflowによる実験・モデル管理を自分で構築し、説明できるようにする。**予測精度の最大化は目的ではありません。**

## フォルダ構成

| フォルダ | 役割 |
|---|---|
| `01_data/` | データ前処理、日次集計、PostgreSQL、学習用CSV作成に使うデータ |
| `02_model/` | 学習・TimeSeriesSplit・週次運用再現・MLflow・評価 |
| `03_app/` | Streamlitダッシュボード |
| `04_tests/` | 自動テスト |
| `05_docs/` | 実行手順、分析・評価の説明（旧方式の記録も含む） |

ルート直下の`compose.yaml`はサービス連携、`Dockerfile`はPython実行環境、`requirements.txt`は依存ライブラリを定義します。

## 処理の全体像

```text
小売販売データ
  ↓ 01_data：前処理・日次集計・DB保存
PostgreSQL
  ↓ 02_model/training/training_data.py：学習・評価用CSV生成
LightGBM
  ↓ 02_model/training/weekly_operations.py：時系列検証・4週テスト・18週の運用再現
MLflow：学習条件、週別MAE・WAPE、モデルv1/v2、再学習の判断を記録
```

**別経路（画面表示用）：** `02_model/weekly/forecast.py`はMLflowの学習済みモデルを読み、予測をPostgreSQLに保存します。`03_app/app.py`はDBの保存済み予測と実績を読み取ります。**現状、上の週次運用シミュレーションの結果がStreamlitへ自動反映されるわけではありません。**

## 初期データ登録から運用開始モデルまで

`python -m weekly.replay`で、専用の空DBへの日次登録、初期学習、6月4週の検証、7月4週の独立テスト、7月31日時点のモデル準備を一括実行できます。8月以降の予測・監視・再学習は実行しません。

100品番は既存SQLを使い、2010年12月6日〜2011年5月29日の実績だけで選び直します。検証は週単位のTimeSeriesSplitで、3候補の全検証週を合算したWAPEから選定します。独立テストでは7月3日までの確定ラベルで学習したモデルを固定し、7月4〜31日を評価します。

既存DBを保護するため、データのあるDBへの初期一括処理は拒否します。Composeのプロジェクト名、MLflow、出力先を分離する手順と実測値は[ローカル実行手順](05_docs/LOCAL_RUN.md#6-初期登録学習検証独立テストの専用環境)を参照してください。日次バッチ単体の同日再実行は従来どおり二重計上されません。

## ローカルで動かす

現在の検証用Dockerプロジェクト名は`demandpilot_reorganized`です。既存のv2と混同しないため、コマンドには同じ`-p`を指定します。

```bash
docker compose -p demandpilot_reorganized --profile tools build
docker compose -p demandpilot_reorganized --profile tools up -d db mlflow dashboard
docker compose -p demandpilot_reorganized --profile tools run --rm pipeline pytest -q 04_tests
```

- Streamlit（現在のローカル設定）：http://localhost:8503
- MLflow（現在のローカル設定）：http://localhost:5002

データとモデルがない初回環境では、画面だけ起動しても予測結果は表示されません。具体的なデータ生成・MLflow記録手順は [05_docs/LOCAL_RUN.md](05_docs/LOCAL_RUN.md) を参照してください。

## 実装・確認できた範囲

- PostgreSQLとDocker環境を起動し、**63件の自動テストが成功**（2026年10月23日、ユーザーのMacで確認）。Altairの非推奨API警告1件あり。
- LightGBMの4週時系列検証・独立4週テスト・18週の運用再現を実行。MLflowに履歴と登録モデルを保存。
- モデルのWAPEがベースラインより2週連続で悪化したため、2011年10月30日に1回再学習。
- 旧方式の個別予測はDBへ保存でき、Streamlitの商品別画面に予測数量を表示できることを確認。

**未実施：** 自動スケジュール・クラウド本番運用・新しい運用シミュレーションからDB/Streamlitへの自動反映。実在庫の最適化でもありません。

## 公開に関する注意

`01_data/storage/`、`02_model/mlflow/`、`02_model/results/`は実データ・MLflow SQLite・成果物等の保存先です。GitHubにアップロードする前に`.gitignore`と追跡対象を確認してください。ローカルでの動作確認とGitHubへの公開は別作業です。
