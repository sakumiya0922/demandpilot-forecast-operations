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


## 6. 初期登録・学習・検証・独立テストの専用環境

既存の日次CSV一式が`01_data/storage/daily/`にあることを前提にする。入力は読み取り専用でマウントする。既存の`demandpilot_reorganized`や`demandpilot_chrono`は使用しない。以下のプロジェクト名・出力先は新規検証用の例であり、既に使用している場合は別名へ変更する。

```bash
export COMPOSE_PROJECT_NAME=demandpilot_initial_local
export STORAGE_MODE=ro
export MLFLOW_DIR=./02_model/results/initial-local/mlflow
export RESULTS_DIR=./02_model/results/initial-local/runs

docker compose --profile tools build pipeline
docker compose --profile tools up -d db
docker compose --profile tools run --rm -e DEMANDPILOT_INITIAL_DB=1 pipeline python -m weekly.replay
docker compose --profile tools run --rm pipeline pytest -q -p no:cacheprovider 04_tests
```

- Composeプロジェクト名から専用の`<project>_postgres_data`ボリュームが作られる。既存DBの初期化やデータ移行は行わない。
- `DEMANDPILOT_INITIAL_DB=1`と空の実績テーブルを確認してから登録する。一括処理を同じDBで再実行すると拒否する。途中で失敗した場合も自動削除や初期化をしない。
- 対象は2010-12-01〜2011-07-31の日次CSV。日付順に既存の日次バッチを呼び、取引明細と日次集計を保存する。日次バッチ単体の同日再実行はトランザクション内の置換であり二重計上しない。
- 登録後は両テーブルの日付範囲、前処理後の明細件数、日次集計件数を確認する。8月以降があれば停止する。
- 出力はUUID付きディレクトリへ保存する。既存の学習CSV、選定品番CSV、モデル、実験履歴を上書きしない。

### 日付と未来情報

| 段階 | 利用する範囲 |
|---|---|
| 品番選定 | 2010-12-06〜2011-05-29。20週以上販売された上位100品番 |
| 初期学習 | 5月31日までに確定した7日ラベル。最初の候補の固定条件で学習 |
| 検証 | 基準日6月5・12・19・26日。対象実績は6月6日〜7月3日 |
| 検証Foldの学習 | 初回は5月31日まで、以後は各基準日までに確定したラベル。週ごとに100品番をまとめる |
| 独立テスト | 基準日7月3・10・17・24日。対象実績は7月4〜31日 |
| テスト用モデル | 7月3日までに確定したラベルで一度学習。4週ともモデルとパラメータを固定 |
| 運用開始モデル | 7月31日までに確定したラベルで、選定済み条件を使って再学習 |

学習行は日曜基準なので、初期学習の最後の正解は5月29日、運用開始モデルの最後の正解は7月31日になる。特徴量は基準日当日までの過去28日から生成する。検証・テストRunの`evaluated_as_of`は各対象週の日曜日で、実績確定後の評価を表す。

CSVのない日付も`input/ingestion.json`へ記録する。元の分割CSV一式が完全であることが前提であり、CSV一覧だけでは後から削除されたファイルと元から明細のない日を区別できない。既知の共通空白期間2010-12-27〜2011-01-02を含む履歴・正解は、既存の特徴量処理で除外する。

### 週別の実測値

MAEは1品番の翌7日合計数量に対する平均絶対誤差。WAPEは100品番の誤差合計÷実績合計。表の日付は予測対象期間で、評価は各期間の終了後に行う。

| 段階 | 対象期間（2011年） | LightGBM MAE | WAPE | ベースライン MAE | WAPE |
|---|---|---:|---:|---:|---:|
| 検証1 | 06-06〜06-12 | 144.40 | 51.97% | 175.79 | 63.27% |
| 検証2 | 06-13〜06-19 | 126.59 | 48.85% | 173.66 | 67.01% |
| 検証3 | 06-20〜06-26 | 119.90 | 56.49% | 171.25 | 80.68% |
| 検証4 | 06-27〜07-03 | 117.96 | 62.25% | 139.66 | 73.70% |
| テスト1 | 07-04〜07-10 | 146.51 | 55.46% | 174.23 | 65.96% |
| テスト2 | 07-11〜07-17 | 117.67 | 60.38% | 201.62 | 103.46% |
| テスト3 | 07-18〜07-24 | 119.70 | 45.22% | 157.99 | 59.68% |
| テスト4 | 07-25〜07-31 | 161.09 | 52.80% | 192.86 | 63.22% |

### MLflowと次工程への受け渡し

実験名は`demandpilot-initial-training`。初期学習、3候補×4Fold、固定モデルの独立テスト4週、運用開始モデルを親子Runで保存する。全検証週の合算WAPEで候補を選び、テスト結果で条件を変更しない。学習CSV、各Fold・固定テスト・運用開始モデルの学習行、品番一覧、日次CSVのSHA-256、MAE/WAPE、コードを保存する。

親Runと運用開始モデルの子Runに`handoff.json`を保存する。モデルURI、モデルRun ID、登録モデル名とバージョン、入力特徴量と順序付き100品番、品番CSVのSHA-256、全パラメータ、利用実績の終了日、初回予測基準日を含む。

次工程では、同じtracking URIを設定し、`mlflow.lightgbm.load_model(handoff["model_uri"])`でnative LightGBMを読み、`training.model.matrix(features, handoff["product_codes"])`でカテゴリ型を復元して推論する。MLflow pyfuncやJSONサービングのインターフェースは今回の対象外。保存後のnativeモデル再読込と予測値の一致は実行時に検証する。現行`weekly.forecast`が要求する旧`conditions.json`との接続は次工程で行う。

### 2026年10月24日の確認結果

- 専用Composeプロジェクト：`demandpilot_initial_20261024`
- 新規DBボリューム：`demandpilot_initial_20261024_postgres_data`
- 日次CSV：193ファイル。CSVのない日付：50日（記録を保持）
- 取引明細：279,367件、商品別日次販売：156,862件
- 両テーブルとも2010-12-01〜2011-07-31。8月以降の登録なし
- 選定条件：`num_leaves=15, learning_rate=0.05, n_estimators=100`
- 検証全体：LightGBM MAE 127.21／WAPE 54.20%、ベースライン165.09／70.34%
- テスト全体：LightGBM MAE 136.24／WAPE 52.97%、ベースライン181.68／70.63%
- 運用開始モデル：2,600行、予測基準日2011-01-30〜2011-07-24、最終正解2011-07-31
- 最終確認Run：`5de1567d6896446b9ff14c34c8cab42a`
- 運用開始モデルRun：`8430f244809b40eba1724229d5acb86f`
- 登録モデル：`DemandPilotWeeklyLightGBM` version `6`
- モデルURI：`models:/m-9fd6da1b4fbe4723a190be70e2222b90`
- 再確認でも候補・検証指標・テスト指標が一致
- 初期の試行では子Runが実験0の一時領域へ保存される不具合を検出。実験IDを明示する修正後、親子23Runとモデル実体の永続保存を確認
- 修正前の試行・モデルversion 1〜4は削除せず`incomplete_artifact_persistence`等の識別タグで保持。次工程では検証済みversion 6を使用
- 作成コンテナ終了後、別コンテナから最終モデルを読み込み、100品番の推論に成功
- 自動テスト：既存63件＋追加8件＝71件成功、失敗0件。Altair・SQLAlchemy由来の非推奨警告6件
- 既存データと元MLflow保存領域の454ファイルをSHA-256で前後照合し、変更なし

最終確認のMLflowは`02_model/results/initial-validation-20261024/mlflow/`、実行結果は同ディレクトリの`runs/`に保存する。コンテナにはそれぞれ`/app/02_model/mlflow/`と`/app/02_model/results/`としてマウントする。MLflowのartifact URIはこのコンテナ内パスを基準にする。Docker内にGitコマンドがないため自動Git SHA記録は警告となるが、コードのスナップショットと入力SHA-256は保存される。

8月以降の日次取り込み、日曜確定後の予測、翌週評価、2週連続悪化時の再学習、予測・評価履歴のDB保存はこの経路では未実装。
