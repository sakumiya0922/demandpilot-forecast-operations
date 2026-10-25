> 履歴資料です。旧方式・旧環境の記録を保存しています。現行手順は[LOCAL_RUN.md](../LOCAL_RUN.md)、現行設計は[ARCHITECTURE.md](../ARCHITECTURE.md)を参照してください。ここにある旧コマンド・旧件数は現行版の仕様ではありません。

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

## 3. 旧CSV方式の実験履歴

`demandpilot-weekly-operations`には、旧18週シミュレーションの履歴を保持する。旧シャドー運用の実行経路は現行の`weekly.replay --mode operations`に置き換えた。新しい実行は後述の7節を使う。初期学習の共通関数は`training.weekly_operations`に残す。

## 4. 別経路：DBへの個別予測と画面表示

`02_model/weekly/forecast.py`は、指定された**旧方式の学習Runが保存した`conditions.json`**等を参照し、予測結果をDBの`weekly_forecasts`へ保存する。新方式`weekly_operations.py`のRun IDをそのまま指定できる構成にはなっていない。

```bash
docker compose -p demandpilot_reorganized --profile tools run --rm pipeline \
  python -m weekly.forecast --help
```

**旧方式についての制約：** この個別コマンドは旧`conditions.json`を前提とする。現行の日次・週次運用は7節の統合経路を使い、DB/Streamlitへ反映する。

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


## 7. 8月〜9月の日次・週次運用

### 分離環境の準備

この工程は、初期DBのコピーとMLflowのコピーを使用する。新規のComposeプロジェクト・ボリューム・ホスト保存先を用意し、元のDBや実験領域へは接続しない。初回の検証では以下を使用した。

```bash
export COMPOSE_PROJECT_NAME=demandpilot_operations_20261024
export STORAGE_DIR=./02_model/results/operations-validation-20261024/storage
export STORAGE_MODE=ro
export MLFLOW_DIR=./02_model/results/operations-validation-20261024/mlflow
export RESULTS_DIR=./02_model/results/operations-validation-20261024/runs
export DASHBOARD_PORT=8504
export MLFLOW_PORT=5003
```

既にこの環境が存在する場合は復元や初期化を繰り返さず、後述の運用コマンドを再実行する。別の新規検証にはプロジェクト名と保存先を変更する。

新規環境の手順：

1. 元の初期DBを`pg_dump --data-only --no-owner --no-privileges`で読み出す。
2. 元のMLflowフォルダ一式を、まだ存在しない検証用`MLFLOW_DIR`へコピーする。SQLiteだけでなくArtifactsも必要。
3. `docker compose --profile tools build pipeline`、`docker compose --profile tools up -d db`で新規DBを作り、その新規DBにだけダンプを復元する。新設テーブルは`schema.sql`の初回起動処理で作成される。
4. 元CSVを保持したまま、検証用`STORAGE_DIR/daily/`を次のコマンドで準備する。入力ExcelとCSVのマウントは読み取り専用。出力先が既に存在する場合は拒否する。

```bash
docker compose --profile tools run --rm \
  -v "$PWD/01_data/storage/daily:/original:ro" \
  -v "$PWD/$STORAGE_DIR:/validation" \
  -v "/absolute/path/Online Retail.xlsx:/tmp/source.xlsx:ro" \
  pipeline python -m preparation.prepare_csv /tmp/source.xlsx \
  --existing-dir /original --calendar-start 2011-08-01 --calendar-end 2011-10-02 \
  --output-dir /validation/daily
```

Excelのパスは手元の実ファイルを指定する。CSVがない日の件数を元Excelで日付ごとに確認し、0件の場合だけヘッダーのみCSVを作る。元Excelで対象期間が確認できない場合や、その日に取引がある場合は停止する。`empty_days.json`にExcelのSHA-256・日付・0件・CSVのSHA-256を保存する。取引明細は追加しない。

### 一括実行・再実行

```bash
docker compose --profile tools run --rm -e DEMANDPILOT_OPERATIONS_DB=1 \
  pipeline python -m weekly.replay --mode operations --initial-model-version 6

docker compose --profile tools run --rm pipeline pytest -q -p no:cacheprovider 04_tests
docker compose --profile tools build dashboard mlflow
docker compose --profile tools up -d dashboard mlflow
```

- 運用開始モデルのモデルID・Run ID・100品番・学習終了日は、登録バージョンの実際の`handoff.json`から取得する。
- 月曜04:00 JSTに前週を評価し、必要なら更新して、05:00 JSTに翌7日を予測する。その後で当日の日次CSVを登録する。取引が0件の日も正常処理を台帳に記録する。
- 事前確認では対象63ファイルの存在だけを調べ、当日の販売数量は当日ファイルから読む。予測は前日まで、学習は正解の終了日がその時点までの行だけを使う。DBに未来日付があれば停止する。
- 各評価では7日分の処理済み台帳、CSVハッシュ、DB明細・集計の一致を確認する。0件と未取込を区別する。
- WAPEがベースラインを厳密に上回る週が2回続いたときだけ再学習し、悪化カウントをリセットする。同値・WAPE未定義なら連続悪化をリセットする。更新モデルはその日の予測から使用する。シャドー採用ルールは使わない。
- 最後は9月26日〜10月2日を10月3日に評価する。10月3日のCSVを取り込まず、次週の予測も作らない。
- 同一運用は同じUUID、各予測は運用UUIDと対象週から生成する同じUUIDを使用する。DBの一意制約・日次台帳・MLflowのイベントキーで中断から再開し、完了後の再実行は同じ結果を返す。処理済みCSVが変わった場合は停止する。
- セッションロックで同時実行を拒否する。1DBに異なる運用キーを混在させない。過去実績の訂正を上書きして再評価する運用は対象外。

### 保存テーブルと確認結果

| テーブル | 件数 | 内容 |
|---|---:|---|
| transactions | 364,680 | 初期279,367件＋運用85,313件 |
| daily_sales | 203,304 | 初期156,862件＋運用46,442件 |
| daily_batches | 63 | 8/1〜10/2の全日。0件日10日も記録 |
| weekly_forecast_runs | 9 | 再現日時・使用モデル・MLflow予測Run |
| weekly_forecasts | 900 | 9週×100品番。ベースラインは保存済みsales_sum_7 |
| weekly_evaluations | 9 | 対応する予測ID、評価日時、MAE/WAPE、悪化カウント |
| weekly_evaluation_items | 900 | 予測ID・品番・確定実績・予測・誤差のスナップショット |
| model_updates | 0 | 発生時に理由・旧/新モデル・データ期間を保存 |
| operation_replays | 1 | 再開用の親Run・開始モデル・完了結果 |

日次入力は86,939件、前処理後85,313件。予測9回、評価9回、再学習0回。全週でLightGBMのWAPEがベースラインより小さかった。初期学習後の探索や特徴量追加はしていない。

| 予測対象週（2011年） | LightGBM MAE | WAPE | ベースライン MAE | WAPE |
|---|---:|---:|---:|---:|
| 08-01〜08-07 | 168.35 | 55.33% | 270.51 | 88.90% |
| 08-08〜08-14 | 163.68 | 58.89% | 199.65 | 71.83% |
| 08-15〜08-21 | 140.54 | 55.05% | 184.02 | 72.08% |
| 08-22〜08-28 | 120.48 | 47.07% | 157.21 | 61.42% |
| 08-29〜09-04 | 139.41 | 62.07% | 185.62 | 82.65% |
| 09-05〜09-11 | 132.49 | 50.30% | 206.60 | 78.43% |
| 09-12〜09-18 | 143.89 | 52.42% | 158.24 | 57.65% |
| 09-19〜09-25 | 163.16 | 50.49% | 189.02 | 58.49% |
| 09-26〜10-02 | 123.91 | 51.32% | 168.52 | 69.79% |

MLflow実験は`demandpilot-chronological-operations`。親1・予測9・評価9の計19Run、親Runの週次指標は各9ステップ。再実行でRun・指標・DB履歴が増えないことを確認した。再学習が発生した場合は同じ実験内に再学習Runと登録モデルRunを追加し、理由・旧/新モデル・使用データ・選定済み条件を保存する。

- 親Run：`b4d56a4c0b5447759a9153089224d428`
- 運用ID：`12cbe3c0-07a6-59fe-b7a3-73b908d066c8`
- 使用モデル：`DemandPilotWeeklyLightGBM` version 6
- モデルID：`models:/m-9fd6da1b4fbe4723a190be70e2222b90`
- モデルRun：`8430f244809b40eba1724229d5acb86f`
- 保存先：`02_model/results/operations-validation-20261024/`内の`storage/`・`mlflow/`・`runs/<運用ID>/`

元Excelで0件と確認した日：8/6・13・20・27・29、9/3・10・17・24、10/1。確認元ExcelのSHA-256は`43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d`。

自動テストは77件成功。実データでは別プロセスからモデルを再読込し、保存予測と一致することを確認した。StreamlitはAppTestで9週の指標と全体評価・商品別分析・実行情報を確認し、例外0件。ブラウザでも最終週の51.32%／69.79%と9週の推移を表示できた。元Excel・既存CSV・元MLflow履歴558ファイルのハッシュと、初期DBの全データダンプが実行前後で一致した。

今回の実データでは再学習条件に達しなかったため、新モデルの運用実績はない。再学習・即時更新・悪化カウントのリセットは、意図的に外れる初期モデルを使った結合テストで確認した。自動スケジュール・クラウド運用・任意期間への拡張は対象外。
