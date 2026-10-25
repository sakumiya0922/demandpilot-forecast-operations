> 履歴資料です。旧方式・旧環境の記録を保存しています。現行手順は[LOCAL_RUN.md](../LOCAL_RUN.md)、現行設計は[ARCHITECTURE.md](../ARCHITECTURE.md)を参照してください。ここにある旧コマンド・旧件数は現行版の仕様ではありません。

# 05：週次予測

日曜までの実績から、固定100品番の翌月曜〜日曜の販売数量を予測する。毎週月曜日に実行する想定である。

## 実行方法

基準日までの日次取り込みを終えてから、プロジェクト直下で実行する。

```sh
# 学習コマンドが出力したrun_idを指定する
MODEL_RUN_ID='<学習時のrun_id>'
docker compose run --rm pipeline python -m weekly.forecast \
  --origin 2011-06-05 --run-id "$MODEL_RUN_ID"
```

| 引数 | 内容 |
|---|---|
| --origin | 実績を締める日曜日 |
| --run-id | 使用する学習済みモデルのMLflow実行ID |
| --products | 固定品番一覧。既定は01_data/analysis/target_products.csv |
| --source-dir | 日別CSVの場所。既定は01_data/storage/daily/ |
| --tracking-dir | MLflow管理DB・成果物の場所。既定は02_model/mlflow/ |
| --prediction-id | 予測実行UUID。省略時は新規発行 |

学習実行IDと予測実行IDは別である。予測の再実行では新しい予測IDを発行し、同じ週の過去の予測も残す。
同じ予測IDの再登録はエラーになる。

## 処理の流れ

1. 直近28日のCSVとDB明細・日次集計を照合する。
2. 学習時と共通の処理で、過去7日・28日合計、28日標準偏差、月、商品コードを用意する。
3. 対象品番と特徴量定義の一致、学習ラベル最終日が基準日以前であることを確認する。
4. 指定されたモデルで予測し、負の予測値は0に補正する。
5. 100品番の予測・入力・モデル情報を同一トランザクションで保存する。

整数への丸めは行わない。学習やモデル選択はこのコマンドの対象外である。

## 保存する情報

`weekly_forecasts`に予測実行ID、対象週、基準日、品番、予測数量、入力特徴量、MLflow実行ID、モデルURI、保存日時を記録する。
主キーは`prediction_run_id × stock_code`である。実績は保存せず、評価時に`daily_sales`から照合する。

## 履歴の扱い

商品別の販売記録なしは、CSVとDBの一致を確認した期間内で0として扱う。
直近28日に一度も売れていない対象品番は、期間前の最新実績1行で品番の存在を確認する。その行は特徴量計算のカレンダー範囲外なので、数量の集計には入らない。
未登録品番や未来にしか実績のない品番は拒否する。

照合は、分割済みCSV群を完全に保持していることが前提である。CSVとDBの同じ日が両方欠けている場合は検知できない。
既知の共通空白期間は不明値として扱い、予測に使わない。

## 旧形式のDBを使う場合

新規DBでは`01_data/preparation/schema.sql`が現在の形式を作成するため、移行は不要である。
旧形式の空テーブルから移行する場合だけ、次を実行する。旧形式に予測行が残っていれば処理を停止する。

```sh
docker compose exec -T db psql -v ON_ERROR_STOP=1 -U demandpilot -d demandpilot \
  < 02_model/weekly/migrate_forecasts.sql
```

実績照合・評価（当時の参照: `step06_evaluation.md`） / 4週間の実行記録（当時の参照: `../05_docs/evaluation_results/four_week_replay_57f6d5b6bf5a43ff856b67dbef4cac2e/report.md`）
