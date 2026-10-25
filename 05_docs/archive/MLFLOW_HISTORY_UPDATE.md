> 履歴資料です。旧方式・旧環境の記録を保存しています。現行手順は[LOCAL_RUN.md](../LOCAL_RUN.md)、現行設計は[ARCHITECTURE.md](../ARCHITECTURE.md)を参照してください。ここにある旧コマンド・旧件数は現行版の仕様ではありません。

# MLflowの週次グラフとモデル履歴

このパッチは、既存の`02_model/training/weekly_operations.py`を上書きし、`04_tests/`のテストを追加します。**過去のMLflow履歴は削除しません。** 実行すると別の親Runが生成されます。

## 記録する履歴

- 親RunのMetricsに、4検証週×各ハイパーパラメータ候補の`validation_candidate_N_*`、4テスト週の`test_weekly_*`、18運用週の`operations_weekly_*`を保存します。
- LightGBM・ベースラインそれぞれのMAE・WAPEを、0から始まる週番号`step`で記録します。`operations/metric_step_calendar.json`に`step`と週の日付の対応を保存します。
- 初期採用モデルと再学習モデルを、それぞれ独立した子Runに記録します。モデルはMLflow Modelsの`DemandPilotWeeklyLightGBM`に**登録**します。
- `models/version_history.json`で、プロジェクト内のモデルv1/v2とMLflow登録バージョンを紐付けます。MLflow登録バージョンは実行するたびに増えるため、v1/v2と一致するとは限りません。
- `models/version_history.json`の`week_to_model`に、予測週→使用モデルRun ID→登録バージョン→採用予測の種類を記録します。再学習直後はモデルが自動採用されない場合があるため、`selected_policy`も確認してください。

## 反映と確認

```bash
cd ~/Projects/demandpilot-reorganized
unzip -o ~/Downloads/demandpilot-mlflow-visual-model-patch.zip

docker compose -p demandpilot_reorganized --profile tools build pipeline

docker compose -p demandpilot_reorganized --profile tools run --rm pipeline \
  pytest -q 04_tests/test_mlflow_weekly_tracking.py 04_tests/test_weekly_operations.py

docker compose -p demandpilot_reorganized --profile tools run --rm pipeline \
  python -m training.weekly_operations
```

学習CSVは既存の`01_data/storage/training/weekly_training_full.csv`を再利用できます。

MLflowは`http://localhost:5002`です。新しい`weekly-model-lifecycle` Runを開き、`Model metrics`で`operations_weekly_lightgbm_wape`と`operations_weekly_baseline_wape`を比較します。`Models` → `DemandPilotWeeklyLightGBM`には登録済みバージョンが追加されます。Runの`Artifacts`で`models/version_history.json`を開けば、予測週との対応を確認できます。

## 制限

- 週次指標の横軸は**日付ではなくステップ番号**です。日付は`metric_step_calendar.json`で照合します。
- `Models`への登録はモデルの追跡であり、実環境へ自動的にデプロイすることではありません。
- 既存のStreamlit画面・PostgreSQL・モデルの切り替え処理は変更しません。
- このパッチは構文検査のみ済んでいます。MLflow登録処理を含むDocker実行・新規Runの表示はMacで確認してください。
