# MLflowの読みやすい概要レポート

既存の`weekly-model-lifecycle`親Runに保存済みの4種類のJSON成果物から、**目的別の4セクション**を含む日本語HTMLレポートを作成します。学習やテストは再実行しません。

```bash
# 現在のプロジェクトルートで実行
# 変更後は pipeline だけ再ビルド
# 最新の完了済み週次運用Runを自動選択

docker compose -p demandpilot_reorganized --profile tools build pipeline

docker compose -p demandpilot_reorganized --profile tools run --rm pipeline \
  python -m training.overview_report
```

出力は`02_model/mlflow/weekly_operations_report.html`です。Macのブラウザでこのファイルを開けます。MLflowのArtifactsにも`report/weekly_operations_report.html`として登録します。

MLflow標準のModel metricsに自動生成された多数のチャートは**そのまま残ります**。標準UIの自動チャートを直接削除・並べ替えるパッチではありません。概要レポートを優先して確認してください。
