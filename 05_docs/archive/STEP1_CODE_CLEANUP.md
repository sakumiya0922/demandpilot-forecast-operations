> 履歴資料です。旧方式・旧環境の記録を保存しています。現行手順は[LOCAL_RUN.md](../LOCAL_RUN.md)、現行設計は[ARCHITECTURE.md](../ARCHITECTURE.md)を参照してください。ここにある旧コマンド・旧件数は現行版の仕様ではありません。

# STEP 1 — 旧学習コードの整理

現在の `demandpilot-reorganized` に上書き展開してください。

## 変更
- `training/model.py`：旧train.pyから共通の `PARAMETERS` と `matrix` を移動
- `evaluation/metrics.py`：旧train.pyからMAE・WAPEの `metrics` を移動
- `weekly_operations.py`、`forecast.py`、`evaluate.py`：共通処理のimportを変更
- `test_train.py`：使われなくなった旧分割方法のテストを共通処理のテストに置き換え
- `README.md`：旧コマンドの参照があれば新コマンドへ修正

## 重要：train.pyの削除はテスト合格後だけ
旧 `02_model/training/train.py` はZIPから削除できないため、ローカルでテスト実行が成功した後に手動で削除します。

```bash
cd ~/Projects/demandpilot-reorganized
unzip -o ~/Downloads/demandpilot-model-code-cleanup-patch.zip
docker compose -p demandpilot_reorganized --profile tools build pipeline
docker compose -p demandpilot_reorganized --profile tools run --rm pipeline pytest -q 04_tests
# 全件成功を確認したあとだけ実行：
rm 02_model/training/train.py
```

実データ・MLflow履歴・PostgreSQLの内容は変更しません。
※ZIPに含まれるコードの構文と共通関数の挙動は確認しました。MacのDocker実機テストは未実施です。
