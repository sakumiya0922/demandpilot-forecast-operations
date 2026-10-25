# 設計と保存責務

## 7テーブルの役割

| テーブル | 主キー・制約 | 保存する内容 |
|---|---|---|
| transactions | source_row PK | 元Excelの行番号を持つ前処理後の明細 |
| daily_sales | sales_date, stock_code PK | SQLで集計した商品別日次数量 |
| daily_batches | batch_date PK | CSVハッシュ・入力件数・採用件数・0件の根拠・正常完了日時 |
| operation_replays | replay_id PK、replay_key UNIQUE | 開始モデル、親Run、実行状態、確定した結果 |
| weekly_forecast_runs | prediction_run_id PK、replay_id/週 UNIQUE、replay FK | 再現日時、使用モデル、MLflow予測Run |
| weekly_forecasts | prediction_run_id, stock_code PK | 予測値、ベースライン、入力特徴量、確定後のactual_quantity |
| weekly_evaluations | prediction_run_id PK/FK | 評価日時、MAE/WAPE、連続悪化数、MLflow評価Run |

旧`weekly_evaluation_items`は`weekly_forecasts.actual_quantity`へ統合しました。予測値・モデルID・特徴量をJSONに二重保存せず、確定実績だけを追加します。誤差列は同じ計算関数から取得し、週別指標は評価ヘッダーに固定します。予測値は更新しません。未評価はNULL、実績0は0なので区別できます。

評価ヘッダーの作成と全商品の実績更新は1トランザクションです。対象商品が存在しない場合などは全体をロールバックします。評価再実行は予測IDの一意制約で重複を防ぎます。手動の個別予測も残すため、予測明細から運用専用ヘッダーへの必須FKは設けず、運用処理で全商品の対応を確認します。

旧`model_updates`の正本はMLflowの再学習イベント・`new_model.json`・登録モデルへ一本化しました。更新理由、再現日時、データ期間、旧/新モデルのURI・Run IDを保存し、完了結果にも一覧を残します。新しいモデルの登録直後に停止しても、同じイベントから登録済みモデルを回収します。

`operation_replays`は残しました。初回予測前には予測ヘッダーが存在しないため、開始モデルと親Runをそこだけで確定できません。また、完了後の再実行・異なる運用キーの混在拒否にも使います。6テーブルへ合わせるために他のテーブルやMLflowタグへ状態管理を分散させません。

## 処理順序と冪等性

制御は[replay.py](../02_model/weekly/replay.py)です。日付順に進め、月曜は前週評価→必要なら再学習→予測→当日の登録です。最後の月曜は評価だけです。

- 日次は同日をトランザクション内で置換する既存バッチを使用。完了台帳があればCSVハッシュを確認して再登録を省略。
- 予測UUIDは運用UUIDと対象週から決定。一意制約で重複を防止。
- MLflowはイベントキーで同じRunを再開。週次指標も同じstep・値を再追加しない。
- モデル更新の途中・モデル更新後で予測前・親Run確定後でDB完了前の中断を結合テストで確認。
- セッションロックで同時実行を拒否。DBとMLflowに跨る分散トランザクションは使わず、イベントIDと確定済み成果物で回復。

## 未来情報を使わない条件

品番選定は2011-05-29まで。特徴量は予測基準日の日曜まで。再学習は`target_end <= origin`の完全な正解だけ。予測・評価の新規実行前にDBの最大実績日を確認し、未来の実績があれば停止します。

評価では対象7日の処理済み台帳・CSVハッシュ・明細・日次集計を照合します。0件日は元Excelの件数0とハッシュを確認して作成した空CSVだけを受け付けます。2010-12-27〜2011-01-02の全商品共通空白は、既存の特徴量処理で除外します。

## コードと依存

- `01_data/preparation`：CSV分割・0件日の根拠作成・スキーマ
- `01_data/daily`：CSV品質確認・COPY・SQL日次集計
- `01_data/analysis`：選定SQLと補助的なEDA。選定SQLは初期処理で直接使用
- `02_model/training`：特徴量・正解、学習、TimeSeriesSplit、MLflow登録
- `02_model/weekly`：既存モデルの推論、保存、時系列制御
- `02_model/evaluation`：共通のMAE/WAPE、実績照合、保存評価の取得と任意の診断レポート
- `03_app`：DBとローカルMLflowを読み取り専用で表示

旧18週専用HTML生成`overview_report.py`は現行の成果物形式と一致せず、実行経路・テストから参照されないため公開ツリーから除外します。ソースは整理前のGitコミットに残り、旧HTMLやMLflow成果物は削除しません。

`migrate_forecasts.sql`は、空の旧形式予測テーブルの互換性を検証する限定的な補助SQLとして残します。非空の旧形式は拒否します。今回の既存稼働DBへは適用しません。任意診断の`save_evaluation`とEDAは未使用と断定できず、テスト・再現用途があるため維持します。SQLAlchemyはMLflowのSQLite backend、matplotlibはEDA・診断レポートで必要です。
