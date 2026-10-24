-- 既存の空の予測テーブルを履歴対応に変更。明細・日次実績は変更しない。
-- データがある旧形式のテーブルには推測値を補わず停止する。
BEGIN;
LOCK TABLE weekly_forecasts IN ACCESS EXCLUSIVE MODE;
DO $migration$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = current_schema() AND table_name = 'weekly_forecasts'
          AND column_name = 'prediction_run_id'
    ) THEN
        IF EXISTS (SELECT 1 FROM weekly_forecasts) THEN
            RAISE EXCEPTION '旧形式の予測履歴があります。自動移行はしません';
        END IF;
        ALTER TABLE weekly_forecasts
            ADD COLUMN prediction_run_id UUID NOT NULL,
            ADD COLUMN mlflow_run_id TEXT NOT NULL CHECK (btrim(mlflow_run_id) <> ''),
            ADD COLUMN model_uri TEXT NOT NULL CHECK (btrim(model_uri) <> ''),
            ADD COLUMN sales_sum_7 NUMERIC NOT NULL CHECK (sales_sum_7 >= 0 AND sales_sum_7 < 'Infinity'::numeric),
            ADD COLUMN sales_sum_28 NUMERIC NOT NULL CHECK (sales_sum_28 >= 0 AND sales_sum_28 < 'Infinity'::numeric),
            ADD COLUMN sales_std_28 DOUBLE PRECISION NOT NULL CHECK (sales_std_28 >= 0 AND sales_std_28 < 'Infinity'::double precision),
            ADD COLUMN month INTEGER NOT NULL CHECK (month = EXTRACT(MONTH FROM forecast_origin));
        ALTER TABLE weekly_forecasts DROP CONSTRAINT weekly_forecasts_pkey;
        ALTER TABLE weekly_forecasts ADD PRIMARY KEY (prediction_run_id, stock_code);
        ALTER TABLE weekly_forecasts ADD CHECK (predicted_quantity < 'Infinity'::numeric);
    END IF;
END;
$migration$;
CREATE INDEX IF NOT EXISTS weekly_forecasts_week_idx ON weekly_forecasts(week_start, created_at);
COMMIT;
