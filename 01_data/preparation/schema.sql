-- 元Excelの行番号を一意キーにする。同じ内容の別行は別の取引として保持する。
-- このキーは1つの固定したOnline Retail Excel内だけで有効。
CREATE TABLE IF NOT EXISTS transactions (
    source_row BIGINT PRIMARY KEY CHECK (source_row >= 2),
    invoice_no TEXT NOT NULL CHECK (btrim(invoice_no) <> ''),
    stock_code TEXT NOT NULL CHECK (btrim(stock_code) <> ''),
    invoice_date TIMESTAMP NOT NULL,
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    unit_price NUMERIC NOT NULL CHECK (unit_price >= 0),
    description TEXT,
    customer_id TEXT,
    country TEXT
);
CREATE INDEX IF NOT EXISTS transactions_invoice_date_idx ON transactions(invoice_date);

CREATE TABLE IF NOT EXISTS daily_sales (
    sales_date DATE NOT NULL,
    stock_code TEXT NOT NULL CHECK (btrim(stock_code) <> ''),
    quantity BIGINT NOT NULL CHECK (quantity > 0),
    PRIMARY KEY (sales_date, stock_code)
);

-- 予測のたびに新しい実行IDを付与し、同じ商品・週の再予測も履歴を残す。
CREATE TABLE IF NOT EXISTS weekly_forecasts (
    prediction_run_id UUID NOT NULL,
    week_start DATE NOT NULL CHECK (EXTRACT(ISODOW FROM week_start) = 1),
    stock_code TEXT NOT NULL CHECK (btrim(stock_code) <> ''),
    forecast_origin DATE NOT NULL,
    predicted_quantity NUMERIC NOT NULL CHECK (predicted_quantity >= 0 AND predicted_quantity < 'Infinity'::numeric),
    model_name TEXT NOT NULL CHECK (btrim(model_name) <> ''),
    mlflow_run_id TEXT NOT NULL CHECK (btrim(mlflow_run_id) <> ''),
    model_uri TEXT NOT NULL CHECK (btrim(model_uri) <> ''),
    sales_sum_7 NUMERIC NOT NULL CHECK (sales_sum_7 >= 0 AND sales_sum_7 < 'Infinity'::numeric),
    sales_sum_28 NUMERIC NOT NULL CHECK (sales_sum_28 >= 0 AND sales_sum_28 < 'Infinity'::numeric),
    sales_std_28 DOUBLE PRECISION NOT NULL CHECK (sales_std_28 >= 0 AND sales_std_28 < 'Infinity'::double precision),
    month INTEGER NOT NULL CHECK (month = EXTRACT(MONTH FROM forecast_origin)),
    actual_quantity BIGINT CHECK (actual_quantity >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (prediction_run_id, stock_code),
    CHECK (forecast_origin = week_start - 1)
);
CREATE INDEX IF NOT EXISTS weekly_forecasts_week_idx ON weekly_forecasts(week_start, created_at);


-- 時系列運用の処理済み記録。0件の日も未取り込みと区別する。
CREATE TABLE IF NOT EXISTS daily_batches (
    batch_date DATE PRIMARY KEY,
    source_sha256 TEXT NOT NULL,
    input_rows BIGINT NOT NULL CHECK (input_rows >= 0),
    accepted_rows BIGINT NOT NULL CHECK (accepted_rows >= 0),
    verified_empty BOOLEAN NOT NULL,
    simulated_at TIMESTAMPTZ NOT NULL,
    report JSONB NOT NULL,
    completed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS operation_replays (
    replay_id UUID PRIMARY KEY,
    replay_key TEXT UNIQUE NOT NULL,
    initial_model JSONB NOT NULL,
    mlflow_run_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed')),
    result JSONB
);
CREATE TABLE IF NOT EXISTS weekly_forecast_runs (
    prediction_run_id UUID PRIMARY KEY,
    replay_id UUID NOT NULL REFERENCES operation_replays,
    week_start DATE NOT NULL,
    simulated_at TIMESTAMPTZ NOT NULL,
    mlflow_forecast_run_id TEXT NOT NULL,
    model_info JSONB NOT NULL,
    UNIQUE (replay_id, week_start)
);
CREATE TABLE IF NOT EXISTS weekly_evaluations (
    prediction_run_id UUID PRIMARY KEY REFERENCES weekly_forecast_runs,
    evaluated_at TIMESTAMPTZ NOT NULL,
    scores JSONB NOT NULL,
    consecutive_bad_weeks INTEGER NOT NULL CHECK (consecutive_bad_weeks >= 0),
    mlflow_run_id TEXT NOT NULL
);

-- 新規DB用。既存の9テーブルDBに破壊的な移行は行わない。
