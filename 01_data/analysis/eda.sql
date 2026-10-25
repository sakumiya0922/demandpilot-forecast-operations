-- EDA固定期間。DBへの変更は行わない。
BEGIN READ ONLY;
WITH weekly AS (
    SELECT date_trunc('week', sales_date)::date AS week_start,
           stock_code, SUM(quantity) AS quantity
    FROM daily_sales
    WHERE sales_date >= DATE '2010-12-06' AND sales_date < DATE '2011-05-30'
    GROUP BY 1, 2
), weeks AS (
    SELECT generate_series(DATE '2010-12-06', DATE '2011-05-23', INTERVAL '7 days')::date AS week_start
), weekly_transactions AS (
    SELECT date_trunc('week', invoice_date)::date AS week_start,
           COUNT(*) AS transaction_rows, COUNT(DISTINCT invoice_date::date) AS sales_days
    FROM transactions
    WHERE invoice_date >= TIMESTAMP '2010-12-06' AND invoice_date < TIMESTAMP '2011-05-30'
    GROUP BY 1
), product_stats AS (
    SELECT stock_code, SUM(quantity) AS total_quantity,
           COUNT(*) AS weeks_with_sales, MAX(quantity) AS max_week_quantity,
           ROUND(SUM(quantity)/25.0, 2) AS mean_calendar_week_quantity
    FROM weekly GROUP BY stock_code
), names AS (
    SELECT DISTINCT ON (stock_code) stock_code, description
    FROM transactions
    WHERE invoice_date >= TIMESTAMP '2010-12-01' AND invoice_date < TIMESTAMP '2011-05-30'
      AND description IS NOT NULL AND btrim(description) <> ''
    ORDER BY stock_code, invoice_date DESC, source_row DESC
)
SELECT json_build_object(
    'summary', (
        SELECT json_build_object(
            'transaction_rows', COUNT(*), 'products', COUNT(DISTINCT stock_code),
            'quantity', SUM(quantity), 'start_date', MIN(invoice_date)::date,
            'end_date', MAX(invoice_date)::date,
            'daily_sales_rows', (SELECT COUNT(*) FROM daily_sales WHERE sales_date >= DATE '2010-12-01' AND sales_date < DATE '2011-06-01')
        ) FROM transactions
        WHERE invoice_date >= TIMESTAMP '2010-12-01' AND invoice_date < TIMESTAMP '2011-06-01'
    ),
    'weekly_totals', (
        SELECT json_agg(r ORDER BY week_start) FROM (
            SELECT w.week_start, w.week_start+6 AS week_end,
                   COALESCE(t.transaction_rows,0) AS transaction_rows,
                   COALESCE(t.sales_days,0) AS sales_days,
                   COUNT(s.stock_code) AS products, COALESCE(SUM(s.quantity),0) AS quantity
            FROM weeks w LEFT JOIN weekly_transactions t USING (week_start)
            LEFT JOIN weekly s USING (week_start)
            GROUP BY w.week_start,t.transaction_rows,t.sales_days
        ) r
    ),
    'products', (
        SELECT json_agg(r ORDER BY total_quantity DESC,stock_code) FROM (
            SELECT s.*, n.description FROM product_stats s LEFT JOIN names n USING (stock_code)
        ) r
    ),
    'weekly_products', (SELECT json_agg(r ORDER BY week_start,quantity DESC,stock_code) FROM weekly r)
);
COMMIT;
