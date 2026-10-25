-- 予測対象を固定するための選定。DBは読み取りのみ。
-- 2010-12-06～2011-05-29の25週を使用する。
-- 全商品共通で記録のない2010-12-27～2011-01-02は頻度の分母から除く。
-- 休業・欠落の原因は未確定。学習時に通常の需要ゼロと決めつけない。
-- 20週以上販売された商品の合計数量上位100品番。同数は商品コード昇順。
-- 出力の週平均は、販売ゼロ週も含むカレンダー上の25週で割る（以前の表と同じ）。
BEGIN READ ONLY;
COPY (
    WITH weekly AS (
        SELECT
            date_trunc('week', sales_date)::date AS week_start,
            stock_code,
            SUM(quantity) AS weekly_quantity
        FROM daily_sales
        WHERE sales_date >= DATE '2010-12-06'
          AND sales_date < DATE '2011-05-30'
        GROUP BY 1, 2
    ), product_totals AS (
        SELECT
            stock_code,
            SUM(weekly_quantity) AS total_quantity,
            ROUND(SUM(weekly_quantity) / 25.0, 2) AS mean_calendar_week_quantity,
            COUNT(*) FILTER (
                WHERE week_start <> DATE '2010-12-27'
                  AND weekly_quantity > 0
            ) AS weeks_with_sales
        FROM weekly
        GROUP BY stock_code
    )
    SELECT
        ROW_NUMBER() OVER (ORDER BY total_quantity DESC, stock_code) AS rank,
        stock_code,
        total_quantity,
        mean_calendar_week_quantity,
        weeks_with_sales
    FROM product_totals
    WHERE weeks_with_sales >= 20
    ORDER BY total_quantity DESC, stock_code
    LIMIT 100
) TO STDOUT WITH CSV HEADER;
COMMIT;
