"""固定期間のSQL集計・CSV検査からEDAレポートを再生成する（DBは読み取りのみ）。"""
import csv
from datetime import date, timedelta
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/demandpilot-v2-eda-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '02_model' / 'results' / 'analysis'
OUT.mkdir(parents=True, exist_ok=True)
# SQLに固定期間を記載。psqlはエラー時停止し、SQL自体もREAD ONLYで実行する。
result = subprocess.run(['docker','compose','exec','-T','db','psql','-qAt','-v','ON_ERROR_STOP=1',
                         '-U','demandpilot','-d','demandpilot'],
                        input=(ROOT/'01_data/analysis/eda.sql').read_text(),cwd=ROOT,text=True,capture_output=True,check=True)
data = json.loads(result.stdout)
summary=data['summary']; products=data['products']; weeks=data['weekly_totals']; weekly=data['weekly_products']
target=list(csv.DictReader((ROOT/'01_data/analysis/target_products.csv').open()))
selected=[r['stock_code'] for r in target]
assert len(selected)==len(set(selected))==100
# 固定一覧を再選定・上書きせず、集計が当時の条件と一致するか検証する。
eligible=[r for r in products if r['weeks_with_sales']>=20]
assert selected==[r['stock_code'] for r in eligible[:100]], '固定対象と集計が不一致'
assert len(weeks)==25
empty=[r for r in weeks if r['transaction_rows']==0]
assert [r['week_start'] for r in empty]==['2010-12-27']

raw_counts={'input_rows':0,'cancelled':0,'nonpositive_quantity':0,'negative_price':0,'accepted_rows':0}
raw_weekly={r['week_start']:{'raw_rows':0,'csv_days':0} for r in weeks}
inputs=[]; anomaly=[]
for path in sorted((ROOT/'01_data/storage/daily').glob('*.csv')):
    if not '2010-12-01'<=path.stem<='2011-05-31': continue
    day=date.fromisoformat(path.stem)
    monday=(day-timedelta(days=day.weekday())).isoformat()
    if monday in raw_weekly: raw_weekly[monday]['csv_days']+=1
    inputs.append({'file':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            raw_counts['input_rows']+=1
            if monday in raw_weekly: raw_weekly[monday]['raw_rows']+=1
            cancelled=row['InvoiceNo'].strip().upper().startswith('C')
            q=Decimal(row['Quantity']); price=Decimal(row['UnitPrice'])
            if cancelled: key='cancelled'
            elif q<=0: key='nonpositive_quantity'
            elif price<0: key='negative_price'
            else: key='accepted_rows'
            raw_counts[key]+=1
            if row['StockCode'].strip()=='23166' and abs(q)>=1000:
                anomaly.append({k:row[k] for k in ['source_row','InvoiceNo','InvoiceDate','StockCode','Quantity','UnitPrice']})
assert raw_counts['accepted_rows']==summary['transaction_rows']
for row in weeks: row.update(raw_weekly[row['week_start']])
assert all(r['raw_rows']==0 for r in empty)

def write_csv(name,rows):
    with (OUT/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

write_csv('weekly_totals.csv',weeks)
write_csv('product_ranking.csv',products)
write_csv('weekly_products.csv',weekly)
write_csv('selected_top100.csv',[{**r,'selected_rank':selected.index(r['stock_code'])+1} for r in products if r['stock_code'] in selected])
if anomaly: write_csv('23166_large_raw_transactions.csv',anomaly)
record={'generated_at_jst':__import__('datetime').datetime.now(__import__('datetime').timezone(timedelta(hours=9))).isoformat(),
        'python':sys.version,'platform':platform.platform(),'matplotlib':matplotlib.__version__,
        'numpy':np.__version__,'summary':summary,'quality':raw_counts,'input_csv_days':len(inputs),
        'input_files':inputs,'target_products_sha256':hashlib.sha256((ROOT/'01_data/analysis/target_products.csv').read_bytes()).hexdigest(),
        'sql_sha256':hashlib.sha256((ROOT/'01_data/analysis/eda.sql').read_bytes()).hexdigest()}
(OUT/'run_record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
dates=[date.fromisoformat(r['week_start']) for r in weeks]
gap=dates.index(date(2010,12,27))
def finish(name):
    plt.savefig(OUT/name,bbox_inches='tight',facecolor='white'); plt.close()

fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
axes[0].bar(dates,[r['quantity'] for r in weeks],width=5,color='#287b9e')
axes[0].set(ylabel='Observed units',title='All products: Monday–Sunday sales (25 calendar weeks)')
axes[1].bar(dates,[r['raw_rows'] for r in weeks],width=5,color='#5c9380',label='Raw CSV rows')
axes[1].bar(dates,[r['transaction_rows'] for r in weeks],width=3,color='#164f63',label='Accepted DB rows')
axes[1].set(ylabel='Transaction rows',xlabel='Week starting Monday'); axes[1].legend(loc='upper right')
for ax in axes:
    ax.axvspan(dates[gap]-timedelta(days=3),dates[gap]+timedelta(days=3),color='#e3a34a',alpha=.3)
axes[0].annotate('No raw records: Dec 27 – Jan 2\nCause unconfirmed',(dates[gap],0),xytext=(dates[gap]+timedelta(days=8),135000),arrowprops={'arrowstyle':'->'},fontsize=9)
fig.autofmt_xdate(); fig.tight_layout(); finish('weekly_overview.png')

lookup={(r['stock_code'],r['week_start']):r['quantity'] for r in weekly}
def series(code): return [lookup.get((code,r['week_start']),0) if r['transaction_rows'] else np.nan for r in weeks]
fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
axes[0].bar(dates,series('23166'),width=5,color='#ce713e')
axes[0].set(ylabel='Observed units',title='23166: sales concentrated in one week (returns/cancellations excluded)')
for code in selected[:5]: axes[1].plot(dates,series(code),marker='.',label=code,linewidth=1.2)
axes[1].set(ylabel='Observed units',xlabel='Week starting Monday',title='Selected top 5 products')
axes[1].legend(ncol=5,fontsize=8)
for ax in axes: ax.axvspan(dates[gap]-timedelta(days=3),dates[gap]+timedelta(days=3),color='#e3a34a',alpha=.3)
fig.autofmt_xdate(); fig.tight_layout(); finish('product_weekly.png')

fig,axes=plt.subplots(1,2,figsize=(13,6))
axes[0].hist([r['weeks_with_sales'] for r in products],bins=np.arange(.5,25.5,1),color='#287b9e',edgecolor='white')
axes[0].axvline(19.5,color='#ce713e',linestyle='--',label='Candidate: >=20 weeks')
axes[0].set(xlabel='Weeks with positive sales (24 weeks with records)',ylabel='Products',title='Sales frequency across all products'); axes[0].legend(fontsize=8)
top=[next(p for p in products if p['stock_code']==code) for code in selected[:20]][::-1]
axes[1].barh([r['stock_code'] for r in top],[r['total_quantity'] for r in top],color='#287b9e')
axes[1].set(xlabel='Total observed units',title='Selected top 20 / total quantity ranking')
fig.tight_layout(); finish('selection.png')

focus=next(r for r in products if r['stock_code']=='23166')
fraction=focus['max_week_quantity']/focus['total_quantity']*100
share=sum(r['total_quantity'] for r in products if r['stock_code'] in selected)/sum(r['total_quantity'] for r in products)*100
def table(rows,columns,labels):
    lines=['| '+' | '.join(labels)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for row in rows:
        lines.append('| '+' | '.join(str(row[c]).replace('|','/') if row[c] is not None else '' for c in columns)+' |')
    return '\n'.join(lines)
text=f'''# DemandPilot v2：初期データの探索と予測対象の選定

分析日：{record['generated_at_jst'][:10]}（日本時間）。DBは読み取りのみ。モデルの学習・予測は未実行。

## 目的と対象

毎週月曜日に当週月曜〜日曜の販売数量を予測するため、継続的な販売実績がある商品を選ぶ。
半年分の取り込み期間は2010-12-01〜2011-05-31。週次分析は端の不完全な週を除き、
2010-12-06〜2011-05-29の25週を対象とする。集計はSQLで行い、週次実績のテーブルは追加しない。

## 取り込みと品質

| 指標 | 値 |
|---|---:|
| 入力CSVの日数 | {len(inputs):,} |
| 元CSVの明細数 | {raw_counts['input_rows']:,} |
| キャンセル伝票の除外 | {raw_counts['cancelled']:,} |
| その他数量0以下の除外 | {raw_counts['nonpositive_quantity']:,} |
| その他単価負の除外 | {raw_counts['negative_price']:,} |
| 登録明細数 | {summary['transaction_rows']:,} |
| 商品数（取り込み期間全体） | {summary['products']:,} |
| 日次集計行数 | {summary['daily_sales_rows']:,} |
| 販売数量合計 | {summary['quantity']:,} |

除外件数はキャンセル→数量0以下→単価負の順で排他的に計数する。単価0は含む。
商品コードの独自除外はしていないため、必ずしも通常の物販商品のみではない。
同内容でも元行番号が異なる明細は保持する。顧客ID等の欠損は許容する。
これは返品・キャンセルを除く観測販売数量であり、返品差引後の純販売数量や欠品補正済み需要ではない。

## 全商品共通の空白週

![全商品の週別数量・明細件数](analysis_results/weekly_overview.png)

**2010-12-27〜2011-01-02は、DBの登録明細も日別CSVの元明細も0件。**
それ以外の24週には取引記録がある。日別CSVの件数を確認しているため、
DB取り込みでの除外だけによって空白週が発生したわけではない。
ただし今回は元Excel自体を再検査せず、全行を保持して分割済みのCSVと照合している。
休業や元データ欠落などの原因は確定できない。

「25週中24週で販売」は、このデータで記録のある24週のすべてで販売されたことを示す。
当初は商品の販売頻度が低い可能性を挙げたが、全商品集計の確認により共通の空白週を識別した。
商品選定の頻度の分母は24週とする。従来の週平均数量は比較のため合計÷25を維持し、列名を
`mean_calendar_week_quantity` として区別する。
販売のあった週数はキャンセル等を除外した明細に基づき、注文があった週数と同義ではない。

{table(weeks,['week_start','week_end','raw_rows','transaction_rows','products','quantity'],['週開始','週終了','元明細','登録明細','販売商品数','販売数量'])}

## 数量1位23166を対象外とした理由

![23166と選定上位商品の週別数量](analysis_results/product_weekly.png)

23166は合計{focus['total_quantity']:,}個だが、販売は{focus['weeks_with_sales']}週のみ。
最大の1週が{focus['max_week_quantity']:,}個（合計の{fraction:.2f}%）を占める。
毎週の販売傾向を学ぶ目的では継続的な履歴が不足しているため、予測対象から外す。
今後も一時的な販売だけだと断定したものではない。元明細は削除しない。

元CSVの23166について数量の絶対値が1,000以上の明細も確認した：

{table(anomaly,['InvoiceDate','InvoiceNo','Quantity','UnitPrice'],['取引日時','伝票番号','数量','単価']) if anomaly else '該当明細なし。'}

負数量やキャンセルが併記されていても、現在のバッチは販売側の正数量を残すルールである。
この点からも、ランキングを純需要や純売上と読み替えない。

## 予測対象100品番

![販売頻度と選定上位20品番](analysis_results/selection.png)

1. 記録のある24週のうち20週以上で販売がある商品を候補とする。
2. 分析期間の販売数量合計の降順で上位100品番を採用。同数は商品コード昇順。
3. 100品番を[固定一覧](target_products.csv)に保存し、6月以降の結果で選び直さない。

候補は{len(eligible):,}品番。選定100品番の数量は週次分析期間全商品の{share:.2f}%を占める。
20週という基準と100品番という数は、継続性と実装規模を考慮した今回の方針であり、
統計的な最適値ではない。

以下は上位20品番。全100件は[選定結果CSV](analysis_results/selected_top100.csv)、全商品は[ランキングCSV](analysis_results/product_ranking.csv)を参照。

{table([{'rank':i+1,**next(p for p in products if p['stock_code']==code)} for i,code in enumerate(selected[:20])],['rank','stock_code','description','total_quantity','weeks_with_sales'],['順位','品番','商品名','合計数量','販売週数'])}

## 次の学習処理に引き継ぐこと

- 選定は12月〜5月の全体を参照している。この期間内で行う検証は、商品選定まで含めた完全に未知の評価ではない。
- v2の6月以降は今回の集計・選定に使用していない。ただし同じUCIデータを用いた旧版の分析結果は既知であり、データセット全体が完全に未知とは主張しない。
- 日次のカレンダーは維持する。共通空白週を単純に削除して前後の週を隣接させない。
- 共通空白週を含む特徴量や正解ラベルは、原因が未確定なため別扱いとする方針を学習実装時に明示する。
- ほかの日の販売ゼロ補完も「未入荷・未取り込み」と区別が必要。週の一部に記録があることは、すべての日の完全性を保証しない。
- 商品ごとの通常のゼロ販売、販売水準の変化、大口取引への依存を今後も確認する。数量上位だけで予測可能性は保証されない。

## 再現方法と成果物

DB集計は[eda.sql](eda.sql)、選定条件は[select_products.sql](select_products.sql)。
元CSVの品質ルールは[日次バッチ](../01_data/daily/daily_batch.py)と同じ優先順で再計数し、DB登録件数との一致を確認した。
[再生成スクリプト](build_eda.py)は現在のDBとCSVからCSV・図・本文を作り直す。
同じ出力先のEDA成果物を上書きするため、記録を残したい場合は事前に退避する。
固定した `target_products.csv` は上書きしない。DBやモデルも変更しない。

```sh
cd ~/Projects/demandpilot-v2
python3 -m venv /tmp/demandpilot-v2-eda-env
/tmp/demandpilot-v2-eda-env/bin/pip install matplotlib=={matplotlib.__version__} numpy=={np.__version__}
/tmp/demandpilot-v2-eda-env/bin/python 01_data/analysis/build_eda.py
```

Dockerが起動し、v2のDBが稼働していることが前提。DB内に6月以降が追加されても期間条件は固定。
ただし過去の明細修正等があれば再集計結果は変わる。入力CSVのSHA-256、SQL・対象一覧のSHA-256、
生成時刻・実行環境は[実行記録](analysis_results/run_record.json)に保存している。

- [週別全体集計](analysis_results/weekly_totals.csv)：25週の空白週を含む。
- [商品別週次集計](analysis_results/weekly_products.csv)：販売のある商品×週のみ。行なしは欠落かゼロかをこのCSVだけで判別できない。
- [23166の大口元明細](analysis_results/23166_large_raw_transactions.csv)：顧客IDは含めない。

集計CSV・図はEDAの記録であり、DBの新規テーブルや本番処理の入力ではない。
'''
(ROOT/'05_docs/eda.md').write_text(text)
print(json.dumps({'summary':summary,'quality':raw_counts,'candidate_products':len(eligible),'selected_share_pct':share,'anomaly':anomaly},ensure_ascii=False))
print('Created 05_docs/eda.md, 05_docs/analysis_analysis_results/*.csv, 3 PNG charts and run_record.json')
