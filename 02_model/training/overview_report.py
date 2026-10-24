"""既存MLflow Runから読みやすい日本語の週次運用レポートを作る。

学習・実験の再実行はしない。既存成果物を読み取り、overview.htmlだけ追記する。
"""
import argparse
from html import escape
import json
from pathlib import Path
import tempfile

import mlflow
from mlflow.tracking import MlflowClient


def fmt_wape(value):
    return "—" if value is None else f"{value*100:.1f}%"


def fmt_mae(value):
    return "—" if value is None else f"{value:,.1f}"


def svg_chart(monitoring, retrain_events):
    vals = [(i, r) for i, r in enumerate(monitoring)
            if r.get("scores") and r["scores"]["lightgbm"].get("wape") is not None]
    if not vals:
        return '<p>評価済みの週次データはありません。</p>'
    ymax = max([r["scores"][key]["wape"] for _, r in vals for key in ("lightgbm", "baseline")]+[0.1])*1.09
    x0, y0, width, height = 78, 42, 1080, 260
    n = max(len(vals)-1, 1)
    def px(index): return x0 + index * width / n
    def py(value): return y0 + height - value/ymax*height
    lines = []
    for j in range(5):
        val = ymax*j/4
        y = py(val)
        lines.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+width}" y2="{y:.1f}" stroke="#e4e8ee"/>')
        lines.append(f'<text x="{x0-12}" y="{y+4:.1f}" font-size="12" text-anchor="end" fill="#64748b">{val*100:.0f}%</text>')
    models = (("lightgbm", "#2563eb", "LightGBM"),("baseline", "#e07a2f", "ベースライン"))
    for key,color,name in models:
        points=" ".join(f'{px(i):.1f},{py(r["scores"][key]["wape"]):.1f}' for i,r in vals)
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>')
        for i,r in vals:
            lines.append(f'<circle cx="{px(i):.1f}" cy="{py(r["scores"][key]["wape"]):.1f}" r="3.2" fill="{color}"/>')
    for i,r in vals:
        if i % 2==0 or i == vals[-1][0]:
            label = r["origin"][5:]
            lines.append(f'<text x="{px(i):.1f}" y="{y0+height+23}" text-anchor="middle" font-size="12" fill="#64748b">{escape(label)}</text>')
    for ev in retrain_events:
        day=ev["triggered_on"]
        # update after monitoring previous week, before forecast of the triggering date
        idx=next((i for i,r in vals if r["origin"]==day),None)
        if idx is not None:
            x=px(idx)-width/(n*2)
            lines.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0+height}" stroke="#be185d" stroke-width="2" stroke-dasharray="6 5"/>')
            lines.append(f'<text x="{x+6:.1f}" y="{y0+12}" fill="#be185d" font-size="12">{escape(day[5:])} 再学習</text>')
    return f'<svg viewBox="0 0 1220 365" role="img" aria-label="週次WAPEのLightGBMとベースラインの推移">{ "".join(lines) }</svg>'


def render(summary, monitoring_doc, selection, versions, run_id):
    monitor=monitoring_doc.get("monitoring",[])
    events=monitoring_doc.get("retrain_events",[])
    ranking=selection.get("ranking",[])
    rows=[]
    for candidate in ranking:
        params=candidate.get("parameters",{})
        s=candidate.get("scores",{})
        rows.append(f'<tr><td>{escape(str(candidate.get("candidate","—")))}</td><td>{escape(str(params.get("num_leaves","—")))}</td><td>{escape(str(params.get("learning_rate","—")))}</td><td>{escape(str(params.get("n_estimators","—")))}</td><td>{fmt_wape(s.get("lightgbm",{}).get("wape"))}</td></tr>')
    selected=summary.get("selected_hyperparameters",{})
    v=summary.get("cv_overall",{})
    t=summary.get("test_overall",{})
    compare=[]
    for label,stage in (("時系列検証（4週）",v),("独立テスト（4週）",t)):
        compare.append(f'<tr><td>{label}</td><td>{fmt_wape(stage.get("lightgbm",{}).get("wape"))}</td><td>{fmt_wape(stage.get("baseline",{}).get("wape"))}</td><td>{fmt_mae(stage.get("lightgbm",{}).get("mae"))}</td><td>{fmt_mae(stage.get("baseline",{}).get("mae"))}</td></tr>')
    ver_rows=[]
    for model in versions.get("model_versions",[]):
        ver_rows.append(f'<tr><td>v{escape(str(model.get("logical_model_version","—")))}</td><td>{escape(str(model.get("trained_through","—")))}</td><td>{escape(str(model.get("reason","—")))}</td><td>{escape(str(model.get("registered_model_version","—")))}</td><td><code>{escape(str(model.get("model_run_id","—")))[:12]}</code></td></tr>')
    eval_rows=[]
    for r in monitor:
        score=r.get("scores",{})
        eval_rows.append(f'<tr><td>{escape(str(r.get("origin","—")))}</td><td>v{escape(str(r.get("model_version","—")))}</td><td>{fmt_wape(score.get("lightgbm",{}).get("wape"))}</td><td>{fmt_wape(score.get("baseline",{}).get("wape"))}</td><td>{"悪化" if r.get("worse_than_baseline") else "良好"}</td></tr>')
    event_text="、".join(escape(e["triggered_on"]) for e in events) or "なし"
    return f'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DemandPilot｜週次運用レポート</title><style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Hiragino Kaku Gothic ProN","Yu Gothic",sans-serif;margin:0;background:#f7f9fc;color:#1d2f45;line-height:1.6}}main{{max-width:1160px;margin:36px auto;padding:0 22px}}h1{{font-size:30px;margin:0}}h2{{font-size:21px;margin:0 0 14px}}p{{color:#52657a}}section{{background:white;border:1px solid #e4eaf2;border-radius:13px;padding:24px;margin:22px 0}}.lead{{font-size:16px;margin:4px 0 23px}}.kpis{{display:flex;gap:12px;flex-wrap:wrap}}.kpi{{background:#eff5ff;border-radius:10px;padding:14px 20px;min-width:170px}}.kpi b{{display:block;font-size:22px;color:#174e9b}}.small{{font-size:13px;color:#6c7f94}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{text-align:left;border-bottom:1px solid #e6ecf2;padding:9px 10px}}th{{background:#f5f7fb}}.table-scroll{{overflow-x:auto}}svg{{width:100%;height:auto}}.legend{{display:flex;gap:24px;font-size:14px;margin-bottom:12px}}.dot{{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:7px}}code{{font-size:12px}}</style></head><body><main>
<h1>DemandPilot｜週次モデル運用レポート</h1><p class="lead">何を学習し、どのモデルを選び、運用中にいつ再学習したかを確認するための概要です。</p>
<div class="kpis"><div class="kpi">運用週数<b>{len(monitor)}週</b></div><div class="kpi">再学習<b>{len(events)}回</b></div><div class="kpi">採用ハイパーパラメータ<b>num_leaves = {escape(str(selected.get('num_leaves','—')))}</b></div></div>
<section><h2>1. モデル選定 ― どの設定を採用したか</h2><p>6月の4週をTimeSeriesSplitで検証し、WAPEが小さい候補を選定。</p><div class="table-scroll"><table><thead><tr><th>候補</th><th>num_leaves</th><th>learning_rate</th><th>n_estimators</th><th>検証WAPE</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section><h2>2. テスト ― ベースラインを上回ったか</h2><div class="table-scroll"><table><thead><tr><th>評価期間</th><th>LightGBM WAPE</th><th>ベースライン WAPE</th><th>LightGBM MAE</th><th>ベースライン MAE</th></tr></thead><tbody>{''.join(compare)}</tbody></table></div></section>
<section><h2>3. 週次監視 ― 精度は悪化していないか</h2><p>8〜11月の予測を週ごとに実績と照合。縦軸はWAPE（低いほど良い）、横軸は予測基準日。ピンクの破線は再学習タイミングです。</p><div class="legend"><span><i class="dot" style="background:#2563eb"></i>LightGBM</span><span><i class="dot" style="background:#e07a2f"></i>ベースライン</span><span style="color:#be185d">┆ 再学習</span></div>{svg_chart(monitor,events)}</section>
<section><h2>4. 再学習 ― いつ・なぜモデルが変わったか</h2><p>再学習日：<b>{event_text}</b>。WAPEがベースラインより2週連続で悪化した場合、確定済みデータだけで再学習。</p><div class="table-scroll"><table><thead><tr><th>モデル</th><th>学習基準日</th><th>作成理由</th><th>MLflow登録版</th><th>学習Run</th></tr></thead><tbody>{''.join(ver_rows)}</tbody></table></div></section>
<section><h2>補足｜週ごとの詳細</h2><div class="table-scroll"><table><thead><tr><th>予測基準日</th><th>モデル</th><th>LightGBM WAPE</th><th>ベースライン WAPE</th><th>比較</th></tr></thead><tbody>{''.join(eval_rows)}</tbody></table></div></section>
<p class="small">過去データを使用した運用シミュレーションです。自動スケジューリングや本番配信は含みません。MLflow親Run：{escape(run_id)}</p>
</main></body></html>'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',help='weekly-model-lifecycleの親Run ID。省略すると最新の完了済み親Runを使用')
    parser.add_argument('--tracking-dir',type=Path,default=Path('02_model/mlflow'))
    parser.add_argument('--output',type=Path,default=Path('02_model/mlflow/weekly_operations_report.html'))
    args=parser.parse_args()
    uri='sqlite:///'+str(args.tracking_dir.resolve()/'mlflow.db')
    mlflow.set_tracking_uri(uri)
    client=MlflowClient()
    if not args.run_id:
        exp=client.get_experiment_by_name('demandpilot-weekly-operations')
        if exp is None:
            parser.error('実験 demandpilot-weekly-operations が見つかりません')
        runs=client.search_runs([exp.experiment_id], order_by=['attributes.start_time DESC'], max_results=100)
        parents=[r for r in runs if r.info.status=='FINISHED' and r.data.tags.get('purpose')=='historical_replay' and not r.data.tags.get('mlflow.parentRunId')]
        if not parents:
            parser.error('完了済みの週次運用親Runがありません')
        args.run_id=parents[0].info.run_id
    print('対象Run:',args.run_id)
    def artifact(path):
        downloaded=client.download_artifacts(args.run_id,path)
        return json.loads(Path(downloaded).read_text(encoding='utf-8'))
    summary=artifact('summary.json')
    monitoring=artifact('operations/monitoring.json')
    selection=artifact('tuning/selection.json')
    versions=artifact('models/version_history.json')
    report=render(summary,monitoring,selection,versions,args.run_id)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(report,encoding='utf-8')
    client.log_artifact(args.run_id,str(args.output),'report')
    print('概要レポートを作成:',args.output)
    print('MLflow成果物: report/'+args.output.name)
    print('既存の学習・予測履歴は変更していません。')

if __name__=='__main__': main()
