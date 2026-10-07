"""Describe development forecast failures; never reads reserved-test outcomes."""
import csv
import json
from collections import defaultdict
from pathlib import Path


def run(root):
    source=root/'work/period_checks/predictions.csv'
    rows=list(csv.DictReader(source.open(encoding='utf-8')))
    report=json.loads((source.parent/'results.json').read_text())
    cutoff=report.get('final_test_evaluated')
    if cutoff is not False: raise ValueError('Expected development-only predictions')
    moves=defaultdict(dict)
    for row in rows:
        for key in ('actual_price','persistence','predicted_price'): row[key]=float(row[key])
        row['seed']=int(row['seed'])
        actual,baseline,predicted=row['actual_price'],row['persistence'],row['predicted_price']
        row['actual_change_pct']=100*(actual/baseline-1)
        row['predicted_change_pct']=100*(predicted/baseline-1)
        row['model_error_usd']=abs(predicted-actual)
        row['baseline_error_usd']=abs(baseline-actual)
        row['extra_error_usd']=row['model_error_usd']-row['baseline_error_usd']
        row['direction_wrong']=(actual-baseline)*(predicted-baseline)<0
        moves[row['period_start']][row['target_date']]=abs(row['actual_change_pct'])
    thresholds={period:sorted(values.values())[int(.9*(len(values)-1))] for period,values in moves.items()}
    grouped=defaultdict(list)
    for row in rows:
        row['large_move']=abs(row['actual_change_pct'])>=thresholds[row['period_start']]
        grouped[(row['period_start'],row['model'])].append(row)
    summaries=[]
    for (period,model),values in grouped.items():
        large=[r for r in values if r['large_move']]
        quiet=[r for r in values if not r['large_move']]
        avg=lambda key,items:sum(r[key] for r in items)/len(items) if items else None
        summaries.append({'period_start':period,'model':model,'seeds':len({r['seed'] for r in values}),
                          'evaluation_dates':len({r['target_date'] for r in values}),
                          'mean_extra_error_usd':avg('extra_error_usd',values),
                          'forecasts_worse_than_baseline_pct':100*sum(r['extra_error_usd']>0 for r in values)/len(values),
                          'wrong_direction_pct':100*sum(r['direction_wrong'] for r in values)/len(values),
                          'large_move_threshold_pct':thresholds[period],
                          'large_move_model_mae_usd':avg('model_error_usd',large),
                          'large_move_baseline_mae_usd':avg('baseline_error_usd',large),
                          'other_day_model_mae_usd':avg('model_error_usd',quiet),
                          'other_day_baseline_mae_usd':avg('baseline_error_usd',quiet),
                          'mean_absolute_predicted_change_pct':avg('predicted_change_abs', [{'predicted_change_abs':abs(r['predicted_change_pct'])} for r in values]),
                          'mean_absolute_actual_change_pct':avg('actual_change_abs',[{'actual_change_abs':abs(r['actual_change_pct'])} for r in values])})
    out=root/'work/failure_analysis'
    out.mkdir(parents=True,exist_ok=True)
    with (out/'details.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'summary.json').write_text(json.dumps({'summaries':summaries,'final_test_evaluated':False,
         'method':'Large moves: top approximately 10% of absolute daily returns within each evaluation period, defined retrospectively for diagnosis only. Seeds share outcomes; not independent samples.'},indent=2),encoding='utf-8')
    for summary in summaries:
        print(summary['period_start'],summary['model'], 'extra MAE $',round(summary['mean_extra_error_usd'],2),
              'predicted/actual mean absolute daily move %',round(summary['mean_absolute_predicted_change_pct'],3),round(summary['mean_absolute_actual_change_pct'],3),
              'large-move model/baseline MAE $',round(summary['large_move_model_mae_usd'],2),round(summary['large_move_baseline_mae_usd'],2))


if __name__=='__main__': run(Path(__file__).resolve().parents[1])
