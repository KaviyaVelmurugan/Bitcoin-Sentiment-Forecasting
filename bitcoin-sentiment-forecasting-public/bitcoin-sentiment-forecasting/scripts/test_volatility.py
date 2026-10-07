"""Add causal three-day realized volatility; preserve existing comparison runs."""
import argparse
import csv
import json
from datetime import timedelta
from pathlib import Path
import numpy as np
from test_periods import folds
from train_lstm import arrays
from build_experiment import read_prices, metrics


def features(rows, prices):
    result=[]
    for row in rows:
        days=[row['origin_date']-timedelta(days=i) for i in range(9,-1,-1)]
        vol=[]
        for day in days:
            returns=[np.log(prices[day-timedelta(days=j)]/prices[day-timedelta(days=j+1)]) for j in range(3)]
            vol.append(float(np.std(returns,ddof=0)))
        result.append(np.column_stack([row['price_features'],vol]))
    return np.asarray(result,dtype=np.float32)


def prepare(train, evaluation, prices):
    x,v=features(train,prices),features(evaluation,prices)
    mean,std=x.mean(axis=(0,1)),x.std(axis=(0,1))
    std[std==0]=1
    _,_,y,ym,ys,_,_=arrays(train,evaluation,False)
    return (x-mean)/std,(v-mean)/std,y,ym,ys


def run(root,prepare_only):
    periods,manifest=folds(root)
    prices,_=read_prices(root/'work/data/coinmetrics_btc.csv')
    base=root/'work/period_checks'
    previous=json.loads((base/'results.json').read_text())
    if previous['epochs']!=20 or previous['seeds']!=[1,42,123]: raise ValueError('Comparison settings do not match')
    for field in ('price_sha256','sentiment_sha256'):
        if previous[field]!=manifest[field]: raise ValueError('Source data differs from original experiment')
    for start,end,train,evaluation in periods:
        x,v,y,*_=prepare(train,evaluation,prices)
        assert x.shape==(len(train),10,2) and np.isfinite(x).all() and np.isfinite(v).all()
        # Change all prices after this origin; its volatility feature must not change.
        row=evaluation[0]
        changed={day:(value*2 if day>row['origin_date'] else value) for day,value in prices.items()}
        assert np.array_equal(features([row],prices),features([row],changed))
        print(f'{start}: checked volatility features and future-price exclusion',flush=True)
    if prepare_only:
        print('Ready. No training run; final test excluded.')
        return
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    records=list(previous['records'])
    prediction_rows=list(csv.DictReader((base/'predictions.csv').open()))
    for start,end,train,evaluation in periods:
        actual=np.array([r['target_price'] for r in evaluation])
        baseline=np.array([r['current_price'] for r in evaluation])
        baseline_error=metrics(actual,baseline)['mae_usd']
        for seed in previous['seeds']:
            tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed)
            x,v,y,ym,ys=prepare(train,evaluation,prices)
            model=tf.keras.Sequential([tf.keras.layers.Input(shape=(10,2)),tf.keras.layers.LSTM(16),tf.keras.layers.Dense(1)])
            model.compile(optimizer=tf.keras.optimizers.Adam(.001),loss='mse')
            print(f'{start}: price + volatility LSTM, seed {seed}',flush=True)
            model.fit(x,y,epochs=20,batch_size=32,shuffle=False,verbose=2)
            predicted=baseline*np.exp(model.predict(v,verbose=0).ravel()*ys+ym)
            if not np.isfinite(predicted).all(): raise ValueError('Nonfinite predictions')
            score=metrics(actual,predicted)
            records.append({'period_start':start,'period_end':end,'training_examples':len(train),'evaluation_examples':len(evaluation),
                            'training_last_target':str(train[-1]['target_date']),'model':'price_volatility_lstm','seed':seed,**score,
                            'improvement_pct':100*(1-score['mae_usd']/baseline_error)})
            for i,row in enumerate(evaluation):
                prediction_rows.append({'period_start':start,'model':'price_volatility_lstm','seed':seed,'target_date':str(row['target_date']),
                                        'actual_price':actual[i],'persistence':baseline[i],'predicted_price':predicted[i]})
    out=root/'work/volatility_checks';out.mkdir(parents=True,exist_ok=True)
    with (out/'predictions.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(prediction_rows[0]));writer.writeheader();writer.writerows(prediction_rows)
    report={**previous,'records':records,'volatility_definition':'Population standard deviation of last three daily log returns at each input time step',
            'selection_note':'Volatility feature selected after inspecting development failures; not independent final evidence'}
    temporary=out/'results.tmp';temporary.write_text(json.dumps(report,indent=2));temporary.replace(out/'results.json')
    print('Finished. Refresh Bitcoin page and choose Volatility comparison.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    run(Path(__file__).resolve().parents[1],args.prepare_only)
