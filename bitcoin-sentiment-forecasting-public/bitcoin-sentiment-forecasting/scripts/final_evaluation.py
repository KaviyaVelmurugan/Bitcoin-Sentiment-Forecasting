"""One frozen holdout evaluation. Development changes after it need new future data."""
import argparse
import csv
import json
from datetime import date
from pathlib import Path
import numpy as np
from train_lstm import dataset, arrays
from test_volatility import prepare
from build_experiment import read_prices, make_examples, metrics


def prepare_final(root):
    _,_,manifest=dataset(root)  # verifies source hashes before building the final split
    prices,_=read_prices(root/'work/data/coinmetrics_btc.csv')
    with (root/'work/data/fear_greed_validated.csv').open() as handle:
        sentiment={date.fromisoformat(r['date_utc']):int(r['fng_value']) for r in csv.DictReader(handle)}
    rows=make_examples(prices,sentiment)
    cutoff=date.fromisoformat(manifest['split']['reserved_final_test']['first_target'])
    test=[r for r in rows if r['target_date']>=cutoff]
    train=[r for r in rows if r['target_date']<test[0]['origin_date']]
    assert len(test)==451 and train[-1]['target_date']<test[0]['origin_date']
    # Feature preparation does not score or inspect final forecast errors.
    for volatility in (False,True):
        x,v,y,*_ = prepare(train,test,prices) if volatility else arrays(train,test,False)
        assert np.isfinite(x).all() and np.isfinite(v).all() and np.isfinite(y).all()
    plan={'models':['persistence','price_lstm','price_volatility_lstm'],'seeds':[1,42,123],
          'epochs':20,'batch_size':32,'lstm_units':16,'learning_rate':.001,'shuffle':False,
          'window_days':10,'volatility_days':3,'target':'next-day log return converted to USD reference price',
          'primary_metric':'MAE in USD','primary_forecast':'arithmetic mean of three seed price forecasts; no best-seed selection',
          'train_count':len(train),'last_training_target':str(train[-1]['target_date']),
          'first_test_target':str(test[0]['target_date']),'last_test_target':str(test[-1]['target_date']),'test_count':len(test),
          'price_sha256':manifest['price_sha256'],'sentiment_sha256':manifest['sentiment_sha256']}
    out=root/'work/final_evaluation';out.mkdir(parents=True,exist_ok=True)
    plan_path=out/'frozen_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan: raise ValueError('Frozen plan differs; do not change it after final evaluation')
    if not plan_path.exists(): plan_path.write_text(json.dumps(plan,indent=2))
    return train,test,prices,plan,out


def run(root,prepare_only):
    train,test,prices,plan,out=prepare_final(root)
    if prepare_only:
        print(f"Frozen plan saved: {len(train)} training examples, {len(test)} final examples. No final forecasts or errors computed.")
        return
    if (out/'results.json').exists():
        print('Final evaluation already completed. View its saved results; no retraining performed.')
        return
    lock=out/'running.lock'
    with lock.open('x') as handle: handle.write('Final evaluation running')
    try:
        import tensorflow as tf
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(2)
        actual=np.array([r['target_price'] for r in test])
        baseline=np.array([r['current_price'] for r in test])
        predictions={'persistence':baseline};seed_scores=[]
        for volatility,name in ((False,'price_lstm'),(True,'price_volatility_lstm')):
            individual=[]
            for seed in plan['seeds']:
                tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed)
                x,v,y,ym,ys,*_ = prepare(train,test,prices) if volatility else arrays(train,test,False)
                model=tf.keras.Sequential([tf.keras.layers.Input(shape=x.shape[1:]),tf.keras.layers.LSTM(16),tf.keras.layers.Dense(1)])
                model.compile(optimizer=tf.keras.optimizers.Adam(.001),loss='mse')
                print(f'Final evaluation: {name}, seed {seed}',flush=True)
                model.fit(x,y,epochs=20,batch_size=32,shuffle=False,verbose=2)
                predicted=baseline*np.exp(model.predict(v,verbose=0).ravel()*ys+ym)
                if not np.isfinite(predicted).all(): raise ValueError('Nonfinite predictions')
                individual.append(predicted);seed_scores.append({'model':name,'seed':seed,**metrics(actual,predicted)})
            predictions[name]=np.mean(individual,axis=0)
        scores={name:metrics(actual,p) for name,p in predictions.items()}
        for values in scores.values(): values['improvement_pct']=100*(1-values['mae_usd']/scores['persistence']['mae_usd'])
        with (out/'predictions.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=['target_date','actual_price']+list(predictions));writer.writeheader()
            for i,row in enumerate(test):writer.writerow({'target_date':str(row['target_date']),'actual_price':actual[i],**{name:p[i] for name,p in predictions.items()}})
        report={'final_scores':scores,'individual_seed_scores':seed_scores,'plan':plan,'final_test_evaluated':True,
                'status':'Final period now inspected. Future model changes require genuinely new evaluation data.',
                'limitations':'One historical holdout, no significance estimate or trading profitability claim; source availability/revision assumptions remain.'}
        temporary=out/'results.tmp';temporary.write_text(json.dumps(report,indent=2));temporary.replace(out/'results.json')
        print('Finished. Choose Final evaluation on the Bitcoin page. This holdout is now inspected.')
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    run(Path(__file__).resolve().parents[1],args.prepare_only)
