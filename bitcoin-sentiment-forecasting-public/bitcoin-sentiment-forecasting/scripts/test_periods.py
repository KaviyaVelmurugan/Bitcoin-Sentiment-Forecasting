"""Expanding-history development checks. Reserved final test is excluded."""
import argparse
import csv
import json
from datetime import date
from pathlib import Path
import numpy as np
from train_lstm import dataset, arrays
from build_experiment import metrics

PERIODS = [('2021-01-01','2021-06-30'),('2021-07-01','2021-12-31'),('2022-01-01','2022-06-30')]


def folds(root):
    development, _, manifest = dataset(root)
    result = []
    for start, end in PERIODS:
        start, end = date.fromisoformat(start), date.fromisoformat(end)
        evaluation = [r for r in development if start <= r['target_date'] <= end]
        train = [r for r in development if r['target_date'] < evaluation[0]['origin_date']]
        assert len(train) > 100 and len(evaluation) > 50
        assert max(r['target_date'] for r in train) < evaluation[0]['origin_date']
        assert end < date.fromisoformat(manifest['split']['reserved_final_test']['first_target'])
        result.append((str(start), str(end), train, evaluation))
    return result, manifest


def run(root, epochs, seeds, prepare_only):
    periods, manifest = folds(root)
    for start, end, train, evaluation in periods:
        for combined in (False, True):
            x, v, y, *_ = arrays(train, evaluation, combined)
            assert np.isfinite(x).all() and np.isfinite(v).all() and np.isfinite(y).all()
        print(f'{start} to {end}: {len(train)} earlier training examples, {len(evaluation)} evaluation examples', flush=True)
    if prepare_only:
        print('All three periods checked. No model trained; reserved final test excluded.')
        return
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    out = root/'work/period_checks'
    out.mkdir(parents=True, exist_ok=True)
    records, prediction_rows = [], []
    for start, end, train, evaluation in periods:
        actual = np.array([r['target_price'] for r in evaluation])
        baseline = np.array([r['current_price'] for r in evaluation])
        base_metrics = metrics(actual, baseline)
        common = {'period_start':start,'period_end':end,'training_examples':len(train),'evaluation_examples':len(evaluation),
                  'training_last_target':str(train[-1]['target_date'])}
        records.append({**common,'model':'persistence','seed':None,**base_metrics,'improvement_pct':0.0})
        for seed in seeds:
            for combined, name in ((False,'price_lstm'),(True,'price_sentiment_lstm')):
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(seed)
                x, v, y, ym, ys, _, _ = arrays(train,evaluation,combined)
                model=tf.keras.Sequential([tf.keras.layers.Input(shape=x.shape[1:]),tf.keras.layers.LSTM(16),tf.keras.layers.Dense(1)])
                model.compile(optimizer=tf.keras.optimizers.Adam(.001),loss='mse')
                print(f'{start}: {name}, seed {seed}',flush=True)
                model.fit(x,y,epochs=epochs,batch_size=32,shuffle=False,verbose=2)
                predicted=baseline*np.exp(model.predict(v,verbose=0).ravel()*ys+ym)
                if not np.isfinite(predicted).all(): raise ValueError('Nonfinite predictions')
                score=metrics(actual,predicted)
                records.append({**common,'model':name,'seed':seed,**score,'improvement_pct':100*(1-score['mae_usd']/base_metrics['mae_usd'])})
                for i,row in enumerate(evaluation):
                    prediction_rows.append({'period_start':start,'model':name,'seed':seed,'target_date':str(row['target_date']),
                                            'actual_price':actual[i],'persistence':baseline[i],'predicted_price':predicted[i]})
    # Only completed experiments are published to the dashboard; partial runs remain invisible.
    with (out/'predictions.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(prediction_rows[0])); writer.writeheader(); writer.writerows(prediction_rows)
    report={'records':records,'epochs':epochs,'seeds':seeds,'final_test_evaluated':False,
            'price_sha256':manifest['price_sha256'],'sentiment_sha256':manifest['sentiment_sha256'],
            'status':'Earlier development checks; seed variation is not a statistical confidence interval',
            'selection_note':'Periods chosen after inspecting the initial validation result; not independent final evidence'}
    temporary=out/'results.tmp'
    temporary.write_text(json.dumps(report,indent=2)); temporary.replace(out/'results.json')
    print('Finished. Refresh the Bitcoin page and choose Earlier-period comparisons.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    run(Path(__file__).resolve().parents[1],20,[1,42,123],args.prepare_only)
