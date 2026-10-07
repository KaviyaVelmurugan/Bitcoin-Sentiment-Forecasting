"""Run development LSTMs only; reserved final test never enters training/evaluation."""
import argparse
import csv
import hashlib
import json
from datetime import date
from pathlib import Path
import numpy as np
from build_experiment import read_prices, make_examples, metrics


def dataset(root):
    prices, _ = read_prices(root / 'work/data/coinmetrics_btc.csv')
    with (root / 'work/data/fear_greed_validated.csv').open() as handle:
        sentiment = {date.fromisoformat(r['date_utc']): int(r['fng_value']) for r in csv.DictReader(handle)}
    rows = make_examples(prices, sentiment)
    a, b = int(len(rows)*.7), int(len(rows)*.85)
    train, validation = rows[:a], rows[a:b]
    train = [r for r in train if r['target_date'] < validation[0]['origin_date']]
    validation = [r for r in validation if r['target_date'] < rows[b]['origin_date']]
    manifest = json.loads((root / 'work/experiment/development_results.json').read_text())
    assert hashlib.sha256((root / 'work/data/coinmetrics_btc.csv').read_bytes()).hexdigest() == manifest['price_sha256'], 'Price archive changed: review split before training'
    assert hashlib.sha256((root / 'work/data/fear_greed_validated.csv').read_bytes()).hexdigest() == manifest['sentiment_sha256'], 'Sentiment archive changed: review split before training'
    assert len(train) == manifest['split']['train']['examples']
    assert len(validation) == manifest['split']['validation']['examples']
    return train, validation, manifest


def arrays(train, validation, combined):
    def features(rows):
        return np.array([np.column_stack([r['price_features'], r['fng_features']]) if combined else np.array(r['price_features'])[:,None] for r in rows], dtype=np.float32)
    x, v = features(train), features(validation)
    mean, std = x.mean(axis=(0,1)), x.std(axis=(0,1))
    std[std == 0] = 1
    y = np.array([r['target_return'] for r in train], dtype=np.float32)
    ym, ys = float(y.mean()), float(y.std())
    if ys == 0: raise ValueError('Constant training target')
    return (x-mean)/std, (v-mean)/std, (y-ym)/ys, ym, ys, mean, std


def run(root, epochs, prepare_only):
    train, validation, manifest = dataset(root)
    for combined in (False, True):
        x, v, y, *_ = arrays(train, validation, combined)
        assert x.shape == (2099, 10, 2 if combined else 1)
        assert np.isfinite(x).all() and np.isfinite(v).all() and np.isfinite(y).all()
    if prepare_only:
        print('Prepared both datasets: 2,099 training and 449 validation examples. Final test excluded. No model trained.')
        return
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    output = root / 'work/lstm'
    output.mkdir(parents=True, exist_ok=True)
    actual = np.array([r['target_price'] for r in validation])
    predictions = {'persistence': np.array([r['current_price'] for r in validation])}
    for combined, name in ((False,'price_lstm'), (True,'price_sentiment_lstm')):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)
        x, v, y, ym, ys, mean, std = arrays(train, validation, combined)
        model = tf.keras.Sequential([tf.keras.layers.Input(shape=x.shape[1:]),
                                    tf.keras.layers.LSTM(16), tf.keras.layers.Dense(1)])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001), loss='mse')
        print(f'Training {name}: {epochs} epochs, seed 42')
        history = model.fit(x,y,epochs=epochs,batch_size=32,shuffle=False,verbose=2)
        returns = model.predict(v,verbose=0).ravel()*ys+ym
        prices = predictions['persistence']*np.exp(returns)
        if not np.isfinite(prices).all(): raise ValueError('Nonfinite predictions')
        predictions[name] = prices
        model.save(output / (name+'.keras'))
        (output / (name+'_preprocessing.json')).write_text(json.dumps({'feature_mean':mean.tolist(),'feature_std':std.tolist(),'target_mean':ym,'target_std':ys,'history':history.history},indent=2))
    scores = {name:metrics(actual,pred) for name,pred in predictions.items()}
    for values in scores.values():
        values['mae_improvement_vs_persistence_pct'] = 100*(1-values['mae_usd']/scores['persistence']['mae_usd'])
    report = {'validation_only_scores':scores,'split':manifest['split'],'final_test_evaluated':False,
              'epochs':epochs,'seed':42,'window_days':10,'sentiment_lag_days':1,
              'price_sha256':manifest['price_sha256'],'sentiment_sha256':manifest['sentiment_sha256'],
              'status':'Historical development experiment; single seed; not current forecasting or final performance'}
    with (output / 'validation_predictions.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['target_date','actual_price']+list(predictions))
        writer.writeheader()
        for i,r in enumerate(validation): writer.writerow({'target_date':r['target_date'],'actual_price':actual[i],**{name:p[i] for name,p in predictions.items()}})
    (output / 'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if not 1<=args.epochs<=100: parser.error('epochs must be 1–100')
    run(Path(__file__).resolve().parents[1],args.epochs,args.prepare_only)
