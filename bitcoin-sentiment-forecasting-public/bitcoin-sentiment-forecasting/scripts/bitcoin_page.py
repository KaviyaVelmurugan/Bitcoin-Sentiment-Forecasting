"""Local research dashboard; no API key, scraping, training, or trading."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from demo_dashboard import render

root=Path(__file__).resolve().parents[1]
st.set_page_config(page_title='Bitcoin Sentiment Forecasting',page_icon='₿',layout='wide')
st.title('Bitcoin Sentiment Forecasting')
st.caption('Your local research dashboard · historical experiments and news processing')
local_available=(root/'work/data/coinmetrics_btc.csv').exists()
choices=['Synthetic demo']+(['Local research results'] if local_available else [])
mode=st.sidebar.radio('Data mode',choices,index=1 if local_available else 0)
if mode=='Synthetic demo':
    render(st)
    st.stop()
st.warning('Price history ends in May 2026. This page does not show a live Bitcoin price or a current forecast.')
final_complete=(root/'work/final_evaluation/results.json').exists()
test_status='Final evaluation has been completed; that period is now inspected.' if final_complete else 'Final test remains reserved.'
st.caption(test_status)
tab1,tab2,tab3,tab4=st.tabs(['Bitcoin prices','Forecast comparisons','News sentiment','Why forecasts failed'])
with tab1:
    prices=pd.read_csv(root/'work/data/coinmetrics_btc.csv',usecols=['time','PriceUSD']).dropna()
    prices['time']=pd.to_datetime(prices['time'])
    st.metric('Last recorded reference price',f"${prices.iloc[-1]['PriceUSD']:,.2f}")
    st.caption(f"Recorded date: {prices.iloc[-1]['time'].date()} · USD reference price, not an exchange execution quote")
    st.line_chart(prices.set_index('time')['PriceUSD'])
    st.markdown('Price data: [Coin Metrics](https://github.com/coinmetrics/data), [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Local noncommercial research.')
with tab2:
    options={'Simple baseline and regression':root/'work/experiment','LSTM comparison':root/'work/lstm','Earlier-period comparisons':root/'work/period_checks','Volatility comparison':root/'work/volatility_checks','Final evaluation':root/'work/final_evaluation'}
    selection=st.selectbox('Choose experiment',list(options))
    folder=options[selection]
    report_path=folder/('development_results.json' if selection=='Simple baseline and regression' else 'results.json')
    if selection=='Final evaluation':
        if not report_path.exists():
            st.info('The frozen final evaluation is ready. Run final_evaluation.py once in VS Code to generate the results.')
        else:
            report=json.loads(report_path.read_text())
            st.caption(f"Final target dates: {report['plan']['first_test_target']} to {report['plan']['last_test_target']}")
            st.dataframe(pd.DataFrame(report['final_scores']).T.rename(columns={'mae_usd':'Average error ($)','rmse_usd':'RMSE ($)','improvement_pct':'Improvement vs baseline (%)'}))
            st.caption('LSTM forecasts average the three fixed seeds. No best seed was selected.')
            final_predictions=pd.read_csv(folder/'predictions.csv');final_predictions['target_date']=pd.to_datetime(final_predictions['target_date'])
            st.line_chart(final_predictions.set_index('target_date'))
            with st.expander('Individual seed results'): st.dataframe(pd.DataFrame(report['individual_seed_scores']),hide_index=True)
            st.warning(report['status'])
            st.caption(report['limitations'])
    elif selection in ('Earlier-period comparisons','Volatility comparison'):
        st.caption('Each model learns from earlier dates and is evaluated on the following six months. '+test_status)
        if not report_path.exists():
            st.info('Run this experiment once in VS Code. Results appear here when all tests finish.')
            script='test_volatility.py' if selection=='Volatility comparison' else 'test_periods.py'
            st.code('python scripts/'+script,language='powershell')
        else:
            report=json.loads(report_path.read_text())
            records=pd.DataFrame(report['records'])
            records['Period']=records['period_start']+' to '+records['period_end']
            st.write('Three periods and three random seeds per LSTM. The baseline is unchanged.')
            if selection=='Volatility comparison': st.caption('New feature: recent three-day volatility, calculated using only returns known at each input date. Original experiment results are retained.')
            summary=records.groupby(['Period','model'],as_index=False).agg(Average_error_USD=('mae_usd','mean'),RMSE_USD=('rmse_usd','mean'),Improvement_percent=('improvement_pct','mean'))
            st.dataframe(summary,hide_index=True)
            st.caption('LSTM errors are averaged across seeds. Positive improvement means lower error than persistence. Seed spread is not a confidence interval.')
            period=st.selectbox('Inspect period',records['Period'].unique())
            detail=records[records['Period']==period]
            st.dataframe(detail[['model','seed','training_examples','evaluation_examples','mae_usd','rmse_usd','improvement_pct']],hide_index=True)
            forecasts=pd.read_csv(folder/'predictions.csv')
            model=st.selectbox('Model',[name for name in records['model'].unique() if name!='persistence'])
            seed=st.selectbox('Random seed',report['seeds'])
            rows=forecasts[(forecasts['period_start']==detail.iloc[0]['period_start'])&(forecasts['model']==model)&(forecasts['seed']==seed)].copy()
            rows['target_date']=pd.to_datetime(rows['target_date'])
            st.line_chart(rows.set_index('target_date')[['actual_price','persistence','predicted_price']])
            rows['absolute_error_usd']=(rows['predicted_price']-rows['actual_price']).abs()
            st.write('Largest errors in this period')
            st.dataframe(rows.nlargest(10,'absolute_error_usd')[['target_date','actual_price','predicted_price','absolute_error_usd']],hide_index=True)
    elif report_path.exists():
        report=json.loads(report_path.read_text())
        st.caption('Development validation results. These are historical forecasts. '+test_status)
        st.dataframe(pd.DataFrame(report['validation_only_scores']).T.rename(columns={'mae_usd':'Average error ($)','rmse_usd':'RMSE ($)','mae_improvement_vs_persistence_pct':'Improvement vs baseline (%)'}))
        predictions=pd.read_csv(folder/'validation_predictions.csv')
        predictions['target_date']=pd.to_datetime(predictions['target_date'])
        columns=[c for c in predictions if c not in ('origin_date','target_date','latest_fng_date')]
        st.line_chart(predictions.set_index('target_date')[columns])
        chosen=st.selectbox('Inspect largest errors',[c for c in columns if c not in ('actual_price',)],key='errors')
        actual='actual_price'
        predictions['absolute_error_usd']=(predictions[chosen]-predictions[actual]).abs()
        st.dataframe(predictions.nlargest(10,'absolute_error_usd')[['target_date',actual,chosen,'absolute_error_usd']])
    else:
        st.info('LSTM results will appear after you run train_lstm.py in VS Code.')
with tab3:
    path=root/'work/private_news/processed/processing_summary.json'
    if path.exists():
        summary=json.loads(path.read_text())
        st.metric('Articles with Bitcoin sentiment scores',summary['bitcoin_scored'])
        st.bar_chart(pd.Series(summary['labels'],name='Articles'))
        st.write(f"Unresolved or ambiguous: {summary['unresolved_or_ambiguous']}")
        st.caption('Language sentiment is not a prediction of market direction. News scores are not yet included in the forecasting models. Provider text remains in private files.')
    else: st.info('Process the collected news first.')
with tab4:
    failure=root/'work/failure_analysis'
    if not (failure/'summary.json').exists():
        st.info('Failure analysis appears after the earlier-period results are analyzed.')
    else:
        report=json.loads((failure/'summary.json').read_text())
        st.write('Compare each LSTM with simply using the previous price.')
        period=st.selectbox('Evaluation period',sorted({r['period_start'] for r in report['summaries']}),key='failure_period')
        model=st.selectbox('Forecast model',['price_lstm','price_sentiment_lstm'],key='failure_model')
        item=next(r for r in report['summaries'] if r['period_start']==period and r['model']==model)
        a,b,c=st.columns(3)
        a.metric('Extra average error vs baseline',f"${item['mean_extra_error_usd']:,.2f}")
        b.metric('Mean predicted daily move',f"{item['mean_absolute_predicted_change_pct']:.2f}%")
        c.metric('Mean actual daily move',f"{item['mean_absolute_actual_change_pct']:.2f}%")
        st.caption('Daily move cards show absolute magnitudes. Positive extra error means the model was worse; negative means better. Summaries average across three seeds.')
        st.dataframe(pd.DataFrame({'Days':['Largest price moves','Other days'],
                                  'Model average error ($)':[item['large_move_model_mae_usd'],item['other_day_model_mae_usd']],
                                  'Baseline average error ($)':[item['large_move_baseline_mae_usd'],item['other_day_baseline_mae_usd']]}),hide_index=True)
        details=pd.read_csv(failure/'details.csv')
        seed=st.selectbox('Training seed',sorted(details['seed'].unique()),key='failure_seed')
        details=details[(details['period_start']==period)&(details['model']==model)&(details['seed']==seed)].copy()
        view=st.radio('Inspect',['Biggest prediction errors','Most extra error vs baseline','Largest market moves'],horizontal=True)
        sort={'Biggest prediction errors':'model_error_usd','Most extra error vs baseline':'extra_error_usd','Largest market moves':'actual_change_pct'}[view]
        if sort=='actual_change_pct': details['sort_move']=details[sort].abs();sort='sort_move'
        st.dataframe(details.nlargest(10,sort)[['target_date','actual_change_pct','predicted_change_pct','model_error_usd','baseline_error_usd','extra_error_usd']],hide_index=True)
        st.caption(report['method'])
        st.info('This diagnoses development errors; it does not establish why markets moved or measure trading profit. '+test_status)
