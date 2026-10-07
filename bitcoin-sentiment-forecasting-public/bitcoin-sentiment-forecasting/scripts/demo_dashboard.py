"""Fictional offline interface demo; no private files or network access."""
import numpy as np
import pandas as pd


def demo_data():
    rng=np.random.default_rng(7)
    prices=50000*np.exp(np.cumsum(rng.normal(0,.018,90)))
    return pd.DataFrame({'Date':pd.date_range('2020-01-02',periods=89),
                         'Fictional actual price':prices[1:],'Persistence':prices[:-1],
                         'Illustrative forecast':prices[:-1]*np.exp(rng.normal(0,.002,89))})


def render(st):
    st.warning('SYNTHETIC DEMO: all prices, forecasts and news counts below are fictional. They demonstrate the interface, not real performance.')
    summary,prices,comparison,news=st.tabs(['Project summary','Demo prices','Demo comparisons','Demo sentiment'])
    data=demo_data()
    with summary:
        st.write('Can sequence models improve next-day Bitcoin forecasts beyond using the latest price?')
        st.write('The historical experiment compared prices, Fear and Greed and volatility. Persistence achieved the lowest final MAE. Read docs/final-report.md for the actual findings.')
        st.caption('CryptoPulse supplies the sentiment pipeline; SignalGuard informs evaluation and failure analysis. News sentiment was not used in the historical forecasting models.')
    with prices: st.line_chart(data.set_index('Date')['Fictional actual price'])
    with comparison:
        rows=[]
        for name in ['Persistence','Illustrative forecast']:
            error=data[name]-data['Fictional actual price']
            rows.append({'Demo forecast':name,'Demo MAE ($)':float(error.abs().mean()),'Demo RMSE ($)':float(np.sqrt((error**2).mean()))})
        st.dataframe(pd.DataFrame(rows),hide_index=True)
        st.line_chart(data.set_index('Date'))
        st.caption('Synthetic display calculations; these are not trained LSTM results.')
    with news:
        st.bar_chart(pd.Series({'Positive':8,'Neutral':6,'Negative':4},name='Fictional article counts'))
        st.caption('No provider text, private news files or API key is loaded.')
