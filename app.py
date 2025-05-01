import requests
import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
from dateutil.relativedelta import relativedelta

# アプリタイトル
st.title("📈 株価予測アプリ")

# サイドバー
st.sidebar.write("""## 詳細情報""")
ticker = st.sidebar.text_input("ティッカーシンボルを入力してください", "AAPL")
start_date = st.sidebar.date_input("開始日", datetime.today() - relativedelta(years=3))
end_date = st.sidebar.date_input("終了日", datetime.today())

# Alpha Vantage APIの設定
api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]

def get_stock_alpha_vantage(stock, start, end, api_key):
    url = f"https://www.alphavantage.co/query"
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': stock,
        'apikey': api_key,
        'outputsize': 'full'
    }
    response = requests.get(url, params=params)
    data = response.json()

    # データが無い場合
    if "Time Series (Daily)" not in data:
        st.error(f"株価データが取得できませんでした。ティッカーやAPIキーをご確認ください。")
        return pd.DataFrame()

    # 日付をインデックスとして、DataFrameに変換
    df = pd.DataFrame(data['Time Series (Daily)']).T
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # 開始日と終了日の範囲をフィルタリング
    start = pd.to_datetime(start)  # start_date を Timestamp に変換
    end = pd.to_datetime(end)  # end_date を Timestamp に変換
    df = df[(df.index >= start) & (df.index <= end)]

    return df

if st.button('予測する'):

    # プログレスバーとステータステキスト
    progress_bar = st.progress(0)
    status_text = st.empty()

    # ステップ1：データ取得
    status_text.text('株価データを取得中...')
    data = get_stock_alpha_vantage(ticker, start_date, end_date, api_key)
    # データが空の場合のチェック
    if data.empty:
        st.error("""
                 株価データが取得できませんでした。
                 アクセス制限の可能性が高いです。時間を空けて試して下さい""")
        st.stop()

    time.sleep(0.3)
    progress_bar.progress(10)

    # ステップ2：特徴量作成
    status_text.text('特徴量を作成中...')
    data['Target'] = data['4. close'].shift(-1)  # 4. closeが株価の終値に対応
    features = ['1. open', '2. high', '3. low', '4. close', '5. volume']

    X = data[features].iloc[:-1]
    y = data['Target'].iloc[:-1]
    time.sleep(0.3)
    progress_bar.progress(25)

    # ステップ3：データ分割
    status_text.text('データをトレーニング・テストに分割中...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    time.sleep(0.3)
    progress_bar.progress(40)

    # ステップ4：モデル学習
    status_text.text('モデルを学習中...')
    dtrain = lgb.Dataset(X_train, y_train)
    dvalid = lgb.Dataset(X_test, y_test, reference=dtrain)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
    }

    feature_names = X_train.columns.tolist()

    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dvalid],
        feature_name=feature_names,
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=10)] 
        # , lgb.log_evaluation(0)
    )
    time.sleep(0.3)
    progress_bar.progress(60)

    # ステップ5：テストデータ予測
    status_text.text('テストデータを予測中...')
    predictions = model.predict(X_test)
    time.sleep(0.3)
    progress_bar.progress(70)

    # ステップ6：5営業日先予測
    status_text.text('5営業日先を予測中...')
    last_features = X.iloc[-1].values.reshape(1, -1)
    future_preds = []
    future_dates = []
    current_date = data.index[-1]

    for i in range(5):
        pred_price = model.predict(last_features)[0]
        future_preds.append(pred_price)
        current_date += pd.tseries.offsets.BDay(1)
        future_dates.append(current_date)

        last_features[0][features.index('1. open')] = pred_price
        last_features[0][features.index('2. high')] = pred_price
        last_features[0][features.index('3. low')] = pred_price
        last_features[0][features.index('4. close')] = pred_price

        progress_bar.progress(70 + (i + 1) * 5)
        time.sleep(0.3)

    # ステップ7：グラフ用データ準備
    status_text.text('グラフを作成中...')
    history_actual = pd.DataFrame({
        'Date': X_test.index,
        'Type': 'Actual',
        'Value': y_test.values
    })

    history_pred = pd.DataFrame({
        'Date': X_test.index,
        'Type': 'Predicted',
        'Value': predictions
    })

    future_pred = pd.DataFrame({
        'Date': future_dates,
        'Type': 'Future_Predicted',
        'Value': future_preds
    })

    plot_df = pd.concat([history_actual, history_pred, future_pred])
    time.sleep(0.3)
    progress_bar.progress(100)

    # ステップ8：グラフ描画
    status_text.text('予測完了！')

    base = alt.Chart(plot_df).encode(
        x='Date:T',
        y='Value:Q',
        color=alt.Color('Type:N', scale=alt.Scale(
            domain=['Actual', 'Predicted', 'Future_Predicted'],
            range=['blue', 'orange', 'red']
        )),
    )

    line_actual = base.transform_filter(
        alt.datum.Type == 'Actual'
    ).mark_line()

    line_pred = base.transform_filter(
        alt.datum.Type == 'Predicted'
    ).mark_line()

    line_future = base.transform_filter(
        alt.datum.Type == 'Future_Predicted'
    ).mark_line(strokeDash=[5, 5])

    chart = (line_actual + line_pred + line_future).properties(
        width=700,
        height=400,
        title='Actual vs Predicted (+5営業日 Future Prediction)'
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

    # ステップ9：未来予測結果のテーブル表示
    st.subheader("📄 5営業日分の予測結果")
    future_display = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': future_preds
    })
    st.dataframe(future_display)

