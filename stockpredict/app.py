from flask import Flask, request, jsonify, render_template
import yfinance as yf
import pandas as pd
import numpy as np
import json
from tensorflow.keras.models import load_model
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
import datetime
import os
from sklearn.preprocessing import MinMaxScaler
import base64
from io import BytesIO
import tensorflow as tf
import logging
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend suitable for web servers
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import mplfinance as mpf
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates

# Disable GPU if not needed
tf.config.set_visible_devices([], 'GPU')

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to the models directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

tickers = ["XOM", "CVX", "COP", "NEE", "ENB", "SLB", "KMI", "EOG","SHEL", "TTE"]

# Load all models at startup
try:
    sarimax_models = {}
    lstm_models = {}
    rnn_models = {}

    # Load all SARIMAX models
    for file in os.listdir(MODELS_DIR):
        if file.startswith("SARIMAX_model_") and file.endswith(".pkl"):
            ticker = file.replace("SARIMAX_model_", "").replace(".pkl", "").upper()
            if ticker in [t.upper() for t in tickers]:
                sarimax_models[ticker] = SARIMAXResults.load(os.path.join(MODELS_DIR, file))

    # Load all LSTM models
    for file in os.listdir(MODELS_DIR):
        if file.startswith("LSTM_model_") and file.endswith(".keras"):
            ticker = file.replace("LSTM_model_", "").replace(".keras", "").upper()
            if ticker in [t.upper() for t in tickers]:
                lstm_models[ticker] = load_model(os.path.join(MODELS_DIR, file))

    # Load all RNN models
    for file in os.listdir(MODELS_DIR):
        if file.startswith("RNN_model_") and file.endswith(".keras"):
            ticker = file.replace("RNN_model_", "").replace(".keras", "").upper()
            if ticker in [t.upper() for t in tickers]:
                rnn_models[ticker] = load_model(os.path.join(MODELS_DIR, file))

    logger.info("All models loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models: {e}")

def fetch_historical_data(symbol, num_past_days):
    """
    Fetch historical stock data using yfinance.
    """
    try:
        end_date = datetime.datetime.today()
        start_date = end_date - datetime.timedelta(days=num_past_days * 2)
        df = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        if df.empty:
            return None
        df.sort_index(inplace=True)
        df = df.tail(num_past_days)
        return df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None

def generate_candlestick_chart(data):
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'])])
    fig.update_layout(title="Candlestick Chart", xaxis_rangeslider_visible=False)
    return fig.to_html(full_html=False)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    stock_details = None
    error = None
    stock = "XOM"
    start_date = None
    end_date = None
    data_desc = None
    line_chart_path = None
    ema_chart_path = None
    sma_chart_path = None
    plot_macd_path = None
    rsi_chart_path = None

    if request.method == 'POST':

        stock = request.form.get('stock', 'XOM')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')


        if not start_date or not end_date:
            error = "Please provide both start and end dates."
        else:
            try:
       
                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date)
                df = yf.download(stock, start=start, end=end)

                if df.empty:
                    error = f"No data found for {stock} between {start_date} and {end_date}."
                else:
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

       
                    df.index.name = 'Date' 
                    stock_details = df.reset_index().to_html(
                        classes='table table-bordered table-striped',
                        index=False
                    )

     
                    data_desc = df.describe().to_html(
                        classes='table table-bordered table-striped'
                    )
                    
                    # Vẽ biểu đồ đường
                    line_chart_path = f'static/{stock}_line_chart.png'
                    plt.figure(figsize=(12, 6))
                    plt.plot(df.index, df['Close'], label='Close Price', color='blue')
                    plt.title(f'{stock} Close Price')
                    plt.xlabel('Date')
                    plt.ylabel('Price')
                    plt.legend()
                    plt.grid()
                    plt.savefig(line_chart_path)
                    plt.close()

                    # EMA
                    df['EMA_20'] = df['Close'].ewm(span=20).mean()
                    df['EMA_50'] = df['Close'].ewm(span=50).mean()
                    ema_chart_path = f"static/ema_{stock}.png"
                    plt.figure(figsize=(12, 6))
                    plt.plot(df['Close'], label='Closing Price')
                    plt.plot(df['EMA_20'], label='EMA 20', color='green')
                    plt.plot(df['EMA_50'], label='EMA 50', color='red')
                    plt.legend()
                    plt.title(f"EMA 20 & 50 for {stock}")
                    plt.savefig(ema_chart_path)
                    plt.close()

                    # SMA
                    df['SMA_20'] = df['Close'].rolling(window=20).mean()
                    df['SMA_50'] = df['Close'].rolling(window=50).mean()
                    sma_chart_path = f"static/sma_{stock}.png"
                    plt.figure(figsize=(12, 6))
                    plt.plot(df['Close'], label='Closing Price')
                    plt.plot(df['SMA_20'], label='SMA 20', color='green')
                    plt.plot(df['SMA_50'], label='SMA 50', color='red')
                    plt.legend()
                    plt.title(f"SMA 20 & 50 for {stock}")
                    plt.savefig(sma_chart_path)
                    plt.close                   

                    # RSI
                    delta = df['Close'].diff(1)
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    avg_gain = gain.rolling(window=14).mean()
                    avg_loss = loss.rolling(window=14).mean()
                    rs = avg_gain / avg_loss
                    df['RSI'] = 100 - (100 / (1 + rs))

                    # MACD Calculation
                    short_ema = df.Close.ewm(span=12, adjust=False).mean()
                    long_ema = df.Close.ewm(span=26, adjust=False).mean()
                    macd = short_ema - long_ema
                    signal = macd.ewm(span=9, adjust=False).mean()

                    fig_macd, ax_macd = plt.subplots(figsize=(12, 6))
                    ax_macd.plot(macd, label="MACD", color='b')
                    ax_macd.plot(signal, label="Signal Line", color='r')
                    ax_macd.set_title("MACD and Signal Line")
                    ax_macd.set_xlabel("Time")
                    ax_macd.set_ylabel("Value")
                    ax_macd.legend()
                    plot_macd_path = "static/macd.png"
                    fig_macd.savefig(plot_macd_path)
                    plt.close(fig_macd)

                    # Vẽ biểu đồ RSI
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(df['RSI'], label='RSI (Purple)', color='purple')
                    ax.axhline(70, linestyle='--', color='red', label='Overbought (70)')
                    ax.axhline(30, linestyle='--', color='green', label='Oversold (30)')
                    ax.set_title("RSI (Relative Strength Index)")
                    ax.set_xlabel("Time")
                    ax.set_ylabel("RSI")
                    ax.legend(loc='best')  # Thêm chú thích
                    rsi_chart_path = "static/rsi.png"
                    fig.savefig(rsi_chart_path)
                    plt.close(fig)

            except Exception as e:
                error = f"An error occurred: {str(e)}"
        
        return render_template('dashboard.html',
                               stock=stock,
                               start_date=start_date,
                               end_date=end_date,  
                               stock_details=stock_details,
                               data_desc=data_desc,
                               line_chart_path=line_chart_path,
                               ema_chart_path=ema_chart_path,
                               plot_macd_path=plot_macd_path,
                               sma_chart_path=sma_chart_path,
                               rsi_chart_path=rsi_chart_path,
                               error=error) 

    return render_template('dashboard.html')


@app.route('/forecast')
def forecast_page():
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    return render_template('forecast.html', current_date=current_date)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data
        stock_symbol = request.form.get('stock_symbol', '').upper()
        num_past_days = int(request.form.get('num_past_days', 10))
        n_steps = 10

        # Validate num_past_days
        if num_past_days <= n_steps or num_past_days > 500:
            return jsonify({'error': 'Số ngày quá khứ phải lớn hơn 25 và không vượt quá 500 ngày.'}), 400

        # Fetch historical stock data
        historical_data = fetch_historical_data(stock_symbol, num_past_days)
        if historical_data is None or len(historical_data) < num_past_days:
            return jsonify({'error': f'Không có đủ dữ liệu cho {stock_symbol} trong {num_past_days} ngày gần nhất.'}), 400

        data_for_prediction = historical_data.tail(num_past_days)
        actual_last_price = data_for_prediction['Close'].iloc[-1]

        def create_plot_and_response(plot_dates, plot_prices, next_day_prediction, r2, mape, rmse, model_name, lower_ci=None, upper_ci=None):
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(plot_dates[:-1], plot_prices[:-1], label="Giá Đóng Cửa")
            ax.plot(plot_dates[-2:], plot_prices[-2:], linestyle='--', label="Giá Dự Đoán Ngày Tiếp Theo")
            ax.scatter(plot_dates[-1], next_day_prediction, color='red', label="Giá Dự Đoán")

            if lower_ci is not None and upper_ci is not None:
                ax.fill_between([plot_dates[-1], plot_dates[-1]], lower_ci, upper_ci, color='pink', alpha=0.3, label="Khoảng Tin Cậy 95%")

            ax.set_xlabel("Ngày")
            ax.set_ylabel("Giá (VND)")
            ax.legend()
            plt.xticks(rotation=45)
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plot_base64 = base64.b64encode(buf.getvalue()).decode()
            plt.close(fig)

            labels = [date.strftime('%Y-%m-%d') for date in plot_dates]

            return {
                'model': model_name,
                'labels': labels,
                'predicted_prices': [float(round(price, 2)) for price in plot_prices],
                'predicted_price': float(round(next_day_prediction, 2)),
                'metrics': {
                    'R2_Score':  f"{round(r2 * 100, 2)}%",  
                    'MAPE': float(round(mape, 4)),
                    'RMSE': float(round(rmse, 4)),
                },
                'plot_url': f'data:image/png;base64,{plot_base64}'
            }

        # -----------------------------
        # SARIMAX
        # -----------------------------
        sarimax_result = None
        if stock_symbol in sarimax_models:
            try:
                model = sarimax_models[stock_symbol]
                num_past_days_eval = len(data_for_prediction)
                train_length = len(model.data.endog)

                start_eval = train_length - num_past_days_eval
                end_eval = train_length - 1

                prediction = model.get_prediction(start=start_eval, end=end_eval)
                pred_mean = prediction.predicted_mean
                pred_mean.index = data_for_prediction.index

                r2_sarimax = r2_score(data_for_prediction['Close'], pred_mean)
                r2_sarimax = max(r2_sarimax, 0.0)

                forecast = model.get_forecast(steps=1)
                next_day_prediction_sarimax = float(forecast.predicted_mean.iloc[0])
                confidence_intervals = forecast.conf_int().iloc[0]
                lower_ci_sarimax = float(confidence_intervals.iloc[0])
                upper_ci_sarimax = float(confidence_intervals.iloc[1])

                mae_sarimax = mean_absolute_error([actual_last_price], [next_day_prediction_sarimax])
                mse_sarimax = mean_squared_error([actual_last_price], [next_day_prediction_sarimax])
                rmse_sarimax = np.sqrt(mse_sarimax)
                mape_sarimax = mean_absolute_percentage_error([actual_last_price], [next_day_prediction_sarimax]) * 100

                close_series = data_for_prediction['Close']
                last_date = close_series.index[-1]
                next_day_date = last_date + datetime.timedelta(days=1)

                plot_dates_sarimax = close_series.index.tolist()
                plot_prices_sarimax = close_series.squeeze().tolist()
                plot_dates_sarimax.append(next_day_date)
                plot_prices_sarimax.append(next_day_prediction_sarimax)

                sarimax_result = create_plot_and_response(
                    plot_dates_sarimax, plot_prices_sarimax, next_day_prediction_sarimax,
                    r2_sarimax, mape_sarimax, rmse_sarimax, "SARIMAX",
                    lower_ci=lower_ci_sarimax, upper_ci=upper_ci_sarimax
                )
            except Exception as e:
                logger.error(f"SARIMAX error: {e}")

        # -----------------------------
        # LSTM
        # -----------------------------
        lstm_result = None
        if stock_symbol in lstm_models:
            try:
                lstm_model_for_ticker = lstm_models[stock_symbol]
                data_lstm = data_for_prediction['Close'].values.reshape(-1, 1)
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(data_lstm)

                X_all = []
                for i in range(n_steps, len(scaled_data)):
                    X_all.append(scaled_data[i - n_steps:i, 0])
                X_all = np.array(X_all).reshape((-1, n_steps, 1))

                in_sample_scaled_pred_lstm = lstm_model_for_ticker.predict(X_all)
                in_sample_pred_lstm = scaler.inverse_transform(in_sample_scaled_pred_lstm).flatten()
                actuals_lstm = data_lstm[n_steps:].flatten()

                if len(actuals_lstm) > 0:
                    r2_lstm = r2_score(actuals_lstm, in_sample_pred_lstm)
                    r2_lstm = max(r2_lstm, 0.0)
                else:
                    r2_lstm = 0 
                X_input = scaled_data[-n_steps:].reshape((1, n_steps, 1))
                scaled_prediction_lstm = lstm_model_for_ticker.predict(X_input)
                next_day_price_lstm = scaler.inverse_transform(scaled_prediction_lstm)[0][0]
                next_day_prediction_lstm = float(next_day_price_lstm)

                mae_lstm = mean_absolute_error([actual_last_price], [next_day_prediction_lstm])
                mse_lstm = mean_squared_error([actual_last_price], [next_day_prediction_lstm])
                rmse_lstm = np.sqrt(mse_lstm)
                mape_lstm = mean_absolute_percentage_error([actual_last_price], [next_day_prediction_lstm]) * 100

                last_date = data_for_prediction.index[-1]
                next_day_date = last_date + datetime.timedelta(days=1)

                plot_dates_lstm = data_for_prediction['Close'].index.tolist()
                plot_prices_lstm = data_for_prediction['Close'].squeeze().tolist()
                plot_dates_lstm.append(next_day_date)
                plot_prices_lstm.append(next_day_prediction_lstm)

                lstm_result = create_plot_and_response(
                    plot_dates_lstm, plot_prices_lstm, next_day_prediction_lstm,
                    r2_lstm, mape_lstm, rmse_lstm, "LSTM"
                )
            except Exception as e:
                logger.error(f"LSTM error: {e}")

        # -----------------------------
        # RNN
        # -----------------------------
        rnn_result = None
        if stock_symbol in rnn_models:
            try:
                rnn_model_for_ticker = rnn_models[stock_symbol]
                data_rnn = data_for_prediction['Close'].values.reshape(-1, 1)
                scaler_rnn = MinMaxScaler(feature_range=(0, 1))
                scaled_data_rnn = scaler_rnn.fit_transform(data_rnn)

                X_all_rnn = []
                for i in range(n_steps, len(scaled_data_rnn)):
                    X_all_rnn.append(scaled_data_rnn[i - n_steps:i, 0])
                X_all_rnn = np.array(X_all_rnn).reshape((-1, n_steps, 1))

                in_sample_scaled_pred_rnn = rnn_model_for_ticker.predict(X_all_rnn)
                in_sample_pred_rnn = scaler_rnn.inverse_transform(in_sample_scaled_pred_rnn).flatten()
                actuals_rnn = data_rnn[n_steps:].flatten()

                if len(actuals_rnn) > 0:
                    r2_rnn = r2_score(actuals_rnn, in_sample_pred_rnn)
                    r2_rnn = max(r2_rnn, 0.0)
                else:
                    r2_rnn = 0

                X_input_rnn = scaled_data_rnn[-n_steps:].reshape((1, n_steps, 1))
                scaled_prediction_rnn = rnn_model_for_ticker.predict(X_input_rnn)
                next_day_price_rnn = scaler_rnn.inverse_transform(scaled_prediction_rnn)[0][0]
                next_day_prediction_rnn = float(next_day_price_rnn)

                mae_rnn = mean_absolute_error([actual_last_price], [next_day_prediction_rnn])
                mse_rnn = mean_squared_error([actual_last_price], [next_day_prediction_rnn])
                rmse_rnn = np.sqrt(mse_rnn)
                mape_rnn = mean_absolute_percentage_error([actual_last_price], [next_day_prediction_rnn]) * 100

                last_date = data_for_prediction.index[-1]
                next_day_date = last_date + datetime.timedelta(days=1)

                plot_dates_rnn = data_for_prediction['Close'].index.tolist()
                plot_prices_rnn = data_for_prediction['Close'].squeeze().tolist()
                plot_dates_rnn.append(next_day_date)
                plot_prices_rnn.append(next_day_prediction_rnn)

                rnn_result = create_plot_and_response(
                    plot_dates_rnn, plot_prices_rnn, next_day_prediction_rnn,
                    r2_rnn, mape_rnn, rmse_rnn, "RNN"
                )
            except Exception as e:
                logger.error(f"RNN error: {e}")

        results = [res for res in [sarimax_result, lstm_result, rnn_result] if res is not None]

        if not results:
            return jsonify({'error': f'Không có kết quả từ mô hình cho mã cổ phiếu {stock_symbol}.'}), 500

        for res in results:

            r2_value_str = res['metrics']['R2_Score'].replace('%', '')
            r2_value = float(r2_value_str)
            res['r2_float'] = r2_value

        results.sort(key=lambda x: x['r2_float'], reverse=True)

        for res in results:
            del res['r2_float']

        return jsonify({
            'best_model': results[0]['model'],
            'models': results
        })

    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return jsonify({'error': 'Đã xảy ra lỗi trong quá trình xử lý yêu cầu của bạn.'}), 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)