import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytorch_forecasting import MAE, NBeats, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import pickle
import os
import glob

# Force CPU usage
torch.set_num_threads(1)  # Optional: limit CPU threads
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide GPU devices

def compute_indicators(df):
    df['SMA_20'] = df['value'].rolling(20, min_periods=1).mean()
    delta = df['value'].diff()
    gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0, np.inf)  
    df['RSI'] = 100 - (100 / (1 + rs))
    df['MACD'] = df['value'].ewm(12).mean() - df['value'].ewm(26).mean()
    df['Volume_Ratio'] = df.get('Volume', pd.Series(1, index=df.index)) / df.get('Volume', pd.Series(1, index=df.index)).rolling(10).mean()
    return df.ffill().bfill()

def load_data(folder, ticker=None):
    dfs = []
    files = [f"{folder}/{ticker}.csv"] if ticker else glob.glob(f"{folder}/*.csv")
    for file in files:
        ticker_name = ticker or os.path.basename(file).replace(".csv", "")
        df = pd.read_csv(file)
        df['group'] = ticker_name
        dfs.append(df)
    return dfs

def prepare_data(dfs, external_data_folder="./datasets"):
    merged = pd.concat(dfs, ignore_index=True)
    merged['date'] = pd.to_datetime(merged['date'].astype(str).str.split(' ').str[0])
    merged = merged[merged['date'] >= pd.Timestamp("2018-01-01")]
    merged.rename(columns={'volume': 'Volume', 'close': 'value'}, inplace=True)
    
    for file, cols in [('GDP.csv', {'DATE': 'date', 'GDP': 'gdp'}), 
                       ('CPIAUCSL.csv', {'DATE': 'date', 'CPIAUCSL': 'cpi'}),
                       ('SPY.csv', {'Date': 'date', 'Close': 'spy'})]:
        try:
            ext_data = pd.read_csv(f"{external_data_folder}/{file}", parse_dates=[list(cols.keys())[0]])
            ext_data = ext_data.rename(columns=cols)
            if 'gdp' in ext_data.columns:
                ext_data['gdp'] = ext_data['gdp'] / 50
            merged = pd.merge_asof(merged.sort_values('date'), ext_data.sort_values('date'), 
                                  on='date', direction='backward')
        except FileNotFoundError:
            if 'gdp' not in merged.columns: merged['gdp'] = 0
            if 'cpi' not in merged.columns: merged['cpi'] = 100
            if 'spy' not in merged.columns: merged['spy'] = merged['value'].mean()
    
    merged['month'] = merged['date'].dt.month.astype(str)
    merged['year'] = merged['date'].dt.year
    merged = merged.ffill().bfill()
    
    result = [compute_indicators(group.copy()) for name, group in merged.groupby('group')]
    merged = pd.concat(result, ignore_index=True)
    merged = merged.sort_values(['group', 'date']).reset_index(drop=True)
    merged['time_idx'] = merged.groupby('group').cumcount()
    return merged

def create_datasets(data, max_enc=80, max_pred=20):
    max_time = data.groupby('group')['time_idx'].max().min()
    train_cutoff = int(max_time * 0.8)
    val_cutoff = int(max_time * 0.95)
    
    train_data = data[data['time_idx'] <= train_cutoff]
    val_data = data[(data['time_idx'] > train_cutoff) & (data['time_idx'] <= val_cutoff)]
    
    min_time_idx = max_enc + 1
    train_data = train_data[train_data.groupby('group')['time_idx'].transform('max') >= min_time_idx]
    val_data = val_data[val_data.groupby('group')['time_idx'].transform('max') >= min_time_idx]
    train_data.to_csv('nbeats_train_data.csv', index=False)
    
    training = TimeSeriesDataSet(
        train_data,
        time_idx='time_idx',
        target='value',
        group_ids=['group'],
        min_encoder_length=max_enc,
        max_encoder_length=max_enc,
        min_prediction_length=max_pred,
        max_prediction_length=max_pred,
        static_categoricals=[],
        static_reals=[],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=['value'],
        target_normalizer=GroupNormalizer(
            groups=["group"],
            transformation=None,
            center=False
        ),
        add_relative_time_idx=False,
    )
    
    validation = TimeSeriesDataSet.from_dataset(training, val_data, predict=True, stop_randomization=True)
    return training, validation

def train_model(training, validation, model_name="nbeats_model"):
    train_loader = training.to_dataloader(train=True, batch_size=32, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=128, num_workers=0)
    
    model = NBeats.from_dataset(training, learning_rate=0.0002, widths=[256, 128, 256],
        num_blocks=[3, 2, 2], num_block_layers=[2, 2, 2], expansion_coefficient_lengths=[16, 8, 16],
        sharing=[False, True, True], stack_types=["trend", "seasonality", "generic"], loss=MAE())
    
    os.makedirs('./NBEATSfiles', exist_ok=True)
    os.makedirs('./scalers', exist_ok=True)
    
    trainer = pl.Trainer(
        max_epochs=50, 
        accelerator='cpu',  # Force CPU usage
        devices=1, 
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=15, min_delta=1e-4),
            ModelCheckpoint(dirpath='./NBEATSfiles', filename=f'{model_name}-{{epoch}}-{{val_loss:.4f}}',
                           monitor='val_loss', mode='min', save_top_k=1)
        ]
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    with open(f'./scalers/training_{model_name}.pkl', 'wb') as f:
        pickle.dump(training, f)
    
    return model

def predict_stock(ticker, model_name="nbeats_model", data_folder="./tiingo_data"):
    # Load model and ensure it's on CPU
    model = NBeats.load_from_checkpoint(
        './NBEATSfiles/nbeats_model-epoch=14-val_loss=9.7598.ckpt',
        map_location='cpu'  # Force CPU loading
    )
    model.eval()  # Set to evaluation mode
    
    with open(f'./scalers/training_{model_name}.pkl', 'rb') as f:
        training_dataset = pickle.load(f)
    
    all_data = load_data(data_folder)
    full_data = prepare_data(all_data)
    ticker_data = full_data[full_data['group'] == ticker].sort_values('date').reset_index(drop=True)
    
    if ticker_data.empty:
        raise ValueError(f"No data found for ticker: {ticker}")
    
    max_enc = training_dataset.max_encoder_length
    max_pred = training_dataset.max_prediction_length
    recent_data = ticker_data.tail(max_enc + max_pred + 20).copy()
    
    pred_dataset = TimeSeriesDataSet.from_dataset(training_dataset, recent_data, predict=True, stop_randomization=True)
    pred_loader = pred_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
    
    with torch.no_grad():
        predictions = model.predict(pred_loader, mode='prediction')
    
    # Ensure predictions are on CPU
    predictions = predictions.detach().cpu().numpy().squeeze()
    
    last_date = recent_data['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(predictions), freq='B')
    
    plt.figure(figsize=(15, 8))
    hist_len = min(60, len(recent_data))
    plt.plot(recent_data['date'].iloc[-hist_len:], recent_data['value'].iloc[-hist_len:], 'b-', label='Historical', alpha=0.7)
    plt.plot(recent_data['date'].iloc[-max_enc:], recent_data['value'].iloc[-max_enc:], 'g-', label=f'Encoder Input', linewidth=2)
    plt.plot([recent_data['date'].iloc[-1], future_dates[0]], [recent_data['value'].iloc[-1], predictions[0]], 'r:', alpha=0.7)
    plt.plot(future_dates, predictions, 'r-', label='N-BEATS Predictions', linewidth=3, marker='o', markersize=4)
    plt.title(f'{ticker} Stock Price Prediction (N-BEATS)\nLatest: ${recent_data["value"].iloc[-1]:.2f} → First Prediction: ${predictions[0]:.2f}')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return predictions, future_dates

def main():
    TRAINING = False
    PREDICT_TICKER = "CRM"
    
    if TRAINING:
        all_data = load_data("./tiingo_data")
        merged_data = prepare_data(all_data)
        training, validation = create_datasets(merged_data, max_enc=80, max_pred=20)
        model = train_model(training, validation)
    
    predictions, dates = predict_stock(PREDICT_TICKER)

if __name__ == '__main__':
    main()