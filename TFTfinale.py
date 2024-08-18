import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import torch

def compute_indicators(df):
    df['EMA_12'] = df['value'].ewm(span=12, adjust=False).mean()
    delta = df['value'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['VWAP'] = (df['value'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    return df

def main():
    stock_data = pd.read_csv('bigWMT.csv', parse_dates=['Date'])
    gdp_data = pd.read_csv('GDP.csv', parse_dates=['DATE'])
    gdp_data['GDP'] = gdp_data['GDP']

    CPI_data = pd.read_csv('CPIAUCSL.csv', parse_dates=['DATE'])
    QQQ_data = pd.read_csv('QQQ.csv', parse_dates=['Date'])
    DIA_data = pd.read_csv('DIA.csv', parse_dates=['Date'])
    SPY_data = pd.read_csv('SPY.csv', parse_dates=['Date'])

    stock_data.rename(columns={'Date': 'date', 'Close': 'value'}, inplace=True)
    gdp_data.rename(columns={'DATE': 'date', 'GDP': 'gdp'}, inplace=True)
    CPI_data.rename(columns={'DATE': 'date', 'CPIAUCSL': 'cpiaucsl'}, inplace=True)
    QQQ_data.rename(columns={'Date': 'date', 'Close': 'qqq'}, inplace=True)
    DIA_data.rename(columns={'Date': 'date', 'Close': 'dia'}, inplace=True)
    SPY_data.rename(columns={'Date': 'date', 'Close': 'spy'}, inplace=True)

    merged_data = pd.merge_asof(stock_data.sort_values('date'), gdp_data.sort_values('date'), on='date', direction='backward')
    merged_data = pd.merge_asof(merged_data.sort_values('date'), CPI_data.sort_values('date'), on='date', direction='backward')
    merged_data = pd.merge_asof(merged_data.sort_values('date'), QQQ_data.sort_values('date'), on='date', direction='backward')
    merged_data = pd.merge_asof(merged_data.sort_values('date'), DIA_data.sort_values('date'), on='date', direction='backward')
    merged_data = pd.merge_asof(merged_data.sort_values('date'), SPY_data[['date', 'spy']].sort_values('date'), on='date', direction='backward')

    merged_data['group'] = '0'
    
    merged_data['time_idx'] = np.arange(len(merged_data))
    merged_data['month'] = merged_data['date'].dt.month
    merged_data = compute_indicators(merged_data)
    merged_data.dropna(inplace=True)
    merged_data = merged_data[-1200:]

    # Apply StandardScaler
    feature_columns = ['gdp', 'cpiaucsl', 'qqq', 'dia', 'spy', 'EMA_12', 'RSI', 'VWAP']

    # Splitting the data
    train_data = merged_data[-1200:-130].reset_index(drop=True)
    val_data = merged_data[-1200:-65].reset_index(drop=True)
    test_data = merged_data[-430:].reset_index(drop=True)

    max_encoder_length = 365
    max_prediction_length = 65

    training = TimeSeriesDataSet(
        train_data,
        time_idx='time_idx',
        target='value',
        group_ids=['group'],
        time_varying_unknown_reals=feature_columns,
        time_varying_known_reals=['time_idx'],
        max_encoder_length=max_encoder_length,
        min_encoder_length=max_encoder_length // 2,
        max_prediction_length=max_prediction_length,
        min_prediction_length=1,
        static_categoricals=['group'],
        target_normalizer=GroupNormalizer(groups=["group"], transformation="softplus"),
        allow_missing_timesteps=True,
    )

    # validation = TimeSeriesDataSet.from_dataset(training, val_data, predict=True, stop_randomization=True)
    # train_dataloader = training.to_dataloader(train=True, batch_size=32, num_workers=8)
    # val_dataloader = validation.to_dataloader(train=False, batch_size=320, num_workers=8)

    # tft = TemporalFusionTransformer.from_dataset(
    #     training,
    #     learning_rate=0.00001,
    #     hidden_size=256, 
    #     attention_head_size=6,
    #     dropout=0.3,
    #     hidden_continuous_size=6,
    #     loss=QuantileLoss(),
    #     log_interval=10,
    #     reduce_on_plateau_patience=4,
    # )

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath='./TFTfiles',
    #     filename='{epoch}-{val_MAE:.4f}-WMT_unks',
    #     save_top_k=2,
    #     monitor='val_MAE',
    #     mode='min'
    # )

    # logger = TensorBoardLogger("tb_logs", name="tft_model")

    # trainer = pl.Trainer(
    #     max_epochs=100,
    #     accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    #     callbacks=[checkpoint_callback],
    #     check_val_every_n_epoch=1,
    #     logger=logger
    # )

    # trainer.fit(
    #     model=tft,
    #     train_dataloaders=train_dataloader,
    #     val_dataloaders=val_dataloader
    # )

    np.random.seed(2)

    test_encoder_data = test_data.iloc[:max_encoder_length].copy()
    test_decoder_data = test_encoder_data.iloc[-65:]
    # test_decoder_data = pd.concat([test_decoder_data]*65, ignore_index=True)
    test_decoder_data['time_idx'] = np.arange(test_encoder_data['time_idx'].max() + 1, test_encoder_data['time_idx'].max() + 1 + max_prediction_length)
    test_decoder_data['date'] = pd.date_range(start=test_encoder_data['date'].iloc[-1], periods=max_prediction_length + 1, freq='D')[1:]
    test_decoder_data['month'] = test_decoder_data['date'].dt.month
    new_data = pd.concat([test_encoder_data, test_decoder_data], ignore_index=True)
    new_dataset = TimeSeriesDataSet.from_dataset(training, new_data, stop_randomization=True)
    new_dataloader = new_dataset.to_dataloader(train=False, batch_size=1, num_workers=1)

    tft = TemporalFusionTransformer.load_from_checkpoint("./TFTfiles/epoch=33-val_MAE=3.3764-WMT_unks.ckpt")
    raw_predictions = tft.predict(new_dataloader, mode='raw')
    num_quantiles = raw_predictions['prediction'].shape[2]

    middle_quantile_index = (num_quantiles // 2)
    predictions = raw_predictions['prediction'][:, :, middle_quantile_index].detach().cpu().numpy()
    selected_batch_predictions = predictions[0]
    
    # Inverse transform the predictions
    predicted_values = selected_batch_predictions.reshape(-1, 1).flatten()
    actual_values = test_data['value'].iloc[max_encoder_length:max_encoder_length + max_prediction_length].values.reshape(-1, 1).flatten()

    plt.figure(figsize=(12, 6))
    plt.plot(new_data['gdp'], color='red', label='GDP')
    plt.plot(new_data['cpiaucsl'], color='blue', label='CPIAUCSL')
    plt.plot(new_data['qqq'], color='green', label='QQQ')
    plt.plot(new_data['dia'], color='black', label='DIA')
    plt.plot(new_data['spy'], color='purple', label='SPY')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(predicted_values, color='red', label='Predicted Prices')
    plt.plot(actual_values, color='black', label='Actual Prices')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
