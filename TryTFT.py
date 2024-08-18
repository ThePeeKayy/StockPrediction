import pandas as pd
import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import numpy as np
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import os

def main():
    # Load and preprocess stock data
    stock_data = pd.read_csv('bigNVDA.csv', parse_dates=['Date'])
    gdp_data = pd.read_csv('GDP.csv', parse_dates=['DATE'])
    CPI_data = pd.read_csv('CPIAUCSL.csv', parse_dates=['DATE'])
    VIX_data = pd.read_csv('VIX.csv', parse_dates=['Date'])
    SPX_data = pd.read_csv('SPX.csv', parse_dates=['Date'])
    IXIC_data = pd.read_csv('IXIC.csv', parse_dates=['Date'])

    from sklearn.preprocessing import MinMaxScaler
    train_value_scaler = MinMaxScaler(feature_range=(0, 1))
    test_value_scaler = MinMaxScaler(feature_range=(0, 1))
    gdp_scaler = MinMaxScaler(feature_range=(0, 1))
    cpiaucsl_scaler = MinMaxScaler(feature_range=(0, 1))
    spx_scaler = MinMaxScaler(feature_range=(0, 1))
    vix_scaler = MinMaxScaler(feature_range=(0, 1))
    ixic_scaler = MinMaxScaler(feature_range=(0, 1))

    # Rename columns for consistency
    stock_data.rename(columns={'Date': 'date', 'Close': 'value'}, inplace=True)
    gdp_data.rename(columns={'DATE': 'date', 'GDP': 'gdp'}, inplace=True)
    CPI_data.rename(columns={'DATE': 'date', 'CPIAUCSL': 'cpiaucsl'}, inplace=True)
    SPX_data.rename(columns={'Date': 'date', 'Close': 'spx'}, inplace=True)
    VIX_data.rename(columns={'Date': 'date', 'Close': 'vix'}, inplace=True)
    IXIC_data.rename(columns={'Date': 'date', 'Close': 'ixic'}, inplace=True)

    # Merge stock data with economic indicators based on dates
    merged_data = pd.merge_asof(stock_data.sort_values('date'), gdp_data.sort_values('date'), left_on='date', right_on='date', direction='backward')
    merged_data = pd.merge_asof(merged_data.sort_values('date'), CPI_data.sort_values('date'), left_on='date', right_on='date', direction='backward')
    merged_data = pd.merge_asof(merged_data.sort_values('date'), SPX_data.sort_values('date'), left_on='date', right_on='date', direction='backward')
    merged_data = pd.merge_asof(merged_data.sort_values('date'), VIX_data.sort_values('date'), left_on='date', right_on='date', direction='backward')
    merged_data = pd.merge_asof(merged_data.sort_values('date'), IXIC_data[['date', 'ixic']].sort_values('date'), left_on='date', right_on='date', direction='backward')
    merged_data['group'] = '0'  # Ensure group is a string
    merged_data['time_idx'] = np.arange(len(merged_data))
    merged_data['month'] = merged_data['date'].dt.month

    # Fit the scalers on the merged data (training data) and transform
    merged_data['value'] = train_value_scaler.fit_transform(merged_data[['value']])
    merged_data['gdp'] = gdp_scaler.fit_transform(merged_data[['gdp']])
    merged_data['cpiaucsl'] = cpiaucsl_scaler.fit_transform(merged_data[['cpiaucsl']])
    merged_data['spx'] = spx_scaler.fit_transform(merged_data[['spx']])
    merged_data['vix'] = vix_scaler.fit_transform(merged_data[['vix']])
    merged_data['ixic'] = ixic_scaler.fit_transform(merged_data[['ixic']])

    print(merged_data.head())

    # Define the dataset parameters
    max_encoder_length = 500
    max_prediction_length = 100

    # Define the dataset for training
    training = TimeSeriesDataSet(
        merged_data,
        time_idx='time_idx',
        target='value',
        group_ids=['group'],
        time_varying_unknown_reals=['value'],
        time_varying_known_reals=['gdp', 'cpiaucsl', 'spx', 'vix', 'month', 'ixic'],
        max_encoder_length=max_encoder_length,
        min_encoder_length=max_encoder_length // 2,
        max_prediction_length=max_prediction_length,
        min_prediction_length=1,
        static_categoricals=['group'],
        target_normalizer=GroupNormalizer(
            groups=["group"], transformation="softplus"
        ),
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, merged_data, predict=True, stop_randomization=True)

    # Create dataloaders
    train_dataloader = training.to_dataloader(train=True, batch_size=128, num_workers=8)
    val_dataloader = validation.to_dataloader(train=False, batch_size=128, num_workers=8)

    # #Define the TFT model
    # tft = TemporalFusionTransformer.from_dataset(
    #     training,
    #     learning_rate=0.005,  # Adjusted learning rate for stability
    #     hidden_size=256,
    #     attention_head_size=16,
    #     dropout=0.4,
    #     hidden_continuous_size=8,
    #     output_size=7,
    #     loss=QuantileLoss(),
    #     log_interval=10,
    #     reduce_on_plateau_patience=8
    # )
    # # Define checkpoint callback
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath='./TFTfiles',  # specify your directory
    #     filename='{epoch}-{val_MAE:.2f}-AAPL',  # pattern for the saved filenames
    #     save_top_k=1,  # save only the best model
    #     monitor='val_MAE',  # metric to monitor
    #     mode='min'  # save models with the minimum val_loss
    # )
    # # Define the trainer
    # trainer = pl.Trainer(
    #     max_epochs=20,  # Increase epochs for better training
    #     accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    #     callbacks=[checkpoint_callback],
    #     check_val_every_n_epoch=1
    # )

    # # Train the model
    # trainer.fit(
    #     model=tft,
    #     train_dataloaders=train_dataloader,
    #     val_dataloaders=val_dataloader
    # )


    # For testing, select a random 90-day segment and predict the next 30 days
    np.random.seed(6)
    test_data = merged_data[:-3000]
    print(test_data.head())
    #start_idx = np.random.randint(0, len(test_data) - max_encoder_length - max_prediction_length)
    start_idx = 100
    test_encoder_data = test_data.iloc[start_idx:start_idx + max_encoder_length]
    test_decoder_data = test_data.iloc[start_idx + max_encoder_length:start_idx + max_encoder_length + max_prediction_length]
    
    # Ensure consistency in time_idx
    test_decoder_data['time_idx'] = np.arange(test_encoder_data['time_idx'].max() + 1, test_encoder_data['time_idx'].max() + 1 + max_prediction_length)

    new_data = pd.concat([test_encoder_data, test_decoder_data], ignore_index=True)
    new_dataset = TimeSeriesDataSet.from_dataset(training, new_data, stop_randomization=True)
    new_dataloader = new_dataset.to_dataloader(train=False, batch_size=1, num_workers=1)

    # Make predictions
    tft = TemporalFusionTransformer.load_from_checkpoint("./TFTfiles/epoch=17-val_MAE=0.02-AAPL.ckpt")

    raw_predictions = tft.predict(new_dataloader, mode='raw')
    
    # Extract the median quantile (0.5) predictions
    predictions = raw_predictions['prediction'][:, :, 1].detach().cpu().numpy()

    selected_batch_predictions = predictions[0]

    # Convert the predictions back to the original scale using the test scaler
    predicted_values = train_value_scaler.inverse_transform(selected_batch_predictions.reshape(-1, 1)).flatten()

    # Actual values to compare with predictions
    actual_values = train_value_scaler.inverse_transform(
        test_data['value'].iloc[start_idx + max_encoder_length:start_idx + max_encoder_length + max_prediction_length].values.reshape(-1, 1)
    ).flatten()

    # Plot predictions vs reality
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(predicted_values, color='red', label='Predicted Prices')
    plt.plot(actual_values, color='black', label='Actual Prices')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
