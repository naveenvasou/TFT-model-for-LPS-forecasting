import os
import warnings
import copy
from pathlib import Path
import warnings
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss

from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_optimizer import Ranger


# Create a custom callback to log training and validation loss
class LossLoggingCallback(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 10 == 0:  # Log every 10 batches
            if isinstance(outputs, dict) and "loss" in outputs:
                loss = outputs["loss"]
            else:
                try:
                    loss = outputs[0]["loss"]
                except (IndexError, KeyError, TypeError):
                    return
            trainer.logger.experiment.add_scalar("train_loss", loss, trainer.global_step)

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "val_loss" in metrics:
            trainer.logger.experiment.add_scalar("val_loss_epoch", metrics["val_loss"], trainer.global_step)


class LPSPositionForecasting:
    def __init__(self, data, max_prediction_length, max_encoder_length, unknown_variables, bg_data):
        self.data = data
        self.max_prediction_length = max_prediction_length
        self.max_encoder_length = max_encoder_length
        self.unknown_variables = unknown_variables
        self.bg_data = bg_data

        # Split data
        self.train_df, self.val_df, self.test_df = self.train_val_test_split(data)

        # Create datasets
        self.lat_training = self.dataset(self.train_df, "Latitude")
        self.lon_training = self.dataset(self.train_df, "Longitude")
        self.lat_validation = TimeSeriesDataSet.from_dataset(self.dataset(self.val_df, "Latitude"), self.val_df, predict=True)
        self.lon_validation = TimeSeriesDataSet.from_dataset(self.dataset(self.val_df, "Longitude"), self.val_df, predict=True)

        # DataLoaders
        self.batch_size = 128
        self.lat_train_dataloader = self.lat_training.to_dataloader(train=True, batch_size=self.batch_size, num_workers=4)
        self.lon_train_dataloader = self.lon_training.to_dataloader(train=True, batch_size=self.batch_size, num_workers=4)
        self.lat_val_dataloader = self.lat_validation.to_dataloader(train=False, batch_size=self.batch_size * 10, num_workers=4)
        self.lon_val_dataloader = self.lon_validation.to_dataloader(train=False, batch_size=self.batch_size * 10, num_workers=4)

    def train_val_test_split(self, data):
        # Splits by unique track id
        unique_ids = data['id'].unique()
        train_ids, val_ids = train_test_split(unique_ids, test_size=0.4, random_state=42)
        val_ids, test_ids = train_test_split(val_ids, test_size=0.5, random_state=42)
        train_df = data[data['id'].isin(train_ids)]
        val_df = data[data['id'].isin(val_ids)]
        test_df = data[data['id'].isin(test_ids)]
        return train_df, val_df, test_df

    def dataset(self, data, target, max_pred=None):
        # Prepare TimeSeriesDataSet for TFT
        if max_pred is None:
            max_pred = self.max_prediction_length
        ds = TimeSeriesDataSet(
            data,
            time_idx="time_idx",
            target=target,
            group_ids=["id"],
            min_prediction_idx=24,
            min_encoder_length=self.max_encoder_length,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_pred,
            static_reals=self.bg_data,
            time_varying_known_reals=["hour", "day", "month"],
            time_varying_unknown_reals=self.unknown_variables,
            target_normalizer=GroupNormalizer(groups=["id"], transformation="log1p"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )
        return ds

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        # Compute great-circle distance between two points
        R = 6371  # Earth's radius in kilometers
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    @staticmethod
    def init_trainer(gcv):
        # Create Lightning Trainer
        pl.seed_everything(42)
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-2, patience=5, verbose=False, mode="min")
        lr_logger = LearningRateMonitor()
        logger = TensorBoardLogger("lightning_logs")
        trainer = pl.Trainer(
            max_epochs=46,
            accelerator="gpu",
            enable_model_summary=True,
            gradient_clip_val=gcv,
            limit_train_batches=50,
            callbacks=[lr_logger, early_stop_callback],
            logger=logger
        )
        return trainer

    @staticmethod
    def return_tft(data, lr, hs, hcs, ath, dr):
        # Build TFT model
        tft = TemporalFusionTransformer.from_dataset(
            data,
            learning_rate=lr,
            hidden_size=hs,
            attention_head_size=ath,
            dropout=dr,
            hidden_continuous_size=hcs,
            loss=QuantileLoss(),
            optimizer=Ranger
        )
        print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
        return tft

    def train(self):
        # Train Latitude model
        self.lat_tft = self.return_tft(self.lat_training, 0.004667698728065982, 16, 12, 3, 0.12103404618064319)
        trainer = self.init_trainer(0.06379293111970405)
        trainer.fit(self.lat_tft, train_dataloaders=self.lat_train_dataloader, val_dataloaders=self.lat_val_dataloader)

        # Train Longitude model
        self.lon_tft = self.return_tft(self.lon_training, 0.004667698728065982, 16, 12, 3, 0.12103404618064319)
        trainer = self.init_trainer(0.06379293111970405)
        trainer.fit(self.lon_tft, train_dataloaders=self.lon_train_dataloader, val_dataloaders=self.lon_val_dataloader)


    def evaluate():
      md_ids=[]
      low_ids=[]
      for  i in val_ids:
      
          if(data[data["id"]==i]["MDorNot"].any()):
              md_ids.append(i)
          else:
              low_ids.append(i)
  
      Y_vals_lat = []
      Ypred_vals_lat = []
      Y_vals_lows_lat = []
      Ypred_vals_lows_lat = []
      Y_vals_md_lat = []
      Ypred_vals_md_lat = []
      aw_lat = []
      aw_lows_lat = []
      aw_md_lat = []
      
      for i in test_ids:
          l = len(self.data[self.data["id"] == i  ])
          new_data = self.data[self.data["id"] == i  ][0:l]
          new_data['time_idx'] = list(range(1,len(new_data)+1))
          new_dataset = TimeSeriesDataSet.from_dataset(self.dataset(new_data, "Latitude", max_pred=l-24), new_data,  predict=True,  stop_randomization=True  )
          new_dataloader = new_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
          raw_predictions = lat_tft.predict(new_dataloader, mode="raw", return_x=True)
          predictions = lat_tft.predict(new_dataloader, return_y=True, trainer_kwargs=dict(accelerator="gpu"))
          y_pred = raw_predictions.output.prediction[0][:, 3]
          y = raw_predictions.x['decoder_target']
          Y_vals_lat.append(y[0].cpu())
          Ypred_vals_lat.append(y_pred.cpu())
          attention_weights = lat_tft.interpret_output(raw_predictions.output, reduction="sum")
          aw_lat.append(attention_weights["encoder_variables"][4:].cpu().tolist())
          if i in low_ids:
              Y_vals_lows_lat.append(y[0].cpu())
              Ypred_vals_lows_lat.append(y_pred.cpu())
              aw_lows_lat.append(attention_weights["encoder_variables"][4:].cpu().tolist())
          else:
              Y_vals_md_lat.append(y[0].cpu())
              Ypred_vals_md_lat.append(y_pred.cpu())
              aw_md_lat.append(attention_weights["encoder_variables"][4:].cpu().tolist())
          print(i)
      
      
      Y_vals_lon = []
      Ypred_vals_lon = []
      Y_vals_lows_lon = []
      Ypred_vals_lows_lon = []
      Y_vals_md_lon = []
      Ypred_vals_md_lon = []
      aw_lon = []
      aw_lows_lon = []
      aw_md_lon = []
      
      for i in test_ids:
          l = len(data[data["id"] == i  ])
          new_data = data[data["id"] == i  ][0:l]
          new_data['time_idx'] = list(range(1,len(new_data)+1))
          new_dataset = TimeSeriesDataSet.from_dataset(self.dataset(new_data, "Longitude", max_pred=l-24), new_data,  predict=True,  stop_randomization=True  )
          new_dataloader = new_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
          raw_predictions = lon_tft.predict(new_dataloader, mode="raw", return_x=True)
          predictions = lon_tft.predict(new_dataloader, return_y=True, trainer_kwargs=dict(accelerator="gpu"))
          y_pred = raw_predictions.output.prediction[0][:, 3]
          y = raw_predictions.x['decoder_target']
          Y_vals_lon.append(y[0].cpu())
          Ypred_vals_lon.append(y_pred.cpu())
          attention_weights = lon_tft.interpret_output(raw_predictions.output, reduction="sum")
          aw_lon.append(attention_weights["encoder_variables"][4:].cpu().tolist())
          if i in low_ids:
              Y_vals_lows_lon.append(y[0].cpu())
              Ypred_vals_lows_lon.append(y_pred.cpu())
              aw_lows_lon.append(attention_weights["encoder_variables"][4:].cpu().tolist())
          else:
              Y_vals_md_lon.append(y[0].cpu())
              Ypred_vals_md_lon.append(y_pred.cpu())
              aw_md_lon.append(attention_weights["encoder_variables"][4:].cpu().tolist())
          print(i)
  
      #lead_time_bins = [(0, 24), (24, 48), (48, 72), (72, 96), (96, 120)]
      lead_time_bins = [(i,i+24) for i in range(0,121,24)]
      lead_time_labels = ['0-24 hrs', '24-48 hrs', '48-72 hrs', '72-96 hrs', '96-120 hrs','120-144'] #,'144-168','168-192','192-216','216-240']
      mean_errors = []
      error_spreads = []
      
      for start, end in lead_time_bins:
          # Filter for sufficient length
          Y_vals_lat_new = [i for i in Y_vals_lat if len(i) > end - 1]
          Y_vals_lon_new = [i for i in Y_vals_lon if len(i) > end - 1]
          Ypred_vals_lat_new = [i for i in Ypred_vals_lat if len(i) > end - 1]
          Ypred_vals_lon_new = [i for i in Ypred_vals_lon if len(i) > end - 1]
          
          distances = []
          # Collect distances within the start:end range
          for lat, lon, pred_lat, pred_lon in zip(Y_vals_lat_new, Y_vals_lon_new, Ypred_vals_lat_new, Ypred_vals_lon_new):
              
              actual = np.array([lat[start:end], lon[start:end]])
              predicted = np.array([pred_lat[start:end], pred_lon[start:end]])
              for a, p in zip(actual.T, predicted.T):
      
                  distances.append(haversine_distance(a[0], a[1], p[0], p[1]))
          
          # Calculate mean error and error spread
          distances = np.array(distances)  # Convert to NumPy array for calculations
          mean_errors.append(np.mean(distances))
          error_spreads.append(np.std(distances))
  
      plt.figure(figsize=(8, 5))
      x = np.arange(len(lead_time_labels))
      bar_width = 0.8  # Set a wider bar width to match the example image
      
      # Create bars for mean errors with error bars
      bars = plt.bar(x, mean_errors, yerr=error_spreads, capsize=8, color='cornflowerblue', alpha=0.7)
      
      # Adding labels and title
      plt.xlabel('Forecast lead time (hours)', fontsize=16)
      plt.ylabel('Mean Error (km)', fontsize=16)
      #plt.title('Mean Error and Error Spread for MSLP Forecasting', fontsize=14)
      plt.xticks(x, lead_time_labels, fontsize=14)
      plt.yticks(fontsize=14)
      #plt.ylim(-3, 3)
      # Adding horizontal grid lines for readability
      plt.grid(axis='y', linestyle='--', alpha=0.7)
      plt.grid(axis='x', linestyle='--', alpha=0.7)
      # Adding a dashed line at y=0 for reference
      plt.axhline(0, color='gray', linestyle='--', linewidth=1)
      
      # Adjust layout to avoid clipping
      plt.tight_layout()
      plt.savefig("track_error.png",dpi=300)
      # Show plot
      plt.show()
