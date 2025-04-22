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

# Create a custom callback to log losses more frequently
class LossLoggingCallback(pl.Callback):
  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
      if batch_idx % 10 == 0:  # Log every 10 batches
          # For PyTorch Lightning 2.0+
          if isinstance(outputs, dict) and "loss" in outputs:
              loss = outputs["loss"]
          # For older PyTorch Lightning versions
          else:
              try:
                  loss = outputs[0]["loss"]
              except (IndexError, KeyError, TypeError):
                  return  # Skip if we can't extract loss
              
          trainer.logger.experiment.add_scalar("train_loss", loss, trainer.global_step)
  
  def on_validation_epoch_end(self, trainer, pl_module):
      # This ensures we log validation loss at the end of each validation run
      metrics = trainer.callback_metrics
      if "val_loss" in metrics:
          trainer.logger.experiment.add_scalar("val_loss_epoch", metrics["val_loss"], trainer.global_step)


class LPSIntensityForecasting(self):
  
  def __init__(self, data, target, max_prediction_length, max_encoder_length, unknown_variables, bg_data):
    self.data = data
    self.target = target
    self.max_prediction_length = max_prediction_length
    self.max_encoder_length = max_encoder_length
    self.unknown_variables = unknown_variables
    self.bg_data = bg_data
    self.train_df, self.val_df, self.test_df = self.train_val_test_split(data)
    self.training = self.dataset(self.train_df)
    self.validaion = TimeSeriesDataSet.from_dataset(self.dataset(self.val_df), val_df, predict=True,)
    self.batch_size = 128  # set this between 32 to 128
    self.train_dataloader = self.validaion.to_dataloader(train=True, batch_size=self.batch_size, num_workers=4)
    self.val_dataloader  = self.validaion.to_dataloader(train=False, batch_size=self.batch_size * 10, num_workers=4)


  def train_val_test_split(self, data):
    unique_ids = data['id'].unique()
    train_ids, val_ids = train_test_split(unique_ids, test_size=0.4, random_state=42)
    val_ids, test_ids = train_test_split(val_ids, test_size=0.5, random_state=42)
    
    train_df = data[data['id'].isin(train_ids)]
    val_df = data[data['id'].isin(val_ids)]
    test_df = data[data['id'].isin(test_ids)]
    return train_df, val_df, test_df

  def dataset(self, data):
    ds =  TimeSeriesDataSet(
      data, time_idx="time_idx",
      target=self.target,
      group_ids=["id"],
      min_prediction_idx = 24,
      min_encoder_length=self.max_encoder_length, 
      max_encoder_length=self.max_encoder_length,
      min_prediction_length=1,
      max_prediction_length=self.max_prediction_length,
      static_reals=self.bg_data,
      time_varying_known_reals=["hour","day", "month"],
      time_varying_unknown_reals=self.unknown_variables,
      target_normalizer=GroupNormalizer(
          groups=["id"], transformation="log1p"
      ),   #use softplus and normalize by group
      add_relative_time_idx=True,
      add_target_scales=True,
      add_encoder_length=True,
      allow_missing_timesteps=True)
    return ds

  def train(self):
    pl.seed_everything(42)
  
    # Set up callbacks and logger
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1, patience=10, verbose=True, mode="min")
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger("lightning_logs", name="tft_model")  # Add a name for better organization
    
    # Set up the trainer with our custom callback added
    trainer = pl.Trainer(
        max_epochs=46,
        accelerator="gpu",
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        limit_train_batches=50,
        gradient_clip_val=0.09603541592590253,
        callbacks=[lr_logger, early_stop_callback, LossLoggingCallback()],  # Add our custom callback
        logger=logger)
  
    # Create and configure the TFT model - remove the log_interval and log_val_interval params
    self.tft = TemporalFusionTransformer.from_dataset(
        self.training,
        learning_rate= 0.003738531357168273,
        hidden_size=8,
        attention_head_size=1,
        dropout= 0.1174171245849331,
        hidden_continuous_size=8,
        loss=QuantileLoss(),
        optimizer=Ranger)

    trainer.fit(
        self.tft,
        train_dataloaders=self.train_dataloader,
        val_dataloaders=self.test_dataloader)
    

  def evaluate():
    md_ids=[]
    low_ids=[]
    for  i in test_ids:
        if(data[data["id"]==i]["MDorNot"].any()):
            md_ids.append(i)
        else:
            low_ids.append(i)
          
    Y_vals = []
    Ypred_vals = []
    Y_vals_lows = []
    Ypred_vals_lows = []
    Y_vals_md = []
    Ypred_vals_md = []
    aw = []
    aw_lows = []
    aw_md = []
    
    for i in test_ids:
        l = len(data[data["id"] == i  ])
        if(l==0):
            continue
        new_data = data[data["id"] == i  ][0:l]
        new_data['time_idx'] = list(range(1,len(new_data)+1))
        new_dataset = TimeSeriesDataSet.from_dataset(test_dataset(l-24), new_data,  predict=True,  stop_randomization=True  )
        new_dataloader = new_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
        raw_predictions = tft.predict(new_dataloader, mode="raw", return_x=True)
        predictions = tft.predict(new_dataloader, return_y=True, trainer_kwargs=dict(accelerator="gpu"))
        y_pred = raw_predictions.output.prediction[0][:, 3]
        y = raw_predictions.x['decoder_target']
        Y_vals.append(y[0].cpu())
        Ypred_vals.append(y_pred.cpu())
        attention_weights = tft.interpret_output(raw_predictions.output, reduction="sum")
        aw.append(attention_weights["encoder_variables"][4:].cpu().tolist())
        if i in low_ids:
            Y_vals_lows.append(y[0].cpu())
            Ypred_vals_lows.append(y_pred.cpu())
            aw_lows.append(attention_weights["encoder_variables"][4:].cpu().tolist())
        else:
            Y_vals_md.append(y[0].cpu())
            Ypred_vals_md.append(y_pred.cpu())
            aw_md.append(attention_weights["encoder_variables"][4:].cpu().tolist())
        print(i)

    #lead_time_bins = [(0, 24), (24, 48), (48, 72), (72, 96), (96, 120)]
    lead_time_bins = [(i,i+24) for i in range(0,121,24)]
    lead_time_labels = ['0-24 hrs', '24-48 hrs', '48-72 hrs', '72-96 hrs', '96-120 hrs','120-144'] #,'144-168','168-192','192-216','216-240']
    
    mean_errors = []
    error_spreads = []
    
    for start, end in lead_time_bins:
        # Filter to ensure sufficient length
        Y_vals_new = [i for i in Y_vals if len(i) > end - 1]
        Ypred_vals_new = [i for i in Ypred_vals if len(i) > end - 1]
        
        y = []
        ypred = []
        
        # Collect values within the start:end range without stacking
        for i in Y_vals_new:
            y += i[start:end].tolist()  # Append values from the specified range to y
        
        for i in Ypred_vals_new:
            ypred += i[start:end].tolist()  # Append values from the specified range to ypred
        
        # Calculate errors
        errors = np.array(ypred) - np.array(y)
        mean_errors.append(np.mean(errors) / 100)
        error_spreads.append(np.std(errors) / 100)
    plt.tight_layout()
    plt.savefig("bias.png",dpi=300)
    
    plt.figure(figsize=(8, 5))
    x = np.arange(len(lead_time_labels))
    bar_width = 0.8  # Set a wider bar width to match the example image
    
    # Create bars for mean errors with error bars
    bars = plt.bar(x, mean_errors, yerr=error_spreads, capsize=8, color='cornflowerblue', alpha=0.7)
    
    # Adding labels and title
    plt.xlabel('Forecast lead time (hours)', fontsize=16)
    plt.ylabel('Mean Error (hPa)', fontsize=16)
    #plt.title('Mean Error and Error Sp￼read for MSLP Forecasting', fontsize=14)
    plt.xticks(x, lead_time_labels, fontsize=14)
    #plt.yticks([-2,-1,0,1,2], fontsize=14)
    plt.ylim(-3, 3)
    # Adding horizontal grid lines for readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    # Adding a dashed line at y=0 for reference
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    
    # Adjust layout to avoid clipping
    plt.tight_layout()
    plt.savefig("bias.png",dpi=300)
    # Show plot
    plt.show()
    max_actual = []
    max_pred= []
    for i in range(len(Y_vals)):
        max_actual.append(max(Y_vals[i]))
        max_pred.append(max(Ypred_vals[i]))
    
    plt.scatter(max_actual,max_pred)
    plt.axhline(y=400, color='black', linestyle=':', linewidth=2, label="Horizontal Dotted Line")
    plt.axvline(x=400, color='black', linestyle=':', linewidth=2, label="Vertical Dotted Line")
    plt.xlabel("Actual mslp (Pa)", fontsize=14)
    plt.ylabel("Predicted mslp (Pa)", fontsize=14)
    plt.plot([0, 2000], [0, 2000], linestyle=':', color='black', linewidth=2, label="Diagonal Dotted Line")
    mdornot = []
    for i in test_ids:
        mdornot.append(data[data["id"]==i]["MDorNot"].to_list()[0])
    print(mdornot)
    md_hit = 0
    md_false = 0
    for i in range(0,len(test_ids)):
        actual = data[data["id"]==test_ids[i]]["MDorNot"].to_list()[0]
        pred = any(all(val > 400 for val in Ypred_vals[i][j:j + 6]) for j in range(len(Ypred_vals[i]) - 6 + 1))
        if(actual):
            if(pred):
                md_hit=md_hit+1
        else:
            if(pred):
                md_false = md_false +1
    
    total_md = mdornot.count(True)
    print("Hit Ratio: ", md_hit/total_md)
    print("False Alarm Ratio: ", md_false/len(test_ids))

    
