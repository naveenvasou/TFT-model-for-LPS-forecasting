
# TFT Model for LPS Forecasting

## Overview

This project implements a Temporal Fusion Transformer to predict the trajectory and strength of monsoon LPS over the Indian subcontinent. By leveraging ERA5 reanalysis data and historical LPS track records, the model provides multi-horizon forecasts that can enhance early warning systems and disaster preparedness. This model achieves accurate 5-day forecasts outperforming traditional Numerical Weather Prediction (NWP) models. The TFT architecture is based on the work by Lim et al. (2021), which introduced a novel attention-based model for interpretable multi-horizon time series forecasting[^1].


## Repository Structure

```
.
├── data/
│   ├── processed/                   # ERA5 atmospheric variables
│   └── combine-lows-and-md-data.py  # Merge Lows and MDs data
│   └── era5_data_processing_lows.py # Process ERA5 atmospheric variables for Lows tracks
│   └── era5_data_processing_md.py   # Process ERA5 atmospheric variables for MDs tracks
│   └── process_track_data.py        # Process raw track information into a dataframe
├── models/
│   ├── intensity_model.py      # training and evaluation modules of the TFT model for intensity forecasting
│   └── position_model.py       # training and evaluation modules of the TFT model for position forecasting
├── notebooks/
│   ├── intensity_forecasting.ipynb  # Jupyter notebook for intensity model training and evaluation
│   └── position_forecasting.ipynb   # Jupyter notebook for position model training and evaluation
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch
- PyTorch Forecasting
- pandas
- numpy
- matplotlib


## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work utilizes the Temporal Fusion Transformer architecture, originally developed by Google in collaboration with the University of Oxford.

## References

[^1]: Lim, B., Arik, S. Ö., Loeff, N., & Pfister, T. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*. International Journal of Forecasting, 37(4), 1748–1764. https://doi.org/10.1016/j.ijforecast.2021.03.012
---

For more information and updates, please visit the [project repository](https://github.com/naveenvasou/TFT-model-for-LPS-forecasting).
