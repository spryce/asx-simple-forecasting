# asx-simple-forecasting

AI-powered time series forecasting for the Australian Securities Exchange using daily closing values.

## Overview

This project is an AI prototype designed to experiment with TensorFlow and neural networks for time series forecasting. It leverages deep learning techniques to analyze historical data and predict future trends. The project is still under development and may undergo significant changes.

## Features

- **Data preprocessing**: Clean and filter raw data for model training (`data.json`, `filtered_data.json`).
- **Neural network models**: Multiple machine learning models implemented using TensorFlow (`model.py`, `z-model.py`, `model-2.py`).
- **Training and evaluation**: Scripts to train models, adjust learning rates, and evaluate performance (`gpt-model.py`, `learning_rate.py`).
- **Visualization**: Generate and visualize forecasts with AI-driven predictions (`forecast-1.png`, `forecast-2.png`, `forecast-z-1.png`).
- **Loss tracking**: Monitor training progress with loss visualizations (`training_loss.png`).

## AI-Specific Details

- **Framework**: TensorFlow is used as the primary deep learning framework.
- **Neural Networks**: The project implements custom neural network architectures for time series forecasting.
- **Optimization**: Includes learning rate adjustment and loss tracking to optimize model performance.
- **Scalability**: Designed to handle large datasets for robust forecasting.

## Requirements

Install the required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your data in `data.json`.
2. Run the preprocessing script to generate `filtered_data.json`.
3. Train the model using `model.py` or other model scripts.
4. Generate predictions using `prediction.py`.
5. Visualize the results using the provided forecast images.

## File Descriptions

- **`app.py`**: Main application script.
- **`data.json`**: Raw input data for forecasting.
- **`filtered_data.json`**: Preprocessed data.
- **`model.py`, `z-model.py`, `model-2.py`**: TensorFlow-based neural network models.
- **`gpt-model.py`**: Model training script.
- **`learning_rate.py`**: Learning rate adjustment script.
- **`prediction.py`**: Script for generating predictions.
- **`training_loss.png`**: Visualization of training loss.
- **`forecast-*.png`**: Forecast result visualizations.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

This project was inspired by advancements in artificial intelligence, deep learning, and time series forecasting techniques.
