# Time Series Forecasting with LSTM

## Neural Networks 2025.1  

This project aims to predict Bitcoin closing prices using **LSTM (Long Short-Term Memory)** neural networks applied to historical time series data.

---

## 📂 Project Structure

- `data-bitcoin_timedata_v2.csv` : Dataset with historical Bitcoin prices (`date` and `close`).  
- `notebook.ipynb` : Main notebook with the full pipeline, including exploratory analysis, LSTM model training, evaluation, and extrapolation.  
- `best_model_state_dict.pth` : Trained model with the best performance saved during training (generated after running the notebook).

---

## 🛠 Technologies Used

- Python 3.x  
- Pandas, Numpy  
- Matplotlib, Seaborn  
- Statsmodels (autocorrelation and correlation analysis)  
- Scikit-learn (MinMaxScaler)  
- PyTorch (LSTM, training and evaluation)  
- tqdm (progress bar)

---

## 🔍 Project Pipeline

1. **Data Loading and Preprocessing**  
   - Convert `date` column to datetime  
   - Normalize closing price (`close`) using `MinMaxScaler`  
   - Split data into `train`, `valid`, and `test` sets  

2. **Exploratory Analysis**  
   - Time series visualization  
   - Correlation analysis between lags  
   - Pairplot for joint visualization of lagged variables  

3. **Data Preparation for LSTM**  
   - Create temporal windows (`window_size`) for X (features) and y (target)  
   - Generate `DataLoaders` for training, validation, and testing  

4. **LSTM Modeling**  
   - Define LSTM architecture with dropout and final linear layer  
   - Train using `Adam`, monitor loss, and apply `early stopping`  
   - Optional learning rate scheduler (`StepLR`, `CosineAnnealingLR`, etc.)  

5. **Model Evaluation**  
   - Metrics used: RMSE, MAE, MAPE  
   - Visualize predictions on the test set (`next day`)  
   - Iterative extrapolation for future forecasting

---

## ⚙ Model Configuration

Example of hyperparameters used in the project:

```python
config = {
    "window_size": 7,
    "hidden_size": 80,
    "num_layers": 1,
    "dropout_rate": 0.1,
    "n_epochs": 300,
    "learning_rate": 5e-3,
    "optimizer": torch.optim.Adam,
    "loss_fn": torch.nn.MSELoss(),
    "patience": 20,
    "bs_train": 16,
    "bs_val_test": 256,
    "use_lr_scheduler": True,
    "scheduler_type": "StepLR",
    "scheduler_params": {"StepLR": {"step_size": 20, "gamma": 0.7}}
}
