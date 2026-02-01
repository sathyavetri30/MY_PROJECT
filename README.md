Household Power Consumption Forecasting
Transformer with Attention vs XGBoost
📌 Project Overview

This project focuses on multivariate time series forecasting of household electricity consumption using a Transformer-based deep learning model with attention mechanisms and compares its performance with a traditional XGBoost regression model.

The objective is to accurately predict Global Active Power using historical power consumption data and engineered temporal features. The Transformer model leverages self-attention to capture long-term temporal dependencies, while XGBoost serves as a strong machine-learning baseline using lagged features.

📂 Dataset Description

Dataset: Household Power Consumption

Frequency: Minute-level time series

Target Variable: global_active_power

Features Used:

Global reactive power

Voltage

Global intensity

Sub-metering values (1, 2, 3)

Time-based features

Rolling statistics

⚙️ Environment & Libraries

The project is implemented in Python using the following libraries:

NumPy & Pandas – Data handling and preprocessing

PyTorch – Transformer model development

Scikit-learn – Scaling and evaluation metrics

XGBoost – Baseline regression model

CUDA (optional) – GPU acceleration

The code automatically selects GPU (CUDA) if available, otherwise falls back to CPU execution.

🧹 Data Preprocessing

Column Cleaning

Column names are standardized (lowercase, stripped spaces).

Datetime Processing

The time column is converted into a datetime format and set as the index.

Handling Missing Values

Non-numeric values are coerced into NaN.

Time-based interpolation is applied.

Remaining missing rows are dropped.

🧠 Feature Engineering

To enrich the model’s temporal understanding, the following features are created:

⏰ Time Features

hour – Hour of the day (0–23)

dayofweek – Day of the week (0–6)

📈 Statistical Features

rolling_mean_24 – 24-hour rolling mean of global active power

rolling_std_24 – 24-hour rolling standard deviation

These features help the model capture daily seasonality and volatility patterns.

📏 Data Scaling

All features are standardized using StandardScaler, ensuring:

Zero mean

Unit variance

This is crucial for stabilizing Transformer training and improving convergence.

🔗 Sequence Generation

A sliding window approach is used:

Sequence Length: 48 time steps

Input Shape: (samples, 48, features)

Target: Next time step’s global_active_power

This structure allows the Transformer to learn temporal dependencies across multiple hours.

🧾 Dataset & DataLoader

A custom PyTorch Dataset class is implemented to:

Convert sequences into tensors

Enable efficient batch loading

The DataLoader shuffles training data and uses mini-batches for faster optimization.

🔮 Transformer Model Architecture
Model Components:

Input Embedding Layer

Projects input features into a higher-dimensional space (d_model = 64)

Transformer Encoder

Multi-head self-attention (nhead = 4)

Stacked encoder layers (num_layers = 2)

Attention Extraction

Attention weights are captured for interpretability

Fully Connected Output Layer

Predicts the final power consumption value

Why Transformer?

Captures long-range temporal dependencies

Parallel computation (faster than RNN/LSTM)

Attention enables interpretability

🏋️ Model Training

Loss Function: Mean Squared Error (MSE)

Optimizer: Adam

Learning Rate: 0.001

Gradient Clipping: Prevents exploding gradients

Epochs: 2 (can be increased for better performance)

The model is trained in training mode and evaluated in inference mode.

📊 Evaluation Metrics

Three robust metrics are used:

RMSE (Root Mean Squared Error)
Measures prediction accuracy.

MASE (Mean Absolute Scaled Error)
Scale-independent metric useful for time series.

sMAPE (Symmetric Mean Absolute Percentage Error)
Percentage-based error metric.

👁️ Attention Interpretation

Attention weights are analyzed to identify:

Most influential time steps within the 48-step input window

This provides insights into which historical periods most impact predictions, improving model transparency.

🌳 XGBoost Baseline Model
Feature Strategy:

Lag-1 and Lag-24 features for all variables

Model Configuration:

300 estimators

Depth of 6

Learning rate of 0.05

XGBoost serves as a strong baseline for performance comparison against the Transformer.

📈 Performance Comparison

Results from both models are compiled into a table containing:

RMSE

MASE

sMAPE

The final metrics are saved as final_metrics.csv for reporting and analysis.

💾 Model Saving

The trained Transformer model is saved as:

final_transformer_attention_model.pth


Metrics are stored in CSV format for reproducibility.

✅ Key Outcomes

✔ Demonstrates effective use of Transformer models for time series forecasting
✔ Provides model interpretability through attention analysis
✔ Benchmarks performance against a classical ML model
✔ Suitable for academic projects, research, and production pipelines# MY_PROJECT
