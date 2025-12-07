# Wind Turbine Power Output Prediction Using Machine Learning

## Project Overview

This project implements machine learning models to predict wind turbine power output using operational SCADA (Supervisory Control and Data Acquisition) data. The goal is to develop accurate forecasting tools that enable better grid management and support renewable energy integration.

**Course:** ECGR 4105 - Machine Learning, Section 001  
**Team Members:** Cameron Gorden & Ethan Yang  
**Institution:** University of North Carolina at Charlotte  
**Date:** Fall 2024

## Table of Contents

- [Project Description](#project-description)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Results](#results)
- [Dependencies](#dependencies)
- [References](#references)

## Project Description

Wind energy production is highly variable due to changing environmental conditions. This project uses machine learning to predict wind turbine power generation based on operational data, helping energy companies plan power distribution more effectively.

### Key Features

- Comprehensive data preprocessing and cleaning
- Feature derivation (power coefficient, directional bins)
- Multiple model comparison (Linear Regression, Gradient Boosting, Neural Network)
- Deep learning implementation with multi-layer perceptron architecture
- Extensive data visualization and exploratory analysis
- Performance evaluation using R-squared, RMSE, and MAE metrics

### Objectives

1. Develop accurate ML models for wind power prediction
2. Compare traditional machine learning with deep learning approaches
3. Identify key factors influencing power generation
4. Compare performance across different regression algorithms
5. Provide insights for operational grid management

## Dataset

**Source:** [Wind Turbine SCADA Dataset](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset)  
**Reference Notebook:** [Wind Turbine Power Curve Control](https://www.kaggle.com/code/berkerisen/a-wind-turbine-power-curve-control)

### Dataset Characteristics

- **Total Observations:** 50,473 (after cleaning)
- **Time Period:** Full year 2018
- **Sampling Interval:** 10 minutes
- **Data Retention Rate:** 99.89% after cleaning

### Features

| Feature | Description | Unit |
|---------|-------------|------|
| Date/Time | Timestamp of observation | DateTime |
| Wind Speed | Wind speed measurement | m/s |
| Wind Direction | Wind direction | degrees (0-360) |
| LV ActivePower | Actual power output | kW |
| Theoretical_Power_Curve | Theoretical power based on wind speed | kWh |

### Derived Features

- **powerCoefficient:** Actual power / Theoretical power (efficiency metric)
- **windDirectionBin:** Wind direction discretized into 8 sectors (N, NE, E, SE, S, SW, W, NW)

## Installation

### Prerequisites

- Python 3.7 or higher
- Google Colab account (recommended) or local Jupyter environment
- Kaggle account for dataset access

### Required Libraries
```bash
pip install kagglehub numpy pandas matplotlib seaborn scikit-learn
```

### Setup Instructions

1. Clone or download this repository
2. Install required dependencies (see above)
3. Configure Kaggle API credentials (if running locally)
4. Open the notebook in Google Colab or Jupyter

## Usage

### Running the Complete Pipeline

1. Open `wind_turbine_prediction.ipynb` in Google Colab
2. Run all cells sequentially: `Runtime → Run All`
3. The notebook will automatically:
   - Download the dataset from Kaggle
   - Preprocess and clean the data
   - Generate exploratory visualizations
   - Train three ML models (Linear Regression, Gradient Boosting, Neural Network)
   - Evaluate and compare performance
   - Display results and insights

### Running Individual Sections

The notebook is organized into sections that can be run independently:
```python
# Section 1: Import libraries
# Section 2: Load data
# Section 3: Explore data
# Section 4: Clean data
# Section 5: Engineer features
# Section 6: Create visualizations
# Section 7: Prepare data for ML
# Section 8: Train models (including Neural Network)
# Section 9: Evaluate models
# Section 10: Visualize results
# Section 11: Analyze feature importance
# Section 12: Generate summary
```

### Example: Training Models
```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Train Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, 
                                      max_depth=5, random_state=42)
gb_model.fit(xTrain, yTrain)

# Train Neural Network model
nn_model = MLPRegressor(hidden_layer_sizes=(100, 50, 25), activation='relu',
                        solver='adam', max_iter=500, random_state=42,
                        early_stopping=True)
nn_model.fit(xTrain, yTrain)

# Make predictions
gb_predictions = gb_model.predict(xTest)
nn_predictions = nn_model.predict(xTest)

# Evaluate
gb_r2 = r2_score(yTest, gb_predictions)
nn_r2 = r2_score(yTest, nn_predictions)

print(f"Gradient Boosting R²: {gb_r2:.4f}")
print(f"Neural Network R²: {nn_r2:.4f}")
```

## Code Structure

### Variable Naming Convention

All variables use **camelCase** for consistency:
```python
# Data variables
rawDataFrame, cleanedDataFrame, correlationMatrix

# Feature variables
inputFeatures, targetVariable, featureColumns

# Model variables
modelsToTrain, bestModelName, testPredictions

# Scaler variables
featureScaler, scaledInputFeatures
```

### Section Breakdown

| Section | Description |
|---------|-------------|
| 1 | Library imports and configuration |
| 2 | Download and load dataset from Kaggle |
| 3 | Initial data inspection and statistics |
| 4 | Remove invalid/missing values |
| 5 | Create derived features |
| 6 | Generate visualizations |
| 7 | Split and scale data |
| 8 | Train ML models (Linear Regression, Gradient Boosting, Neural Network) |
| 9 | Compare model performance |
| 10 | Plot results |
| 11 | Analyze feature importance |
| 12 | Final report and insights |

## Results

### Model Performance

| Model | R² Score | RMSE (kW) | MAE (kW) |
|-------|----------|-----------|----------|
| **Gradient Boosting** | **0.9190** | **374.12** | **154.62** |
| Neural Network | 0.9150 | 383.11 | 158.22 |
| Linear Regression | 0.9086 | 397.40 | 184.88 |

### Key Findings

1. **Gradient Boosting achieved best performance:** 91.90% variance explained
2. **Neural Network competitive performance:** 91.50% variance explained (only 0.4% behind)
3. **Wind speed is the dominant predictor:** ~60% feature importance
4. **All models exceeded 90% R-squared:** Demonstrates strong predictive capability
5. **Deep learning validates ensemble methods:** Both advanced approaches effectively captured non-linear relationships

### Neural Network Architecture

- **Input Layer:** 4 features (wind speed, direction, theoretical power, direction bins)
- **Hidden Layer 1:** 100 neurons with ReLU activation
- **Hidden Layer 2:** 50 neurons with ReLU activation
- **Hidden Layer 3:** 25 neurons with ReLU activation
- **Output Layer:** 1 neuron (power prediction)
- **Optimizer:** Adam with learning rate 0.001
- **Early Stopping:** Enabled to prevent overfitting
- **Total Parameters:** ~7,000+ trainable weights

### Visualizations Generated

1. **Power Curve:** Wind speed vs power output relationship
2. **Wind Direction Distribution:** Bimodal pattern showing prevailing winds
3. **Power Output Distribution:** Operational state frequencies
4. **Correlation Heatmap:** Feature relationships
5. **Model Comparison:** R-squared scores across all three models
6. **Feature Importance:** Predictor rankings for Gradient Boosting
7. **Actual vs Predicted:** Scatter plots for all models
8. **Residual Distribution:** Error analysis for best model
9. **Neural Network Loss Curve:** Training convergence visualization

## Dependencies

### Core Libraries
```
kagglehub>=0.2.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
```

### Complete Requirements File

Create a `requirements.txt`:
```
kagglehub
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure
```
wind-turbine-prediction/
│
├── README.md                          # This file
├── wind_turbine_prediction.ipynb     # Main Jupyter notebook
├── requirements.txt                   # Python dependencies
├── report/
│   ├── IEEE_report.pdf               # Final IEEE format report
│   ├── presentation.pdf              # PowerPoint presentation
│   └── figures/                      # Generated visualizations
│       ├── power_curve.png
│       ├── wind_direction.png
│       ├── power_distribution.png
│       ├── correlation_heatmap.png
│       ├── model_comparison.png
│       ├── feature_importance.png
│       ├── actual_vs_predicted_gb.png
│       ├── actual_vs_predicted_nn.png
│       ├── residual_distribution.png
│       └── nn_loss_curve.png
│
└── data/
    └── T1.csv                        # Downloaded dataset (auto-generated)
```

## Troubleshooting

### Common Issues

**Issue:** Kaggle authentication error  
**Solution:** Ensure Kaggle API credentials are configured. In Colab, follow authentication prompts.

**Issue:** Memory error during training  
**Solution:** Reduce dataset size or use fewer trees in ensemble models (reduce `n_estimators`) or smaller neural network (reduce hidden layer sizes).

**Issue:** Neural Network not converging  
**Solution:** Increase `max_iter` parameter or adjust learning rate. Check for proper data normalization.

**Issue:** Plots not displaying  
**Solution:** Add `%matplotlib inline` at the top of the notebook.

**Issue:** Import errors  
**Solution:** Run `!pip install [missing-library]` in a notebook cell.

**Issue:** Neural Network training too slow  
**Solution:** Reduce hidden layer sizes or enable GPU acceleration in Colab (Runtime → Change runtime type → GPU).

## Model Comparison Analysis

### When to Use Each Model:

**Linear Regression:**
- Fastest training and inference
- Most interpretable
- Best for quick baselines
- Suitable when simplicity is priority
- 90.86% accuracy may be sufficient for many applications

**Gradient Boosting:**
- Best overall accuracy (91.90%)
- Good interpretability via feature importance
- Moderate training time
- Best balance of performance and explainability
- **Recommended for production deployment**

**Neural Network:**
- Strong accuracy (91.50%)
- Most flexible architecture
- Can incorporate time-series patterns in future extensions
- Requires more tuning and computational resources
- Best when maximum model flexibility is needed

## Future Improvements

- [ ] Implement time-series models (LSTM, ARIMA) for temporal dependencies
- [ ] Extend Neural Network to use recurrent layers (RNN/LSTM) for sequential patterns
- [ ] Extend to multi-turbine wind farm predictions with wake effect modeling
- [ ] Incorporate weather forecast data for longer prediction horizons
- [ ] Add confidence intervals and prediction uncertainty quantification
- [ ] Deploy models as REST API for real-time forecasting
- [ ] Implement cross-validation for more robust evaluation
- [ ] Test generalization across different turbine models and locations
- [ ] Hyperparameter optimization using grid search or Bayesian optimization
- [ ] Ensemble multiple models for improved predictions

## Team Responsibilities

**Cameron Gorden:**
- Model development and evaluation (Sections 8-11)
- Neural Network implementation and tuning
- Final analysis and conclusions
- Documentation and presentation preparation

**Ethan Yang:**
- Data preprocessing and cleaning (Sections 4-5)
- Exploratory data analysis and visualization (Section 6)
- Baseline model training
- Visual reporting for presentation

## References

[1] B. Erisen, "Wind Turbine SCADA Dataset," Kaggle, 2018. [Online]. Available: https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset

[2] B. Erisen, "A Wind Turbine Power Curve Control," Kaggle, 2018. [Online]. Available: https://www.kaggle.com/code/berkerisen/a-wind-turbine-power-curve-control
