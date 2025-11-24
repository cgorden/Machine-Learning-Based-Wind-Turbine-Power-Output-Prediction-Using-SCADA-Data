# Wind Turbine Power Output Prediction Using Machine Learning

## Project Overview

This project implements machine learning models to predict wind turbine power output using operational SCADA (Supervisory Control and Data Acquisition) data. The goal is to develop accurate forecasting tools that enable better grid management and support renewable energy integration.

**Course:** ECGR 4105 - Machine Learning, Section 001  
**Team Members:** Cameron Gorden & Ethan Yang  
**Institution:** University of North Carolina at Charlotte  
**Date:** Fall 2025

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
- Multiple model comparison (Linear Regression, Random Forest, Gradient Boosting)
- Extensive data visualization and exploratory analysis
- Performance evaluation using R-squared, RMSE, and MAE metrics

### Objectives

1. Develop accurate ML models for wind power prediction
2. Identify key factors influencing power generation
3. Compare performance across different regression algorithms
4. Provide insights for operational grid management

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
   - Train three ML models
   - Evaluate and compare performance
   - Display results and insights

### Running Individual Sections

The notebook is organized into 12 sections. You can run specific sections independently:
```python
# Section 1: Import libraries
# Section 2: Load data
# Section 3: Explore data
# Section 4: Clean data
# Section 5: Engineer features
# Section 6: Create visualizations
# Section 7: Prepare data for ML
# Section 8: Train models
# Section 9: Evaluate models
# Section 10: Visualize results
# Section 11: Analyze feature importance
# Section 12: Generate summary
```

### Example: Training a Single Model
```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Train Gradient Boosting model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, 
                                   max_depth=5, random_state=42)
model.fit(xTrain, yTrain)

# Make predictions
predictions = model.predict(xTest)

# Evaluate
r2 = r2_score(yTest, predictions)
mae = mean_absolute_error(yTest, predictions)

print(f"R-squared: {r2:.4f}")
print(f"MAE: {mae:.2f} kW")
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

| Section | Lines | Description |
|---------|-------|-------------|
| 1 | Imports | Library imports and configuration |
| 2 | Data Loading | Download and load dataset from Kaggle |
| 3 | Exploration | Initial data inspection and statistics |
| 4 | Cleaning | Remove invalid/missing values |
| 5 | Engineering | Create derived features |
| 6 | EDA | Generate visualizations |
| 7 | Preparation | Split and scale data |
| 8 | Training | Train ML models |
| 9 | Evaluation | Compare model performance |
| 10 | Visualization | Plot results |
| 11 | Importance | Analyze feature importance |
| 12 | Summary | Final report and insights |

## Results

### Model Performance

| Model | R² Score | RMSE (kW) | MAE (kW) |
|-------|----------|-----------|----------|
| **Gradient Boosting** | **0.9190** | **374.02** | **154.58** |
| Linear Regression | 0.9090 | 396.60 | 186.36 |
| Random Forest | 0.9051 | 404.85 | 168.66 |

### Key Findings

1. **Gradient Boosting achieved best performance:** 91.90% variance explained
2. **Wind speed is the dominant predictor:** ~60% feature importance
3. **All models exceeded 90% R-squared:** Demonstrates strong predictive capability
4. **Average prediction error:** 154.58 kW (4.3% of rated capacity)

### Visualizations Generated

1. **Power Curve:** Wind speed vs power output relationship
2. **Wind Direction Distribution:** Bimodal pattern showing prevailing winds
3. **Power Output Distribution:** Operational state frequencies
4. **Correlation Heatmap:** Feature relationships
5. **Model Comparison:** R-squared scores across models
6. **Feature Importance:** Predictor rankings for Gradient Boosting

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
│   └── figures/                      # Generated visualizations
│       ├── power_curve.png
│       ├── wind_direction.png
│       ├── power_distribution.png
│       ├── correlation_heatmap.png
│       ├── model_comparison.png
│       └── feature_importance.png
│
└── data/
    └── T1.csv                        # Downloaded dataset (auto-generated)
```

## Troubleshooting

### Common Issues

**Issue:** Kaggle authentication error  
**Solution:** Ensure Kaggle API credentials are configured. In Colab, follow authentication prompts.

**Issue:** Memory error during training  
**Solution:** Reduce dataset size or use fewer trees in ensemble models (reduce `n_estimators`).

**Issue:** Plots not displaying  
**Solution:** Add `%matplotlib inline` at the top of the notebook.

**Issue:** Import errors  
**Solution:** Run `!pip install [missing-library]` in a notebook cell.

## Future Improvements

- [ ] Implement time-series models (LSTM, ARIMA) for temporal dependencies
- [ ] Extend to multi-turbine wind farm predictions
- [ ] Incorporate weather forecast data for longer prediction horizons
- [ ] Add confidence intervals to predictions
- [ ] Deploy model as REST API for real-time forecasting
- [ ] Implement cross-validation for more robust evaluation
- [ ] Test generalization across different turbine models and locations

## Team Responsibilities

**Cameron Gorden:**
- Model development and evaluation (Sections 8-11)
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

## License

This project is for educational purposes as part of ECGR 4105 coursework. Dataset license follows Kaggle's terms of use.



---

**Last Updated:** November 2025
