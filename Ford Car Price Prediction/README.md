# Ford Car Price Prediction

A machine learning project that predicts Ford car prices using Linear Regression. This project demonstrates the complete data science workflow including exploratory data analysis, data preprocessing, model training, and evaluation.

## Project Overview

This project builds a predictive model to estimate used Ford car prices based on various features such as year, mileage, engine size, transmission type, and fuel type. The notebook compares two different encoding approaches for categorical variables.

## Dataset

- **Source**: Kaggle Ford Car Price Prediction Dataset
- **File**: `ford.csv`
- **Features**:
  - `year`: Year of manufacture
  - `mileage`: Total miles driven
  - `tax`: Road tax
  - `mpg`: Miles per gallon (fuel efficiency)
  - `engineSize`: Engine displacement in liters
  - `model`: Car model
  - `transmission`: Transmission type (Manual/Automatic)
  - `fuelType`: Type of fuel (Petrol/Diesel/etc.)
  - `price`: Target variable (car price)

## Methodology

### 1. Data Loading & Exploration (`read_csv`)
- Load the dataset from Kaggle
- Display basic statistics (shape, info, describe)
- Check for missing values

### 2. Exploratory Data Analysis (EDA)
- Histograms of price distribution
- Correlation heatmap of numeric features
- Box plots for categorical variables (year, transmission, fuelType, engineSize, model)
- Scatter plot of mileage vs. price

### 3. Data Preprocessing
The notebook implements two encoding approaches:

**Approach 1: One-Hot Encoding**
- Converts categorical variables into binary columns
- Uses `pd.get_dummies()` for model, transmission, and fuelType
- Scales numerical features using `StandardScaler`

**Approach 2: Label Encoding**
- Converts categorical variables into numeric labels using `LabelEncoder`
- Maintains original feature count
- Also applies `StandardScaler` for normalization

### 4. Model Training
- **Algorithm**: Linear Regression
- **Test Size**: 33% (train/test split)
- **Random State**: 42
- Trains separate models for each encoding approach

### 5. Model Evaluation
- **Metrics Used**:
  - R² Score (coefficient of determination)
  - Adjusted R² Score (accounts for number of features)
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)

## Project Structure

```
ford-car-price-prediction/
├── ford-car-price-prediction.ipynb  # Main notebook
├── README.md                        # This file
└── ford.csv                         # Dataset (from Kaggle)
```

## Dependencies

```python
numpy          # Linear algebra
pandas         # Data manipulation
seaborn        # Statistical visualization
matplotlib     # Plotting
sklearn        # Machine learning library
```

## Installation & Usage

### Prerequisites
- Python 3.x
- Jupyter Notebook or JupyterLab

### Installation
```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

### Running the Notebook
```bash
jupyter notebook ford-car-price-prediction.ipynb
```

## Key Features

✓ Comprehensive EDA with visualizations
✓ Comparison of encoding strategies (One-Hot vs Label Encoding)  
✓ Feature scaling for better model performance
✓ Train-test split for model validation
✓ Multiple evaluation metrics
✓ Clean, well-commented code

## Results

The notebook generates:
- Statistical summaries of the dataset
- Visual insights through multiple chart types
- Two trained Linear Regression models
- Performance metrics (R², Adjusted R²) for both approaches
- Price predictions on test data

## Future Improvements

- Implement advanced models (Random Forest, Gradient Boosting, XGBoost)
- Feature engineering for improved predictions
- Hyperparameter tuning
- Cross-validation for more robust evaluation
- Residual analysis and error visualization
- Deploy model as a web service

## Author Notes

This is an educational project demonstrating:
- Data loading and exploration techniques
- Categorical variable encoding strategies
- Data preprocessing and scaling
- Supervised learning with regression
- Model evaluation and comparison

## License

This project uses publicly available data from Kaggle. Please refer to the original dataset's license.

---

**Happy Predicting!** 🚗📊
