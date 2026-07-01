# Toyota Corolla Price Prediction using Multiple Linear Regression

This project builds a predictive engine using **Multiple Linear Regression (MLR)**, **Lasso Regression**, and **Ridge Regression** to estimate the market price of used Toyota Corolla cars based on their operational and physical specifications.

---

## Dataset Overview
The dataset (`ToyotaCorolla - MLR.csv`) contains 1,436 instances across 11 key parameters:
*   **Price:** Offer price in Euros (Target Variable)
*   **Age_08_04:** Age of the vehicle in months
*   **KM:** Accumulated kilometers on the odometer
*   **Fuel_Type:** Type of fuel used (Petrol, Diesel, CNG)
*   **HP:** Engine Horse Power
*   **Automatic:** Transmission setup ($1 = \text{Yes}$, $0 = \text{No}$)
*   **cc:** Cylinder engine volume in cubic centimeters
*   **Doors / Cylinders / Gears / Weight:** Key structural and mechanical characteristics

---

## Project Steps & Methodology

### 1. Exploratory Data Analysis (EDA)
*   Descriptive statistics computed via `df.describe()`.
*   Data distributions reviewed dynamically using a Seaborn `pairplot()`.
*   Missing value verification completed ensuring a clean baseline ($0$ missing rows).

### 2. Feature Engineering & Preprocessing
*   **Categorical Encoding:** One-hot dummy encoding applied to categorical columns (`Fuel_Type` and `Automatic`) using `pd.get_dummies()`.
*   **Feature Interaction:** Evaluated complex model behavior by computing a custom feature interaction string between vehicle age and mileage:
    $$\text{Age\_KM\_interaction} = \text{Age\_08\_04} \times \text{KM}$$

### 3. Predictive Modeling & Multi-Model Evaluation
Three distinct Ordinary Least Squares (OLS) Linear Regression configurations were constructed alongside two regularized frameworks:

*   **Model 1 (Baseline MLR):** Full-feature multiple linear regression model ($R^2 \approx 0.835$).
*   **Model 2 (Subset MLR):** Regressed specifically on key structural attributes: `Age_08_04`, `KM`, `HP`, and `Weight` ($R^2 \approx 0.851$).
*   **Model 3 (Interaction MLR):** Built upon Model 1 with the addition of the explicit `Age_KM_interaction` term, yielding the highest accuracy profile ($R^2 \approx 0.862$).
*   **Lasso & Ridge Regularization:** Implemented via L1 and L2 penalty coefficients ($\alpha = 0.1$) to shrink weights and control feature collinearity.

---

## Model Performance Metrics

| Evaluation Model | Mean Squared Error (MSE) | R-squared ($R^2$) Score |
| :--- | :---: | :---: |
| **Model 1:** Baseline MLR | 2,203,043.82 | 83.49% |
| **Model 2:** Targeted Subset | 1,993,321.01 | 85.06% |
| **Model 3:** Age/KM Interaction | **1,839,093.41** | **86.22%** |
| **Lasso Regression** ($\alpha=0.1$) | 2,202,270.26 | 83.49% |
| **Ridge Regression** ($\alpha=0.1$) | 2,202,732.24 | 83.49% |

---

## Installation & Setup

1. Install required packages:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
