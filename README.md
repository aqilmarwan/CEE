# CEE Predictive Modeling Project

## 📋 Executive Summary

**Objective**: Build predictive models for Customer Engagement Engine (CEE) to predict conversion likelihood and optimize campaign targeting.

**Approach**: 3-phase implementation covering data exploration, feature engineering, predictive modeling, and business insights.

**Expected Outcome**: AUC-ROC > 0.65 with actionable business recommendations for customer prioritization and channel optimization.

---

## Problem Understanding

The project requires building predictive models for a Customer Engagement Engine (CEE) to predict conversion likelihood. The datasets include:

- **campaigns.csv**: 10 campaigns with metadata (channel, objective, launch_date, audience_size, budget_rm)
- **customers.csv**: ~5,000 customers with attributes (tier, relationship_start, preferred_channel, is_active, age, region)
- **engagements.csv**: Monthly aggregated campaign performance metrics
- **interactions.csv**: ~30,000 individual customer-campaign interactions (granular data for modeling)

---

## Implementation Approach

### Phase 1: Data Exploration & Feature Engineering

**1.1 Data Loading & Initial Exploration**
- Load all four CSV files using pandas
- Examine data types, missing values, distributions
- Understand relationships between datasets (join keys: customer_id, campaign_id)

**1.2 Dataset Joining**
- Primary modeling table: Start with `interactions.csv` (customer × campaign level)
- Join with `customers.csv` on `customer_id` to add customer attributes
- Join with `campaigns.csv` on `campaign_id` to add campaign metadata
- Use `engagements.csv` for validation/aggregation checks

**1.3 Feature Engineering**
- **Temporal Features**:
  - Calculate customer tenure from `relationship_start` to `send_date`
  - Extract time features: day_of_week, month, days_since_launch
  - Time-based features from `send_hour`
- **Engagement History Features**:
  - Use `prior_engagements_90d` from interactions
  - Calculate historical conversion rate per customer
  - Count of past opens/clicks per customer
  - Average revenue per customer historically
- **Customer-Campaign Alignment**:
  - Match indicator: `preferred_channel == channel`
  - Tier-based features (Bronze/Silver/Gold)
  - Active status impact
- **Campaign Features**:
  - Budget per customer (budget_rm / audience_size)
  - Days since campaign launch
  - Objective type (Upsell/Cross-Sell/Retention/Awareness)
- **Device & Timing Features**:
  - Device type (Mobile/Desktop)
  - Send hour (categorical or binned)
  - Region-based features

**1.4 Exploratory Data Analysis**
- Conversion rate by tier, channel, objective, region
- Distribution analysis of key features
- Correlation analysis
- Visualizations: conversion rates across segments, feature distributions

### Phase 2: Predictive Modeling

**2.1 Data Preparation**
- Handle missing values
- Encode categorical variables (one-hot encoding or label encoding)
- Split data: train/validation/test (70/15/15)
- Handle class imbalance (class_weight='balanced')
- **Critical**: Exclude leaky features (has_revenue, open_to_click_rate, click_to_convert_rate)
- Sanitize feature names for XGBoost (remove [, ], < characters)

**2.2 Model Selection & Training**
- Start with baseline models:
  - Logistic Regression (interpretable baseline)
  - Random Forest (feature importance)
  - XGBoost (performance)
- Use cross-validation for hyperparameter tuning
- Address class imbalance with class weights
- Remove early_stopping for cross-validation compatibility

**2.3 Model Evaluation**
- Primary metric: AUC-ROC
- Additional metrics: Precision, Recall, F1-Score
- Confusion matrix analysis
- Feature importance extraction (top 5 features)
- Train vs Validation vs Test comparison
- Overfitting detection

**2.4 Model Interpretation**
- SHAP values for model explainability
- Feature importance plots
- Partial dependence plots for key features
- Identify top 5 most predictive features with explanations

### Phase 3: Business Application & Insights

**3.1 Key Findings Analysis**
- What drives conversion? (top features analysis)
- Segment analysis: which customer segments convert best?
- Channel effectiveness: preferred vs. non-preferred channel impact
- Campaign objective performance
- Timing and device insights

**3.2 Recommendations**
- Customer prioritization: which customers to target first?
- Channel optimization: best channels for different segments
- Campaign timing: optimal send times
- Personalization strategies based on model insights

**3.3 CEE Integration Suggestions**
- How to use model for real-time scoring
- A/B testing framework
- Model retraining cadence
- Monitoring and alerting strategies

---

## Technical Stack

- **Data Manipulation**: pandas, numpy
- **Modeling**: scikit-learn, xgboost
- **Visualization**: matplotlib, seaborn
- **Interpretability**: SHAP
- **Serialization**: pickle, joblib

---

## 📊 Project Structure

### Data Flow
```
interactions.csv (base)
    ↓ JOIN customers.csv
    ↓ JOIN campaigns.csv
    ↓ FEATURE ENGINEERING
    ↓ ENCODING & PREPARATION
    ↓ MODEL TRAINING
    ↓ EVALUATION & INTERPRETATION
    ↓ BUSINESS INSIGHTS
```

### Key Variables
- `modeling_df`: Original joined dataframe (use for EDA)
- `df`: After feature engineering (before encoding)
- `df_encoded`: After encoding (for modeling)
- `X_train/X_val/X_test`: Feature matrices
- `y_train/y_val/y_test`: Target vectors

### Critical Exclusions
- **Leaky Features**: has_revenue, open_to_click_rate, click_to_convert_rate
- **Post-Interaction**: opened, clicked (outcomes, not predictors)
- **Target**: converted (this is what we're predicting)

---

## ⚠️ Important Implementation Notes

1. **Data Leakage Prevention**: Always exclude features derived from post-interaction outcomes
2. **Class Imbalance**: Use class_weight='balanced' (97:1 ratio)
3. **Feature Names**: Sanitize for XGBoost (remove special characters)
4. **Dataframe Management**: Use original dataframe for EDA, encoded for modeling
5. **NumPy Compatibility**: Ensure numpy<2.0 for package compatibility

---

## Success Criteria

✅ **Model Performance**: AUC-ROC > 0.65  
✅ **Top 5 Features**: Identified and explained  
✅ **Business Insights**: Clear, actionable recommendations  
✅ **Interpretability**: SHAP values and feature importance  
✅ **Deployment Ready**: Model artifact saved  

---

## File Structure

```
CEE/
├── cee.ipynb                    # Main implementation notebook
├── cee-dataset/
│   ├── campaigns.csv            # Campaign metadata
│   ├── customers.csv            # Customer attributes
│   ├── engagements.csv         # Monthly aggregated metrics
│   └── interactions.csv        # Individual interactions
└── README.md                    # This file
```

---

*For detailed implementation plan, see `cee-predictive-modeling-plan.plan.md`*
