# ✈️ Flight Price Prediction using AWS SageMaker

![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker-orange?style=for-the-badge&logo=amazon-aws)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-green?style=for-the-badge)

## 📋 Overview

This project provides a robust solution for predicting flight prices using machine learning techniques deployed on AWS SageMaker. The interactive web application allows users to input their flight details and receive instant price predictions, helping travelers make informed booking decisions.

## ✨ Features

- **Interactive UI**: User-friendly interface built with Streamlit
- **Advanced ML Pipeline**: Comprehensive preprocessing and feature engineering
- **Cloud Deployment**: Leveraging AWS SageMaker for scalable predictions
- **Robust Model**: XGBoost regression model with high accuracy

## 🛠️ Technical Components

### Data Processing
- Handles categorical features with sophisticated encoding techniques
- Extracts valuable features from datetime components
- Implements custom transformers for location-based features
- Applies RBF kernel transformations for capturing non-linear relationships
- Winsorizes outliers to improve model stability

### Machine Learning
- RandomForest-based feature selection
- XGBoost regression model for predictions
- Comprehensive ML pipeline with ColumnTransformer and FeatureUnion

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- AWS account with SageMaker access
- Required packages: pandas, numpy, scikit-learn, xgboost, streamlit, etc.

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/flight-price-prediction.git

# Navigate to the project directory
cd flight-price-prediction

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 📊 Usage

1. Open the Streamlit app in your browser
2. Fill in the flight details:
   - Airline
   - Date of journey
   - Source and destination
   - Departure and arrival times
   - Duration
   - Number of stops
   - Additional information
3. Click "Predict" to get the estimated flight price

## 🔗 Architecture

```
                 +------------------+
                 |                  |
User Input ----► | Streamlit App    |
                 |                  |
                 +--------+---------+
                          |
                          ▼
              +-----------------------+
              |                       |
              | Feature Preprocessing |
              |                       |
              +-----------+-----------+
                          |
                          ▼
                 +------------------+
                 |                  |
                 | XGBoost Model    |
                 |                  |
                 +--------+---------+
                          |
                          ▼
                 +------------------+
                 |                  |
                 | Price Prediction |
                 |                  |
                 +------------------+
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- AWS SageMaker for providing the ML infrastructure
- Streamlit for the web application framework
- XGBoost for the powerful gradient boosting framework
- Feature-engine for advanced feature engineering capabilities

## 📧 Contact

For any questions or feedback, please reach out to bhawanadhaka285002@gmail.com
