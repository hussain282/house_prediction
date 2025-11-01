# 🏠 AI House Price Prediction Dashboard

An interactive **Streamlit web app** that predicts house sale prices using a **Gradient Boosting Regressor** trained on the **Ames Housing Dataset**.  
This project demonstrates **data preprocessing, feature engineering, model building, and deployment** of a machine learning model in a web application.

## 🚀 Overview

This project predicts **house sale prices** based on various factors such as construction quality, area, year built, and neighborhood characteristics.  
It includes an elegant web dashboard built using **Streamlit**, with interactive forms, visualizations, and explanations for each input.

## 🧠 Tech Stack

| Category | Tools & Libraries |
|-----------|-------------------|
| Programming | Python 3.10+ |
| ML Model | GradientBoostingRegressor |
| Preprocessing | pandas, numpy, sklearn |
| Web Framework | Streamlit |
| Visualization | matplotlib |
| Model Persistence | joblib |
| Styling | Custom CSS & HTML |

## 🌟 Features

✅ Interactive Input Form  
✅ Feature Explanations with Tooltips  
✅ Feature Importance Visualization  
✅ Actual vs Predicted Graph  
✅ Prediction History Tracking  
✅ Modern Responsive UI  

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/house-price-predictor.git
cd house-price-predictor
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the App
```bash
streamlit run app.py
```

## 📊 Model Details

- **Algorithm:** Gradient Boosting Regressor  
- **R² Score:** ~0.85  
- **RMSE:** ~25,000  
- **Trained On:** Processed Ames Housing Dataset  

Features used include: `OverallQual`, `GrLivArea`, `GarageCars`, `YearBuilt`, `LotArea`, etc.

## 🧰 Dependencies

```
streamlit
pandas
numpy
scikit-learn
matplotlib
xgboost
joblib
```

## 🧠 Future Enhancements

- 🗺️ Map-based visualization  
- 💾 Database for predictions  
- 🧮 Multiple ML models comparison  
- 📊 SHAP-based explainability  

---
**Author:** The King 👑  
AI & ML Developer | Data Science Enthusiast

