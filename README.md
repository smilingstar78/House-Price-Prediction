# 🏠 House Price Prediction Project

## **Project Overview**
This project aims to predict **house prices** using advanced machine learning techniques, including **Random Forest** and **XGBoost Regression**. Accurate prediction of house prices is crucial for **buyers, sellers, real estate investors, and data-driven decision-making** in the housing market.  

The goal is to create a model that can reliably estimate house prices based on multiple features, ranging from **structural attributes of the house** to **location factors**.

---

## **📂 Dataset**
The dataset includes detailed information about houses, including numeric and categorical features. The main features used in this project are:

**Numeric Features:**
- `bedrooms` — Number of bedrooms in the house  
- `bathrooms` — Number of bathrooms  
- `sqft_living` — Square footage of the house  
- `sqft_lot` — Total lot size  
- `floors` — Number of floors  
- `waterfront` — Whether the house has a waterfront view (0/1)  
- `view` — Quality of the view  
- `condition` — Overall condition rating  
- `sqft_above` — Square footage of above-ground living space  
- `sqft_basement` — Square footage of basement  

**Categorical Feature:**
- `ocean_proximity` — Distance to the ocean / ocean access (encoded using **OneHotEncoder**)  

**Target Variable:**
- `price` — Price of the house (continuous numerical variable)

The dataset is **cleaned, processed, and preprocessed** to ensure the highest quality for machine learning. This includes:
- Handling missing values  
- Scaling numeric features (where necessary)  
- One-hot encoding categorical features while maintaining row alignment  

---

## **🔧 Tools & Libraries**
This project was implemented in **Python** using popular data science libraries:

- **pandas** — For data manipulation  
- **numpy** — For numerical operations  
- **scikit-learn** — For preprocessing, model building, and evaluation  
- **xgboost** — For advanced regression modeling  
- **matplotlib & seaborn** — For visualization and exploratory data analysis  

---

## **📊 Exploratory Data Analysis (EDA)**
The dataset was carefully analyzed to understand feature distributions, correlations, and their impact on house prices. Key insights from EDA include:

- Strong positive correlation between `sqft_living` and `price`  
- Moderate correlation between `grade`/`condition` and price  
- Some categorical variables (like `ocean_proximity`) significantly affect pricing  
- Outliers in price and square footage were handled to improve model performance  

Visualization techniques such as **scatter plots, heatmaps, and histograms** were used to gain insights into the data.

---

## **⚙️ Preprocessing**
Preprocessing is one of the most important steps for high-performing models. The following techniques were applied:

1. **Handling Categorical Variables:**
   - `ocean_proximity` was converted into numeric columns using **OneHotEncoder**  
   - Columns were reset and concatenated carefully to avoid row misalignment  

2. **Scaling Numeric Features:**
   - Features like `sqft_living`, `sqft_lot`, `sqft_above`, etc., were scaled using **StandardScaler** where necessary  

3. **Train-Test Split:**
   - Dataset was split into training and testing sets (80% train / 20% test)  
   - Split ensured **no data leakage** between train and test  

---

## **💡 Models Used**
Two machine learning models were implemented and compared:

### **1. Random Forest Regressor**
- Ensemble model using **multiple decision trees**  
- Handles nonlinear relationships well  
- Default parameters: `n_estimators=500`, `max_depth=25`  
- Achieved **R² ≈ 0.82**  

### **2. XGBoost Regressor**
- Gradient boosting algorithm that builds trees **sequentially** to correct previous errors  
- Parameters tuned: `n_estimators=300`, `learning_rate=0.05`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`  
- Achieved **R² ≈ 0.85**, slightly outperforming Random Forest  
- Highly efficient and robust to noise  

---

## **📈 Model Evaluation**
The models were evaluated using:

**Metric:**  
- **R² Score** (Coefficient of Determination) — measures how well the model predicts the target variable  

**Results:**
| Model               | R² Score |
|--------------------|----------|
| Random Forest      | 0.82     |
| XGBoost Regressor  | 0.85     |

**Interpretation:**  
- Random Forest provides a solid baseline  
- XGBoost slightly outperforms RF due to its sequential learning and ability to correct residual errors  
- Both models indicate **strong predictive capability**  

---

## **💡 Feature Importance**
Feature importance analysis helps understand which features contribute most to the prediction:

- `sqft_living` — most important predictor  
- `bathrooms` & `bedrooms` — key structural features  
- `floors`, `view`, and `waterfront` — moderate influence  
- `ocean_proximity` — categorical location-based influence  

> This insight can help real estate stakeholders focus on **high-impact factors** when pricing houses.

---

## **🚀 Challenges & Learnings**
- Ensuring **row alignment** after one-hot encoding categorical variables was critical — misalignment previously caused **R² ≈ 0.02**  
- High-cardinality categorical features (like cities or neighborhoods) can hurt tree-based models if encoded improperly  
- Scaling numeric features was optional for tree-based models, but still applied carefully for clarity  
- XGBoost requires **careful parameter tuning**, but even defaults outperform Random Forest in many cases  

---

## **📌 Key Takeaways**
- Proper **data preprocessing** is more important than fancy models  
- Tree-based models (Random Forest & XGBoost) handle **nonlinear relationships** well  
- XGBoost slightly outperforms Random Forest when features are meaningful  
- Always check **feature-target alignment**, it can make or break your model  

---

## **📈 Future Improvements**
- Include more granular features like `grade`, `sqft_living15`, `zipcode`  
- Apply **cross-validation** for more robust evaluation  
- Hyperparameter tuning using **GridSearchCV / RandomizedSearchCV**  
- Build a **web app** to predict house prices in real-time  

---

## **📎 References**
- [King County Housing Dataset](https://www.kaggle.com/harlfoxem/housesalesprediction)  
- Scikit-learn documentation: [https://scikit-learn.org/](https://scikit-learn.org/)  
- XGBoost documentation: [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)  

---

## **💡 Conclusion**
This project demonstrates the **end-to-end process of building a predictive ML model**:

1. Data understanding & cleaning  
2. Feature engineering & preprocessing  
3. Model selection & training (Random Forest & XGBoost)  
4. Model evaluation & comparison  
5. Feature importance analysis  
6. Insights for real-world decision-making  

Achieving **82–85% R²** demonstrates strong model performance and practical applicability for predicting house prices accurately.

---

## 📬 Contact Me

Connect with me through any of the platforms below:

<p align="center">
  <a href="https://www.linkedin.com/in/muskan-tariq-095a50282/" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" />
  </a>
  <a href="https://www.instagram.com/ai_enthusiast86?igsh=dnRyenAwdTBxdTZ6" target="_blank">
    <img src="https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white" />
  </a>
  <a href="mailto:muskantariq2003@gmail.com" target="_blank">
    <img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" />
  </a>
  <a href="https://www.youtube.com/@ai_enthusiast86?si=bYV1AgkBoCMVUBiK" target="_blank">
    <img src="https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white" />
  </a>
</p>

---
