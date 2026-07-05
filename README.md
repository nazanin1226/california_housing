# California Housing Price Prediction 🏠

This repository contains a professional, end-to-end Machine Learning project aimed at predicting median housing prices in California districts [94]. Grounded in the methodology established in Chapter 2 of Aurélien Géron's *Hands-On Machine Learning with Scikit-Learn and PyTorch*, this project moves systematically from framing the business problem to designing scalable preprocessing pipelines, training ensemble models, and planning for real-world deployment (MLOps) [91, 180].

---

## 📋 Project Checklist & Lifecycle
To maintain industry-standard rigor, this project adheres to a structured ML lifecycle [95]:
1. **Frame the Big Picture:** Align model objectives with business goals [96].
2. **Acquire the Data:** Programmatically fetch, extract, and load real-world data [119, 120].
3. **Exploratory Data Analysis (EDA):** Gain deep geographical, spatial, and correlation insights [131].
4. **Prepare the Data:** Construct robust, clean, and reusable preprocessing pipelines [143, 164].
5. **Model Selection & Evaluation:** Shortlist baseline and non-linear candidates [110, 114].
6. **Fine-Tuning:** Optimize hyperparameters via advanced search algorithms [116, 122].
7. **Rigorous Validation:** Evaluate on the test set and calculate confidence margins [128].
8. **Deployment & MLOps Planning:** Architect the system for serving, monitoring, and maintenance [174, 177].

---

## 🧠 1. Framing the Problem & Metrics
We frame this as a **supervised**, **multiple**, and **univariate regression** task [101]:
* **Supervised:** The model learns from historical data labeled with target values (`median_house_value`) [101].
* **Multiple Regression:** The system leverages multiple input attributes (e.g., longitude, latitude, median income, population) to formulate predictions [101].
* **Univariate Regression:** The goal is to predict a single continuous value per district [101].
* **Batch Learning:** The dataset fits comfortably in memory, and there is no real-time stream of incoming data requiring live model updates [101].

### Performance Measure
We select **Root Mean Squared Error (RMSE)** as our primary metric [102]:
$$\text{RMSE}(\mathbf{X}, y, h) = \sqrt{\frac{1}{m} \sum_{i=1}^{m} \left( h(\mathbf{x}^{(i)}) - y^{(i)} \right)^2}$$
RMSE heavily penalizes larger errors, making it highly suitable for real estate pricing where large discrepancies can lead to poor investment decisions [102, 96]. If our dataset was dominated by massive, uncorrectable outliers, we would instead consider **Mean Absolute Error (MAE)** [106].

---

## 📊 2. Exploratory Data Analysis & Insights
Our raw census dataset contains $20,640$ instances representing California districts [94, 122, 123]. Initial histograms and scatter plots revealed critical data characteristics:
* **Capped Attributes:** Both `housing_median_age` and `median_house_value` are capped (the target is capped at $\$500,000$) [125]. To prevent the model from learning that prices never exceed this limit, we must address this cap before deployment [125].
* **Income Scale:** `median_income` is scaled and capped at $15$ (representing roughly tens of thousands of dollars) [125].
* **Skewed Distributions:** Several attributes exhibit a strong right-skew, which can hinder some learning algorithms [126]. We apply log transformations to make these distributions more symmetric [163].
* **High-Density Areas:** Scatter plotting longitude and latitude with `alpha=0.2` clearly highlights high-density clusters around the Bay Area, Los Angeles, and San Diego [133].
* **Strong Correlations:** Calculating Pearson's $r$ correlation shows that `median_income` is the strongest predictor of `median_house_value` ($r \approx 0.69$) [136, 137, 141].
* **Feature Engineering:** We engineered ratios such as `rooms_per_house`, `bedrooms_ratio`, and `people_per_house` [141]. Notably, the new `bedrooms_ratio` has a strong negative correlation ($r \approx -0.26$) with housing values [141, 142].

---

## 🔀 3. Data Splitting: Overcoming Sampling Bias
A purely random split (`train_test_split`) on small or moderately sized datasets risks introducing sampling bias [614]. Since `median_income` is our most influential predictor [137, 141], we must ensure both training and test sets represent the overall population's income brackets [129].

We bin the continuous income data into five distinct strata and use **`StratifiedShuffleSplit`** to generate representative subsets [129]:
```python
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
```
This guarantees that the proportion of each income category in our subsets perfectly mirrors the overall dataset, eliminating sampling bias [129, 614].

---

## 🛠️ 4. Unified Preprocessing Pipeline
We constructed a modular, production-ready pipeline using Scikit-Learn’s **`ColumnTransformer`** to automate data cleaning and transformation [162, 163, 164]:

```
                                ┌──▶ SimpleImputer (median) ──▶ StandardScaler ──▶ remainder__housing_median_age
                                │
                                ├──▶ ratio_pipeline() ───────▶ StandardScaler ──▶ bedrooms__ratio, etc.
                                │
Raw Census Data ──▶ ColumnTransformer ┼──▶ log_pipeline ───────────▶ StandardScaler ──▶ log__median_income, etc.
                                │
                                ├──▶ ClusterSimilarity ──────▶ StandardScaler ──▶ geo__Cluster similarity
                                │
                                └──▶ cat_pipeline ───────────▶ OneHotEncoder ───▶ cat__ocean_proximity_INLAND, etc.
```

1. **Numerical Imputation:** Missing values in features (like `total_bedrooms`) are programmatically replaced with the feature's median using `SimpleImputer` [123, 146, 163].
2. **Categorical Encoding:** The textual feature `ocean_proximity` is encoded using `OneHotEncoder(handle_unknown="ignore")` [151, 163].
3. **Log Transformation:** Heavily right-skewed numerical features are log-transformed using `FunctionTransformer(np.log, feature_names_out="one-to-one")` to build bell-shaped distributions [126, 163, 615].
4. **Geographical Clustering:** We use a custom **`ClusterSimilarity`** transformer utilizing K-Means to identify $10$ major geographical clusters in California, converting longitude/latitude into Gaussian Radial Basis Function (RBF) similarity features [159, 160].
5. **Feature Scaling:** All features are standardized via `StandardScaler` to ensure our ML algorithms converge quickly and are not biased by varying scales [153, 163, 164, 224].

---

## 🤖 5. Model Selection, Validation, and Fine-Tuning
We evaluated multiple regression models using **K-Fold Cross-Validation** ($k=10$) on the stratified training set to obtain reliable estimates and measure performance precision (standard deviation) [112, 114]:

| Model | Train RMSE | Cross-Validation Mean RMSE | CV Std Dev | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | $\$68,973$ | $\$70,003$ | $\$4,182$ | Suffers from severe **Underfitting** [110, 114]. |
| **Decision Tree** | $\$0.0$ | $\$66,574$ | $\$1,103$ | Suffers from severe **Overfitting** [111, 114]. |
| **Random Forest** | $\$17,551$ | $\$47,038$ | $\$1,021$ | An **Ensemble Model** that averages tree errors [114]. Best baseline [115]. |

### Hyperparameter Fine-Tuning
We utilized **`GridSearchCV`** and **`RandomizedSearchCV`** to explore combinations of preprocessing parameters (e.g., `n_clusters` in K-Means) and model hyperparameters (e.g., `max_features` in Random Forest) simultaneously [116, 122, 168, 615]. 

By setting `refit=True` (default), Scikit-Learn automatically retrains our final best model on the **entire 100% training set** using the optimized parameters, maximizing historical learning before testing [120].

---

## 🎯 6. Final Evaluation & Validation Rigor
To ensure statistical confidence, we evaluated our final model on the untouched **Stratified Test Set** [128].
* **Point Estimate:** The final model achieved an impressive **RMSE of $\$41,445$** [128].
* **95% Confidence Interval:** To determine the precision of this estimate, we computed a 95% confidence interval using **`scipy.stats.bootstrap()`** [128]:
  $$\text{95% Confidence Interval} = [\$39,521, \; \$43,702]$$

This narrow interval provides scientific proof that the model’s performance is highly stable and generalizeable to unseen market data [128].

---

## 🚀 7. MLOps: Deployment, Monitoring, and Maintenance
Building a model is only half the battle. To launch this system in production, we plan the following architecture [174, 177]:
1. **Model Persistence:** Serialize the trained preprocessing and Random Forest pipeline into a portable file using `joblib` [174]:
   ```python
   import joblib
   joblib.dump(final_model, "california_housing_model.pkl")
   ```
2. **Serving Infrastructure:** Deploy the saved model as a microservice (REST API) using frameworks like FastAPI or cloud platforms like Google Cloud Vertex AI to serve predictions via JSON [176, 177].
3. **Monitoring for Data Drift:** Set up telemetry to watch for input data quality [177]. If the incoming census data changes (e.g., median income shifts due to economic drift, or missing values spike), trigger alerts [177].
4. **Automated Retraining (Continuous Training - CT):** Schedule automated scripts to regularly ingest fresh data, label it, retrain the pipeline, evaluate it against the current production model on a fresh test set, and automatically deploy the updated model if performance passes validation [178].

---

## 🛠️ How to Recreate this Project (Reproducibility)
To replicate these results locally, follow these steps [173]:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/california-housing.git
   cd california-housing
   ```

2. **Set Up a Virtual Environment & Install Dependencies:**
   Ensure you have Python 3.12+ installed [233].
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Install Requirements:**
   Create a `requirements.txt` containing [173]:
   ```text
   scikit-learn>=1.4.0
   pandas>=2.2.0
   numpy>=1.26.0
   matplotlib>=3.8.0
   scipy>=1.12.0
   joblib>=1.3.0
   ```

4. **Run the Notebook:**
   Launch Jupyter or open the file in Google Colab to run the cells sequentially [44, 108].
   ```bash
   jupyter notebook california_housing.ipynb
   ```
