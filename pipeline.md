# Project 1 Code Pipeline


```python
!pip install duckdb pandas pyarrow scikit-learn imbalanced-learn xgboost lightgbm bayesian-optimization matplotlib seaborn requests
```

    Requirement already satisfied: duckdb in ./venv/lib/python3.10/site-packages (1.5.1)
    Requirement already satisfied: pandas in ./venv/lib/python3.10/site-packages (2.3.3)
    Requirement already satisfied: pyarrow in ./venv/lib/python3.10/site-packages (23.0.1)
    Requirement already satisfied: scikit-learn in ./venv/lib/python3.10/site-packages (1.7.2)
    Requirement already satisfied: imbalanced-learn in ./venv/lib/python3.10/site-packages (0.14.1)
    Requirement already satisfied: xgboost in ./venv/lib/python3.10/site-packages (3.2.0)
    Requirement already satisfied: lightgbm in ./venv/lib/python3.10/site-packages (4.6.0)
    Requirement already satisfied: bayesian-optimization in ./venv/lib/python3.10/site-packages (3.2.1)
    Requirement already satisfied: matplotlib in ./venv/lib/python3.10/site-packages (3.10.8)
    Requirement already satisfied: seaborn in ./venv/lib/python3.10/site-packages (0.13.2)
    Requirement already satisfied: requests in ./venv/lib/python3.10/site-packages (2.32.5)
    Requirement already satisfied: python-dateutil>=2.8.2 in ./venv/lib/python3.10/site-packages (from pandas) (2.9.0.post0)
    Requirement already satisfied: tzdata>=2022.7 in ./venv/lib/python3.10/site-packages (from pandas) (2025.3)
    Requirement already satisfied: pytz>=2020.1 in ./venv/lib/python3.10/site-packages (from pandas) (2026.1.post1)
    Requirement already satisfied: numpy>=1.22.4 in ./venv/lib/python3.10/site-packages (from pandas) (2.2.6)
    Requirement already satisfied: joblib>=1.2.0 in ./venv/lib/python3.10/site-packages (from scikit-learn) (1.5.3)
    Requirement already satisfied: threadpoolctl>=3.1.0 in ./venv/lib/python3.10/site-packages (from scikit-learn) (3.6.0)
    Requirement already satisfied: scipy>=1.8.0 in ./venv/lib/python3.10/site-packages (from scikit-learn) (1.15.3)
    Requirement already satisfied: sklearn-compat<0.2,>=0.1.5 in ./venv/lib/python3.10/site-packages (from imbalanced-learn) (0.1.5)
    Requirement already satisfied: nvidia-nccl-cu12 in ./venv/lib/python3.10/site-packages (from xgboost) (2.29.7)
    Requirement already satisfied: packaging>=20.0 in ./venv/lib/python3.10/site-packages (from bayesian-optimization) (26.0)
    Requirement already satisfied: colorama>=0.4.6 in ./venv/lib/python3.10/site-packages (from bayesian-optimization) (0.4.6)
    Requirement already satisfied: kiwisolver>=1.3.1 in ./venv/lib/python3.10/site-packages (from matplotlib) (1.5.0)
    Requirement already satisfied: fonttools>=4.22.0 in ./venv/lib/python3.10/site-packages (from matplotlib) (4.62.1)
    Requirement already satisfied: contourpy>=1.0.1 in ./venv/lib/python3.10/site-packages (from matplotlib) (1.3.2)
    Requirement already satisfied: pillow>=8 in ./venv/lib/python3.10/site-packages (from matplotlib) (12.1.1)
    Requirement already satisfied: cycler>=0.10 in ./venv/lib/python3.10/site-packages (from matplotlib) (0.12.1)
    Requirement already satisfied: pyparsing>=3 in ./venv/lib/python3.10/site-packages (from matplotlib) (3.3.2)
    Requirement already satisfied: urllib3<3,>=1.21.1 in ./venv/lib/python3.10/site-packages (from requests) (2.6.3)
    Requirement already satisfied: charset_normalizer<4,>=2 in ./venv/lib/python3.10/site-packages (from requests) (3.4.6)
    Requirement already satisfied: idna<4,>=2.5 in ./venv/lib/python3.10/site-packages (from requests) (3.11)
    Requirement already satisfied: certifi>=2017.4.17 in ./venv/lib/python3.10/site-packages (from requests) (2026.2.25)
    Requirement already satisfied: six>=1.5 in ./venv/lib/python3.10/site-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)


## Imports and Logging


```python
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import io
import requests

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (roc_auc_score, confusion_matrix, 
                             ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from bayes_opt import BayesianOptimization

# Logging
logging.basicConfig(
    filename='pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)
log.info("Pipeline started")
print("Imports successful")
```

    Imports successful


## Load Files


```python
Urls = {
    'transaction': 'https://myuva-my.sharepoint.com/:u:/g/personal/bqu3tr_virginia_edu/IQA0m477oIf-RKPajvdNqdR5AXpsLX6vtr8NlrCK4-5UzLE?download=1',
    'cards': 'https://myuva-my.sharepoint.com/:u:/g/personal/bqu3tr_virginia_edu/IQC7ytwes3e9Q4THqdTdyLl3ASwh6O6OHBHvzi4SDoc7g1k?download=1',
    'email': 'https://myuva-my.sharepoint.com/:u:/g/personal/bqu3tr_virginia_edu/IQCbzr0lDS4yTJTAEQxTZaFqAWBKuDSTrK4H6siEgs5Fw_8?download=1',
    'identity': 'https://myuva-my.sharepoint.com/:u:/g/personal/bqu3tr_virginia_edu/IQCtneexJFggSr_cdos1vysfAca77zSoaBFJqXNQ5-0xcWU?download=1'
}

def load_parquet_from_onedrive(url: str, name: str) -> pd.DataFrame:
    """Download a parquet file from a OneDrive sharing link."""
    try:
        log.info(f"Downloading {name} from OneDrive...")
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=120)
        response.raise_for_status()
        df = pd.read_parquet(io.BytesIO(response.content))
        log.info(f"{name} loaded: {df.shape}")
        return df
    except requests.exceptions.RequestException as e:
        log.error(f"Failed to download {name}: {e}")
        raise
    except Exception as e:
        log.error(f"Failed to read parquet for {name}: {e}")
        raise

dfs = {name: load_parquet_from_onedrive(url, name) for name, url in Urls.items()}

print("Tables Loaded:")
for name, df in dfs.items():
    print(f"{name}: {df.shape}")
```

    Tables Loaded:
    transaction: (1097231, 344)
    cards: (1097231, 9)
    email: (1097231, 12)
    identity: (286140, 79)


## DuckDB SQL Query


```python
try: 
    con = duckdb.connect()

    for name, df in dfs.items():
        con.register(name, df)
        log.info(f"Registered {name} in DuckDB")
    
    query = """
        SELECT
            t.TransactionID,
            t.isFraud,
            t.TransactionAmt,
            t.TransactionDT,
            t.ProductCD,
            c.card1, c.card2, c.card3, c.card4, c.card5, c.card6,
            c.addr1, c.addr2,
            e.P_emaildomain, e.R_emaildomain,
            e.M1, e.M2, e.M3, e.M4, e.M5, e.M6, e.M7, e.M8, e.M9,
            i.DeviceType, i.DeviceInfo,
            i.id_01, i.id_02, i.id_03, i.id_04, i.id_05,
            i.id_06, i.id_07, i.id_08, i.id_09, i.id_10
        FROM transaction t
        LEFT JOIN cards c ON t.TransactionID = c.TransactionID
        LEFT JOIN email e ON t.TransactionID = e.TransactionID
        LEFT JOIN identity i ON t.TransactionID = i.TransactionID
    """

    df = con.execute(query).fetchdf()
    log.info(f"Joined DataFrame shape: {df.shape}")

except Exception as e:
    log.error(f"Error during DuckDB operations: {e}")
    raise
finally:
    con.close()
```

## Feature Engineering


```python
try:
    # Encode categorical features
    cat_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain','R_emaildomain', 'DeviceType', 'DeviceInfo',
                'M1','M2','M3','M4','M5','M6','M7','M8','M9']
    
    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
            log.info(f"Encoded {col}")
    
    # Fill missing numeric with median
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['TransactionID', 'isFraud']]
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    log.info("Filled missing numeric values with median")

    # Drop missing isFraud
    df = df.dropna(subset=['isFraud'])
    log.info(f"DataFrame shape after dropping missing target: {df.shape}")

    # Separate features and target
    X = df.drop(columns=['TransactionID', 'isFraud'])
    y = df['isFraud']

    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
except Exception as e:
    log.error(f"Error during preprocessing: {e}")
    raise    
```

    Features shape: (590540, 34), Target shape: (590540,)


## Train Test Split


```python
try: 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    log.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    log.info(f"After SMOTE - Train shape: {X_train_sm.shape}, Target distribution: {np.bincount(y_train_sm)}")
    print(f"After SMOTE - Train shape: {X_train_sm.shape}, Target distribution: {np.bincount(y_train_sm)}")
except Exception as e:
    log.error(f"Error during train-test split or SMOTE: {e}")
    raise
```

    Train shape: (472432, 34), Test shape: (118108, 34)
    After SMOTE - Train shape: (911804, 34), Target distribution: [455902 455902]


    /tmp/ipykernel_10440/3130091131.py:9: DeprecationWarning: Non-integer input passed to bincount. In a future version of NumPy, this will be an error. (Deprecated NumPy 2.1)
      log.info(f"After SMOTE - Train shape: {X_train_sm.shape}, Target distribution: {np.bincount(y_train_sm)}")
    /tmp/ipykernel_10440/3130091131.py:10: DeprecationWarning: Non-integer input passed to bincount. In a future version of NumPy, this will be an error. (Deprecated NumPy 2.1)
      print(f"After SMOTE - Train shape: {X_train_sm.shape}, Target distribution: {np.bincount(y_train_sm)}")


## Bayesian Optimization


```python
try:
    # XGBoost
    def xgb_eval(n_estimators, max_depth, learning_rate, subsample):
        model = XGBClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=42,
            eval_metric='auc',
            verbosity=0
        )
        return cross_val_score(model, X_train_sm, y_train_sm, cv=3, scoring='roc_auc', n_jobs=-1).mean()
    
    xgb_bo = BayesianOptimization(
        f=xgb_eval,
        pbounds={
            'n_estimators': (100, 500),
            'max_depth': (3, 8),
            'learning_rate': (0.01, 0.3),
            'subsample': (0.6, 1.0)
        },
        random_state=42
    )
    xgb_bo.maximize(init_points=5, n_iter=10)
    best_xgb = xgb_bo.max['params']
    log.info(f"XGBoost best params: {best_xgb}")
    print(f"XGBoost best params: {best_xgb}")

    # LightGBM
    def lgbm_eval(n_estimators, max_depth, learning_rate, num_leaves):
        model = LGBMClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            learning_rate=learning_rate,
            num_leaves=int(num_leaves),
            random_state=42,
            verbosity=-1
        )
        return cross_val_score(
            model, X_train_sm, y_train_sm,
            cv=3, scoring='roc_auc', n_jobs=-1
        ).mean()
    
    lgbm_optimizer = BayesianOptimization(
        f=lgbm_eval,
        pbounds={
            'n_estimators': (100, 500),
            'max_depth': (3, 8),
            'learning_rate': (0.01, 0.3),
            'num_leaves': (20, 100),
        },
        random_state=42, 
        verbose=0
    )
    lgbm_optimizer.maximize(init_points=5, n_iter=10)
    best_lgbm = lgbm_optimizer.max['params']
    log.info(f"Best LGBM params: {best_lgbm}")
    print(f"Best LGBM params: {best_lgbm}")

except Exception as e:
    log.error(f"Error during XGBoost optimization: {e}")
    raise
```

    |   iter    |  target   | n_esti... | max_depth | learni... | subsample |
    -------------------------------------------------------------------------
    | [39m1        [39m | [39m0.9956689[39m | [39m249.81604[39m | [39m7.7535715[39m | [39m0.2222782[39m | [39m0.8394633[39m |
    | [39m2        [39m | [39m0.9432473[39m | [39m162.40745[39m | [39m3.7799726[39m | [39m0.0268442[39m | [39m0.9464704[39m |
    | [39m3        [39m | [39m0.9794623[39m | [39m340.44600[39m | [39m6.5403628[39m | [39m0.0159695[39m | [39m0.9879639[39m |
    | [39m4        [39m | [39m0.9844878[39m | [39m432.97705[39m | [39m4.0616955[39m | [39m0.0627292[39m | [39m0.6733618[39m |
    | [39m5        [39m | [39m0.9891464[39m | [39m221.69689[39m | [39m5.6237821[39m | [39m0.1352640[39m | [39m0.7164916[39m |
    | [39m6        [39m | [39m0.9955510[39m | [39m499.88706[39m | [39m6.7939707[39m | [39m0.1787826[39m | [39m0.8301012[39m |
    | [39m7        [39m | [39m0.9939140[39m | [39m250.37158[39m | [39m6.4126587[39m | [39m0.1865202[39m | [39m0.6594187[39m |
    | [39m8        [39m | [39m0.9871250[39m | [39m472.07116[39m | [39m3.0426091[39m | [39m0.1839634[39m | [39m0.8522011[39m |
    | [35m9        [39m | [35m0.9966049[39m | [35m289.98962[39m | [35m8.0      [39m | [35m0.3      [39m | [35m1.0      [39m |
    | [39m10       [39m | [39m0.9957790[39m | [39m390.06127[39m | [39m7.8239405[39m | [39m0.1471951[39m | [39m0.9529994[39m |
    | [39m11       [39m | [39m0.9714690[39m | [39m100.0    [39m | [39m7.6627160[39m | [39m0.0258197[39m | [39m0.9954716[39m |
    | [39m12       [39m | [39m0.9872805[39m | [39m309.88785[39m | [39m3.0      [39m | [39m0.3      [39m | [39m0.6      [39m |
    | [39m13       [39m | [39m0.9883273[39m | [39m370.88347[39m | [39m3.0      [39m | [39m0.3      [39m | [39m0.6      [39m |
    | [39m14       [39m | [39m0.9965450[39m | [39m271.61287[39m | [39m8.0      [39m | [39m0.3      [39m | [39m1.0      [39m |
    | [39m15       [39m | [39m0.9888044[39m | [39m406.46345[39m | [39m3.0      [39m | [39m0.3      [39m | [39m0.6      [39m |
    =========================================================================
    XGBoost best params: {'n_estimators': np.float64(289.9896201183945), 'max_depth': np.float64(8.0), 'learning_rate': np.float64(0.3), 'subsample': np.float64(1.0)}
    Best LGBM params: {'n_estimators': np.float64(279.3281667809466), 'max_depth': np.float64(8.0), 'learning_rate': np.float64(0.3), 'num_leaves': np.float64(95.83480246150792)}


## Ensemble Model and Evaluation


```python
try:
    # Build tuned models
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)

    xgb = XGBClassifier(
        n_estimators=int(best_xgb['n_estimators']),
        max_depth=int(best_xgb['max_depth']),
        learning_rate=best_xgb['learning_rate'],
        subsample=best_xgb['subsample'],
        random_state=42, 
        eval_metric='auc', 
        verbosity=0
    )

    lgbm = LGBMClassifier(
        n_estimators=int(best_lgbm['n_estimators']),
        max_depth=int(best_lgbm['max_depth']),
        learning_rate=best_lgbm['learning_rate'],
        num_leaves=int(best_lgbm['num_leaves']),
        random_state=42, 
        verbosity=-1
    )

    # Soft voting ensemble
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('lgbm', lgbm)],
        voting='soft',
        n_jobs=-1
    )
    ensemble.fit(X_train_sm, y_train_sm)
    log.info("Ensemble model trained successfully")

    # Evaluate 
    y_pred = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    log.info(f"Ensemble AUC: {auc:.4f}")
    print(f"Ensemble AUC: {auc:.4f}")

except Exception as e:
    log.error(f"Error during model training or evaluation: {e}")
    raise
```

    Ensemble AUC: 0.9493


## Analysis Rationale

* **Why Bayesian Optimization for hyperparameter tuning:** Grid search is computationally expensive on a dataset of this size. Bayesian optimization was chosen because it learns from previous iterations to narrow the search space, finding strong hyperparameter configurations in significantly fewer evaluations. This makes it well suited for large datasets where each training run is costly.

* **Why soft voting ensemble:** There is not a single model that is universally optimal for fraud detection. Random Forest is robust to noise and handles missing values well. XGBoost captures complex non-linear interactions through gradient boosting. LightGBM is highly efficient on large datasets with many features. Combining all three through soft voting, averaging predicted probabilities rather than majority vote, produces a more stable and accurate risk score than any individual model, and reduces the risk of overfitting to one model's weaknesses.

* **Why AUC-ROC as the metric:** Raw accuracy is misleading on imbalanced datasets. A model that predicts every transaction as legitimate achieves high accuracy while catching zero fraud. AUC-ROC measures the model's ability to rank fraudulent transactions above legitimate ones across all decision thresholds, making it a more reliable measure of performance for this problem.

## Visualization


```python
try:
    # Extract fitted rf from ensemble
    rf_fitted = ensemble.named_estimators_['rf']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ROC Curve
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[0])
    axes[0].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[0].spines[['top', 'right']].set_visible(False)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=['Legit', 'Fraud']).plot(ax=axes[1], cmap='Blues', colorbar=False)
    axes[1].set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    # Feature Importance
    rf_importance = pd.Series(rf_fitted.feature_importances_, index=X.columns).nlargest(10).sort_values()
    rf_importance.plot(kind='barh', ax=axes[2], color='teal')
    axes[2].set_title('Top 10 RF Feature Importances', fontsize=14, fontweight='bold')
    axes[2].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=300)
    plt.show()
    log.info("Evaluation plots generated and saved successfully")

except Exception as e:
    log.error(f"Error during visualization: {e}")
    raise
```


    
![png](pipeline_files/pipeline_18_0.png)
    


## Visualization Rationale 

* **Why a ROC Curve:** The ROC curve visualizes the tradeoff between true positive rate (fraud correctly flagged) and false positive rate (legitimate transactions incorrectly flagged) across all possible decision thresholds. This is very informative to fraud analysts who need to choose an operating threshold based on their tolerance for false alarms versus missed fraud. The AUC of 0.95 is displayed on the curve to give a summary of model quality. The steep rise toward the top left corner confirms the model discriminates well between fraud and legitimate transactions across all thresholds.

* **Why a Confusion Matrix:** The confusion matrix at the default 0.50 threshold shows the exact counts of 113,639 true negatives, 2,127 true positives, 336 false positives, and 2,006 false negatives, giving the fraud analyst a glimpse of how many fraud cases are caught versus missed. The low false positive count is particularly meaningful for the end user as it means very few legitimate customers would be incorrectly disrupted.

* **Why a Feature Importance Bar Chart:** Showing the top 10 most predictive features from the Random Forest makes the model more interpretable with M4 (identity match status) and card6 (card type) as the strongest predictors, followed by TransactionAmt and TransactionDT, confirming that both payment behavior and identity signals drive fraud risk. A horizontal bar chart was chosen over a vertical one because feature names are long and read more smoothly on a horizontal axis. 
