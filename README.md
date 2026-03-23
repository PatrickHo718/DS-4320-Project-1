# DS-4320-Project-1: Detecting Credit Card Fraud

### Executive Summary

<br>

<br>

---

### Name - Mu Patrick Ho
### NetID - bqu3tr
### DOI - [url](url)
### Press Release
[**A Machine Learning Approach to Scoring Online Transaction Risk**](url)
### Data - [link to data](https://myuva-my.sharepoint.com/:f:/g/personal/bqu3tr_virginia_edu/IgBAbNXHsneGQokG4ZUMr-LjATbpy9pAfvE82uTdzAjVNNg?e=uOZZ61)
### Pipeline - [analysis code](url)
### License - [MIT](LICENSE)


---
<br>

<br>

## Problem Definition
### General and Specific Problem
* **General Problem:** Credit card fraud is a growing threat in the financial industry that is costing businesses and consumers tens of billions of dollars every year. As online transactions have become increasingly common, fraudsters have adapted their tactics to exploit vulnerabilities in digital payment systems, making detection more challenging than ever.
* **Specific Problem:** Can we predict the probability of fraud in online transactions by analyzing card and payment behavioral patterns to assist bank fraud analysts in prioritizing high-risk transactions for review?
### Rationale
The general problem of credit card fraud detection was refined to focus specifically on card and payment behavior because these features, such as card type, billing address, purchase amount, and product category, represent the most direct signals available to a bank at the moment of a transaction. Unlike device or identity features which may not always be captured, card and payment data is consistently present across all transactions, making it a more reliable foundation for a scoring model. This refinement also aligns closely with how bank fraud analysts actually work because they typically review flagged transactions using payment metadata rather than technical device fingerprints.
### Motivation
Online transaction fraud costs financial institutions billions of dollars annually. The burden of manually reviewing suspicious activity falls heavily on fraud analysts, which it can be very monotonous work. A risk scoring model that ranks transactions by their probability of being fraudulent allows analysts to focus their time on the highest-risk cases first, improving both efficiency and fraud catch rates.
### Press Release Headline and Link
[**A Machine Learning Approach to Scoring Online Transaction Risk**](url)

<br>

<br>

---

## Domain Exposition

### Terminology

| Term | Definition |
|------|------------|
| **Concept Drift** | A challenge in fraud detection where the statistical properties of the target variable change over time as cardholders' behavioral patterns are not static |
| **SMOTE** | An oversampling technique that addresses class imbalance by generating synthetic examples of the minority class (fraudulent transactions) to help train ML models more effectively |
| **Card-Not-Present (CNP) Fraud** | A type of fraud where the physical card is not available during the transaction, common in e-commerce, mail-order, or telephone-order scenarios |
| **Matthews Correlation Coefficient (MCC)** | A ML metric used to evaluate binary classifiers on imbalanced datasets; considered balanced because it accounts for true and false positives and negatives |
| **Bayesian Optimization** | A method used to efficiently tune hyperparameters of ML models by finding optimal configurations in fewer training iterations than exhaustive search |
| **Know Your Customer (KYC)** | An Anti-Money Laundering (AML) compliance process that verifies customer identities by collecting data and evaluating financial behavior to detect anomalies and assess risk |
| **Sliding Window Strategy** | A technique that aggregates transaction data over a specific time period to extract behavioral patterns such as average, maximum, and minimum transaction amounts within cardholder profiles |
| **Ensemble Learning** | A ML approach that combines multiple classifiers to achieve better prediction results than any single model alone |
| **Structuring (Smurfing)** | A money laundering tactic where large transactions are broken into smaller amounts to avoid detection by financial monitoring systems |
| **Feature Engineering** | The process of selecting, extracting, or transforming raw data into meaningful features that improve the performance of ML models |

### Paragraph
Credit card fraud detection's domain is in financial services, where the goal is to identify illegitimate transactions before they result in financial loss. When a cardholder makes an online purchase, the transaction flows through the merchant's payment processor, the card network that may be Visa or Mastercard, and finally the issuing bank in a matter of seconds. Traditional systems flag transactions using fixed thresholds, but these fail to capture the complicated behavioral patterns that separate legitimate from fraudulent activity, which leads to high false positive rates and false negative rates. Machine learning models trained on historical transaction data address this by learning these patterns automatically. They are evaluated using AUC-ROC, which measures how well a model distinguishes fraud from non-fraud across all decision thresholds. Financial institutions pursuing these solutions must also comply with regulations such as the Bank Secrecy Act (BSA) that govern fraud liability and require that detection decisions remain interpretable to both analysts and regulators.

### Reading
| Title | Brief Description | Link |
|---|---|---|
| Using AI to Secure The Future of Payments | Shows emerging trends in the payment ecosystem and highlights how AI-driven solutions are critical for securing diverse payment rails | [Link](https://drive.google.com/file/d/1D_ddYndCPKabJBlFiNC0xDuCawfEY5Zh/view?usp=drive_link) |
| Credit Card Fraud Detection using Machine Learning Algorithm | Proposes cardholder clustering and a sliding window strategy to analyze behavioral patterns and mitigate the challenge of concept drift | [Link](https://drive.google.com/file/d/186PeYaHyIRb8bLsyRNeJjB3rhZ6_CPRT/view?usp=drive_link) |
| CREDIT CARD FRAUD DETECTION USING MACHINE LEARNING: A STUDY | Suggests  Hidden Markov Models and Neural Networks to improve the detection of diverse credit card fraud types | [Link](https://drive.google.com/file/d/1EC0Y52bF0gWBwL8-Je3TzE5pjFMhd8iP/view?usp=drive_link) |
| What is anti-money laundering? A complete guide to AML compliance. | Covers AML compliance to stress the importance of KYC/CDD frameworks and the integration of AI to detect financial threats | [Link](https://drive.google.com/file/d/1DYNGWO4N-l07rEz2Vwi4kL-27heELBPc/view?usp=drive_link) |
| Fraud Detection in Banking Data by Machine Learning Techniques | Uses Bayesian optimization for hyperparameter tuning and ensemble learning for highly imbalanced banking datasets | [Link](https://drive.google.com/file/d/1eamB_CzPW37evvTxPhOTjPR3JB-fs732/view?usp=drive_link) |

---
<br>

<br>

## Data Creation

### Paragraph
The dataset used in this project is the IEEE-CIS Fraud Detection dataset, which is originally published as part of a Kaggle competition hosted by the IEEE Computational Intelligence Society. The data was accessed and downloaded from Kaggle (https://www.kaggle.com/competitions/ieee-fraud-detection) using a registered Kaggle account. The dataset is provided in four CSV files: train_transaction.csv, train_identity.csv, test_transaction.csv, and test_identity.csv, which were downloaded as a compressed zip archive and extracted locally before being uploaded to a UVA OneDrive folder for storage and access. The transaction files contain features related to payment behavior including card information, transaction amount, product category, and a suite of anonymized engineered features (V1-V339), while the identity files contain device and network-related features associated with a subset of transactions.

The two file types are linked by TransactionID as the primary key, with the identity tables representing a subset of transactions for which identity information was available. The target variable isFraud is provided only in the training set, as the test set was originally intended for Kaggle competition submissions. For the purposes of this project, analysis is conducted primarily on the training sets where ground truth labels are available. The raw files were reorganized into four logical relational tables, including transactions, cards, email, and identity, to better reflect the relational structure of the data and support SQL-based querying via DuckDB.

### Code
| File | Description | Link |
|------|-------------|------|
| `main.py` | Loads all csv files, splits the transaction data into the `transactions`, `cards`, and `email` relational tables, and saves the identity data as the `identity` table | [main.py](https://drive.google.com/file/d/1bn4N_3oNwKEi8UqOmZEiz7M3mKLSfUMk/view?usp=drive_link) |

### Bias Identification

Several sources of bias may have been introduced in the data collection process. First, there is selection bias as the dataset was collected exclusively through Vesta Corporation's payment platform. It reflects the fraud patterns specific to that platform's merchant base and customer demographics, so fraud behavior on other platforms may differ significantly. Second, the dataset has a severe class imbalance with fraudulent transactions representing only about 3.5% of all records, which reflects real-world fraud rates but can bias a model toward predicting every transaction as legitimate. Third, many identity features (id_01 through id_38) are missing for a large proportion of transactions, which may reflect a systematic gap. For example, certain device types or merchants may not capture identity signals, meaning the absence of data is itself informative and potentially biased toward certain user groups. Finally, the anonymization of features (V1-V339) prevents full understanding of what those features represent, making it difficult to identify additional sources of bias.

### Bias Mitigation

Class imbalance is addressed directly through SMOTE (Synthetic Minority Oversampling Technique) that generates synthetic fraud examples during training to prevent the model from being biased toward the majority class. Missing identity features are handled through imputation strategies or by allowing tree-based ensemble models to handle missing values natively, reducing the risk that missingness introduces systematic error. To account for the platform-specific nature of the data, model performance is evaluated using AUC-ROC rather than raw accuracy, as AUC-ROC is less sensitive to class imbalance and provides a more honest picture of model performance. The anonymized V-features are retained in the model but their contributions are monitored through feature importance analysis to detect any unexpected over-reliance on features whose meaning cannot be verified.

### Rationale

* Why training set only: The test set contains no `isFraud` labels, making it unsuitable for supervised learning evaluation. All analysis is therefore conducted on the training set where ground truth is available.

* Why split into 4 relational tables: Transaction data was split into `transactions`, `cards`, `email`, and `identity` tables to better reflect the relational structure of real banking systems, where card, identity, and transaction information are stored separately, and to enable meaningful SQL-based querying via DuckDB.

* Why retain `TransactionDT` as timedelta: The reference datetime is not provided, meaning any conversion to an actual timestamp would introduce fabricated precision. The raw timedelta is retained to preserve honesty about what the feature represents.

* Why retain anonymized V-features: Removing the V1-V339 features would discard a significant portion of the engineered signal in the dataset. Despite their opacity, they are retained and monitored through feature importance analysis to detect unexpected over-reliance.

* Why SMOTE for class imbalance: Fraudulent transactions represent only around 3.5% of records. Without resampling, the model would be biased toward predicting every transaction as legitimate. SMOTE generates synthetic fraud examples to correct this during training.

<br>

<br>

---

### Schema

### Data

| Table | Description | Link |
|-------|-------------|------|
| `transaction.csv` | Core transaction records including TransactionID, amount, timestamp, product category, fraud label, and anonymized V features | [transaction.csv](https://drive.google.com/file/d/1sMk_K2TJZ64fwVkPTjKM0OqMjTUdP-oh/view?usp=drive_link) |
| `cards.csv` | Card and billing address features associated with each transaction | [cards.csv](https://drive.google.com/file/d/1eXkBC5SjLjX6wKRssuzyvrgb1rByoB4E/view?usp=drive_link) |
| `email.csv` | Purchaser and recipient email domains and match status features (M1–M9) | [email.csv](https://drive.google.com/file/d/1QSH0CyCN0z18GILAMMM3hTmkAp4B3OfB/view?usp=drive_link) |
| `identity.csv` | Device and network identity features for a subset of transactions | [identity.csv](https://drive.google.com/file/d/1zT-mdrKUJXR5lWv6GmBsdFDrSMmRVlft/view?usp=drive_link) |

### Data Dictionary

* Table includes core features of the 406 columns

| Name | Data Type | Description | Example |
|------|-----------|-------------|---------|
| TransactionID | Integer | Unique identifier for each transaction | 2987000 |
| isFraud | Binary (0/1) | Target variable — 1 indicates fraudulent transaction | 0 |
| TransactionDT | Integer | Timedelta in seconds from a reference datetime | 86400 |
| TransactionAmt | Float | Transaction amount in USD | 68.5 |
| ProductCD | String | Product category code | W |
| card1 | Integer | Card identifier feature 1 | 13926 |
| card4 | String | Card network (Visa, Mastercard, etc.) | visa |
| addr1 | Float | Billing zip code | 315.0 |
| P_emaildomain | String | Purchaser email domain | gmail.com |
| R_emaildomain | String | Recipient email domain | yahoo.com |
| M1 | String | Match status feature 1 | T |
| DeviceType | String | Device category used in transaction | mobile |
| DeviceInfo | String | Specific device model or OS | iOS Device |
| id_01 | Float | Anonymized identity feature | -5.0 |

### Quantification of Uncertainty

| Feature | Min | Max | Mean | Std Dev | Missing (%) | Notes |
|---------|-----|-----|------|---------|-------------|-------|
| TransactionDT | 86,400 | 34,200,000 | 16,403,800 | 10,816,100 | 0% | Timedelta only; wide range reflects full dataset time span |
| TransactionAmt | 0.018 | 31,937.40 | 134.89 | 242.24 | 0% | Right-skewed; large outliers present |
| addr1 | 100 | 540 | 291.24 | 101.89 | 11.97% | Billing zip code region; moderate missingness may indicate anonymization |
| id_01 | -100 | 0 | -11.33 | 14.51 | 50.41% | High missingness; absence itself may be informative |
| id_02 | 2 | 999,869 | 192,659 | 182,613 | 52.13% | Extremely wide range; high missingness |

