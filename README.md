# Instamart Customer Intelligence & Support Copilot

An enterprise-grade Customer Intelligence & Support Copilot designed for Swiggy Instamart metro operations. This end-to-end repository resolves three critical operational bottlenecks identified in weekly leadership reviews: untrusted raw data exports, unaddressed customer lapse/churn, and support ticket triage flooding with sensitive personal data.

The project is structured into four connected pillars:
- **Part A (Data Foundation)**: Audits, standardises, cleans, and stores 13,000+ raw orders in a relational SQLite database (`instamart.db`) and validates financial/operational metrics with analytical SQL.
- **Part B (Customer Segments & Lapse Prediction)**: Builds an RFM segmentation framework with K-Means ($K=4$) and PCA projection, followed by a leak-free Machine Learning classification pipeline predicting 30-day customer lapse with Random Forest and Logistic Regression.
- **Part C (Support Ticket Triage with Guardrails)**: Implements privacy-preserving PII redaction (email, phone, order ID), resilient API calling with exponential backoff and jitter, hierarchical output validation, token budgeting, and segment-aware ticket routing.
- **Part D (Policy RAG Assistant & ReAct Refund Agent)**: Builds an overlapping word-chunking TF-IDF retrieval system over store policies, paired with an autonomous ReAct refund agent equipped with state checkpointing, crash recovery, and stop-safely guardrails.

---

## Overview

In rapid-delivery quick commerce, order frequency and customer retention define unit economics. Instamart delivers groceries, dairy, snacks, personal care, and electronics across six metro cities within minutes. However, noisy transactional logging, unpredicted customer attrition, and unstructured customer support requests threaten operational scalability.

The **Customer Intelligence & Support Copilot** bridges the gap between raw data stores and real-time customer support:
1. **Analytical Ground Truth**: Eliminates duplicate orders, imputes unparseable amounts using catalogue pricing, standardises city and demographic metadata, and provides verified analytical queries for revenue tracking.
2. **Predictive Retention**: Computes customer-level Recency, Frequency, and Monetary (RFM) distributions, clusters customers into actionable business cohorts, and flags at-risk high-value customers using supervised classification before they lapse.
3. **Automated, Privacy-Preserving Support**: Redacts sensitive customer data before prompt generation, validates LLM categorization against JSON schemas and keyword patterns, and routes high-priority fraud or VIP customer tickets directly to specialized queues.
4. **Autonomous Policy Reasoning & Safe Execution**: Employs Retrieval-Augmented Generation (RAG) to ground answers in store policy documents and implements an autonomous ReAct agent with atomic tool execution, token/time limits, and checkpoint-resumption capabilities.

---

## Reproduce

Follow these numbered shell commands in order from a clean terminal to reproduce the entire environment, data generation, data cleaning, modeling, triage, and agent evaluations.

```bash
# 1. Clone the repository and enter the directory
git clone https://github.com/zakki7867/Instamart-Customer-Intelligence-Support-Copilot.git
cd Instamart-Customer-Intelligence-Support-Copilot

# 2. Create an isolated virtual environment (Python 3.11 or newer)
python -m venv venv

# 3. Activate the virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS / Linux:
# source venv/bin/activate

# 4. Install all required dependencies
pip install -r requirements.txt

# 5. Generate deterministic synthetic data and store policies
python generate_data.py

# 6. Part A: Audit raw data, apply cleaning rules, and store clean tables
python src/clean_data.py

# 7. Part A: Validate clean tables against reference checkpoints
python src/store_and_validate.py

# 8. Part A: Run analytical SQL queries and pandas cross-check / EDA
python src/sql_and_eda.py

# 9. Part B: Run RFM segmentation, K-Means clustering, and PCA projection
python src/customer_segmentation.py

# 10. Part B: Train, tune, and evaluate customer lapse prediction models
python src/lapse_prediction.py

# 11. Part C: Execute support ticket triage with PII redaction and routing
python src/support_triage.py

# 12. Part D: Evaluate policy chunking, TF-IDF retrieval, and RAG QA
python src/policy_rag.py

# 13. Part D: Run autonomous refund agent across 7 scenarios and safety demos
python src/refund_agent.py
```

---

## Part A — Data Foundation: Clean, Store, Query

### 1. Raw Data Audit

The raw order, customer, product, and ticket exports were systematically inspected prior to cleaning.

#### Audit Checkpoints Table

| Audit Metric | Raw Value | Reference Checkpoint | Status |
| :--- | :--- | :--- | :--- |
| Raw `orders.csv` rows | 13,084 | 13,084 | Matched |
| Exact duplicate rows in `orders.csv` | 119 | 119 | Matched |
| Pending status rows in `orders.csv` | 656 | 656 | Matched |
| Distinct `order_id` values | 12,309 | 12,309 | Matched |
| Non-numeric `order_amount` entries | 120 | 120 | Matched |
| Negative `delivery_minutes` entries | 106 | 106 | Matched |
| Blank `delivery_minutes` entries | 2,541 | 2,541 | Matched |
| Blank `age` entries in `customers.csv` | 61 | 61 | Matched |
| Impossible age entries outside 10–100 | 6 | 6 | Matched |
| Distinct raw city spellings | 24 | 24 | Matched |
| Distinct non-blank raw gender spellings | 10 | 10 | Matched |
| Blank `gender` entries | 30 | 30 | Matched |

#### Table Shapes, Data Types, and Missing Values

- **`orders.csv` (Raw Shape: 13,084 rows $\times$ 9 columns)**:
  - `order_id`: object, 0 missing (0.00%)
  - `customer_id`: object, 0 missing (0.00%)
  - `product_id`: object, 0 missing (0.00%)
  - `quantity`: int64, 0 missing (0.00%)
  - `order_ts`: object, 0 missing (0.00%)
  - `status`: object, 0 missing (0.00%)
  - `discount_pct`: int64, 0 missing (0.00%)
  - `order_amount`: object, 0 missing (120 unparseable strings 'N/A' or 'error')
  - `delivery_minutes`: object, 2,541 missing (19.42%)
- **`customers.csv` (Raw Shape: 1,200 rows $\times$ 5 columns)**:
  - `customer_id`: object, 0 missing (0.00%)
  - `city`: object, 0 missing (0.00%)
  - `signup_date`: object, 0 missing (0.00%)
  - `age`: float64/object, 61 missing (5.08%)
  - `gender`: object, 30 missing (2.50%)
- **`products.csv` (Raw Shape: 40 rows $\times$ 4 columns)**:
  - `product_id`: object, 0 missing (0.00%)
  - `product_name`: object, 0 missing (0.00%)
  - `category`: object, 0 missing (0.00%)
  - `price_inr`: int64, 0 missing (0.00%)

---

### 2. Documented Cleaning Rules

#### Customers Cleaning Rules
1. **City Standardisation**: Stripped leading/trailing whitespace and applied title-casing. Exactly 6 metro cities remain: Pune (227), Delhi (210), Bengaluru (197), Chennai (191), Hyderabad (191), Mumbai (184).
2. **Gender Normalisation**: Standardised variations (`M`, `male`, `Male`, `MALE` $\rightarrow$ `male`; `F`, `female`, `Female`, `FEMALE` $\rightarrow$ `female`). All 30 blank entries were mapped to `unknown`. Final distribution: female (610), male (560), unknown (30).
3. **Age Validation and Imputation**: Replaced 6 impossible values outside [10, 100] (values: 280, -4, 0, 150, 999, 200) with NaN. Filled all 67 missing ages using the median age of the customer's city via `groupby('city')['age'].transform()`. Post-cleaning: 0 missing ages, min age 18.0, max age 60.0.
4. **Signup Date Parsing**: Parsed `signup_date` strings to datetime timestamps (`YYYY-MM-DD`).

#### Orders Cleaning Rules
1. **Timestamp Conversion**: Converted `order_ts` strings to datetime objects across all 13,084 records.
2. **Exact Duplicate Removal**: Identified and dropped 119 exact duplicate records, leaving 12,965 rows.
3. **Order State Deduplication**: Grouped by `order_id` and kept only the record with the latest `order_ts`. This removed 656 earlier `pending` records, yielding exactly 12,309 unique orders.
4. **Order Amount Numeric Imputation**: Coerced non-numeric strings ('N/A', 'error') to NaN (120 values). Imputed missing amounts using catalogue price and discount formula:
   $$\text{order\_amount} = \text{round}\left(\text{price\_inr} \times \text{quantity} \times \left(1 - \frac{\text{discount\_pct}}{100}\right), 2\right)$$
   Missing `order_amount` count reduced to 0.
5. **Delivery Minutes Correction**: Set 106 negative values to NaN. For completed orders, filled missing values using product category median delivery times among completed orders. Left cancelled orders as missing (1,199 values) to prevent inventing delivery times for unfulfilled orders.

#### Post-Cleaning Summary Checkpoints
- Clean rows: 12,309 (one record per `order_id`)
- Status counts: 11,110 completed, 1,199 cancelled, 0 pending
- Missing values: 0 missing in `order_amount`; 0 missing in completed `delivery_minutes`; 1,199 missing in cancelled `delivery_minutes`
- Completed revenue sum: INR 2,998,668.80
- Validation Script: `python src/store_and_validate.py` prints `ALL CHECKPOINTS PASSED`

---

### 3. SQL Analysis Results (`sql/queries.sql`)

All queries were executed via SQLite on `data/instamart.db`:

| Query ID | Description | Result Metric | Reference Checkpoint | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Q1** | Completed orders > 500, customer total > 8000 | 30 customers | 30 customers | Matched |
| **Q2** | Three-table join (City $\times$ Category Revenue) | 30 rows; Total INR 2,998,668.80 | 30 rows; INR 2,998,668.80 | Matched |
| | Largest cell: Bengaluru / Electronics | INR 233,909.45 | INR 233,909.45 | Matched |
| **Q3** | Products priced above catalogue average | 11 products | 11 products | Matched |
| **Q4** | Customers with no completed orders | 25 customers | 25 customers | Matched |
| **Q5** | Customers with spend above average (CTE) | 550 customers (Avg: INR 2,552.06) | 550 customers; INR 2,552.06 | Matched |
| **Q6** | City spend rank (`RANK`, `DENSE_RANK`, `ROW_NUMBER`) | Exactly 8 rows have `RANK()` $\neq$ `ROW_NUMBER()` | 8 rows | Matched |
| **Q7** | Daily running total revenue | 183 daily rows; Final: INR 2,998,668.80 | 183 rows; INR 2,998,668.80 | Matched |

#### Top Spender per City (Q6)
- **Bengaluru**: Customer `C0082` — INR 12,856.20
- **Chennai**: Customer `C0930` — INR 10,151.40
- **Delhi**: Customer `C0291` — INR 13,742.60
- **Hyderabad**: Customer `C0743` — INR 12,730.05
- **Mumbai**: Customer `C0693` — INR 14,766.55
- **Pune**: Customer `C0426` — INR 10,377.10

> **Explanation of Window Function Ties (Q6)**:
> `RANK()` assigns identical ranks to tied values and skips subsequent positions, whereas `ROW_NUMBER()` assigns strictly unique, sequential integers regardless of ties; exactly 8 customers in the database share identical total spend values with another customer in their respective city, causing `RANK()` to differ from `ROW_NUMBER()`.

---

### 4. pandas Cross-Check and Exploratory Data Analysis

#### City-Level Aggregation (Completed Orders)
- **Bengaluru**: Sum = INR 523,643.40, Mean = INR 276.47, Count = 1,894 orders
- **Chennai**: Sum = INR 463,629.15, Mean = INR 272.72, Count = 1,700 orders
- **Delhi**: Sum = INR 502,829.80, Mean = INR 243.62, Count = 2,064 orders
- **Hyderabad**: Sum = INR 528,589.90, Mean = INR 284.95, Count = 1,855 orders
- **Mumbai**: Sum = INR 479,058.50, Mean = INR 297.00, Count = 1,613 orders
- **Pune**: Sum = INR 500,918.05, Mean = INR 252.48, Count = 1,984 orders
- **Grand Total (Pivot Table Margin)**: INR 2,998,668.80 (Exact match with SQL Q2).

#### Completed Order Amount Distribution
- **Mean**: INR 269.91
- **Median**: INR 135.00
- **Skewness**: +4.61 (heavily right-skewed due to high-value electronics purchases)
- **Quartiles**: Q1 = INR 64.80, Q3 = INR 297.00, IQR = INR 232.20
- **Tukey 1.5$\times$IQR Fences**: Lower Fence = INR -283.50 (0 outliers below), Upper Fence = INR 645.30 (897 outliers above)

#### Correlation Matrix (Completed Orders)
| Variable | order_amount | discount_pct | quantity | delivery_minutes |
| :--- | :--- | :--- | :--- | :--- |
| **order_amount** | 1.0000 | -0.0951 | +0.0873 | -0.0031 |
| **discount_pct** | -0.0951 | 1.0000 | -0.0004 | -0.0098 |
| **quantity** | +0.0873 | -0.0004 | 1.0000 | -0.0126 |
| **delivery_minutes** | -0.0031 | -0.0098 | -0.0126 | 1.0000 |

#### Synthesis of Exploratory Findings (Under 200 Words)
The dataset captures 11,110 completed quick-commerce transactions across six Indian metropolitan cities over a six-month window. The order amount distribution is strongly right-skewed (mean INR 269.91 vs. median INR 135.00, skewness 4.61). The upper 1.5$\times$IQR fence at INR 645.30 identifies 897 upper outliers driven primarily by high-ticket consumer electronics (earbuds, power banks, speakers priced up to INR 1,999) compared to daily essentials under INR 150. Across variables, linear correlations are negligible: delivery duration does not correlate with basket size or discounts ($|r| \le 0.01$). The largest correlation in absolute value is between `order_amount` and `discount_pct` ($r = -0.0951$), followed by `order_amount` and `quantity` ($r = +0.0873$). A correlation of this small magnitude (-0.095) indicates a very weak inverse association and does not establish causation; higher percentage discounts do not meaningfully dictate transaction value in this catalogue.

---

## Part B — Customer Segments and Lapse Prediction

### 1. RFM Table Construction
Snapshot date: `2025-10-01 00:00:00`. Evaluated on 1,175 customers with at least one completed order.
- **Recency (days)**: Median = 19.0, Min = 0, Max = 180
- **Frequency (order count)**: Mean = 9.46, Max = 33
- **Monetary (INR spend)**: Mean = INR 2,552.06, Max = INR 14,766.55

---

### 2. K-Means Clustering and Cluster Selection

All three standardized RFM metrics were clustered for $K \in [2, 8]$ with `random_state=42`:

| Clusters ($K$) | Inertia | Silhouette Score | Business Actionability |
| :--- | :--- | :--- | :--- |
| $K=2$ | 1820.8 | 0.465 | Under-segmented; mixes high-frequency and high-monetary buyers. |
| $K=3$ | 1268.2 | 0.445 | Coarse grouping; fails to isolate at-risk churning customers. |
| **$K=4$** | **743.1** | **0.526** | **Optimal**: Highest silhouette peak, distinct and actionable segments. |
| $K=5$ | 617.4 | 0.480 | Diminishing silhouette; fragments regular shoppers into redundant tiers. |
| $K=6$ | 503.2 | 0.482 | Over-segmented for marketing campaign operations. |
| $K=7$ | 408.5 | 0.416 | Degraded silhouette cohesion. |
| $K=8$ | 366.5 | 0.412 | Excessive fragmentation. |

> **Inertia vs. $K$ Selection**:
> Inertia is monotonically non-increasing with $K$ because adding cluster centroids mathematically minimizes sum-of-squared distances to the nearest centre (reaching zero when $K=N$). Thus, lower inertia alone cannot justify model selection; $K=4$ was selected because it achieves the global silhouette maximum (0.526), exhibits a clear elbow inflection, and produces distinct, actionable customer personas.

---

### 3. Customer Segment Profiling and PCA Projection ($K=4$)

*Note: Cluster numbers are arbitrary numeric identifiers without intrinsic ranking.*

| Segment ID | Name | Size | Mean Recency (days) | Mean Frequency | Mean Monetary (INR) | Recommended Action | `is_premium` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cluster 1** | **Premium Spenders** | 138 | 24.1 | 6.77 | INR 7,129.70 | VIP support routing, early tech drops, concierge desk | **True** |
| **Cluster 2** | **Loyal Frequent** | 419 | 8.8 | 19.40 | INR 3,410.02 | Subscription loyalty perks (Instamart Black), grocery replenishment | False |
| **Cluster 0** | **Occasional Regulars**| 436 | 28.9 | 3.77 | INR 1,057.62 | Mid-week re-engagement nudges, basket-building coupons | False |
| **Cluster 3** | **At-Risk / Inactive** | 182 | 109.0 | 2.20 | INR 685.99 | Win-back push notifications, steep reactivation discounts | False |

- **PCA Projection**: Two components explain **63.4%** (PC1) and **21.8%** (PC2) of the variance, capturing **85.2%** of total RFM variance.
- Output written to `outputs/customer_segments.csv`.

---

### 4. Lapse Problem Definition & Feature Engineering

- **Cutoff Date**: `2025-09-01 00:00:00`.
- **Cohort**: 1,156 customers with $\ge 1$ completed order prior to cutoff.
- **Target Label (`lapsed`)**: 1 if the customer placed 0 completed orders between `2025-09-01` and `2025-09-30`, else 0.
- **Class Distribution**: 443 lapsed (38.32%), 713 retained (61.68%).
- **Split**: 80/20 stratified split (`random_state=42`) $\rightarrow$ 924 train, 232 test.
- **Baseline Model (Always Predict Retained)**: Accuracy = **61.6%**, Lapsed F1 = **0.0000**.
- **Data Leakage Safeguards**:
  - Only transactions strictly preceding `2025-09-01` were used for feature construction.
  - Task 3 whole-window RFM segment labels were excluded from the feature set.
  - All scalers and encoders were fitted strictly on the 924 training rows.
  - **Verification**: Scaler training mean of `frequency` is **8.1147**, compared to the full-cohort mean of **8.0285**. They differ because the scaler learned only from training rows.
  - Missing values in final feature matrix: **0**.

---

### 5. Model Tuning & Evaluation

Models were tuned using 5-fold cross-validation on training data optimizing for positive-class ($y=1$, lapsed) F1-score:

| Model | Hyperparameters / Grid | Best Params | Best CV F1 | Test Precision | Test Recall | Test F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Majority class prediction | — | — | 0.0000 | 0.0000 | 0.0000 |
| **Logistic Regression** | $C \in \{0.01, 0.1, 1, 10, 100\}$ | $C=1$ | 0.6480 | 0.6404 | 0.6404 | 0.6404 |
| **Random Forest** | `max_depth` $\in \{3, 5, 8\}$, `min_samples_leaf` $\in \{5, 10, 20\}$ | depth=8, leaf=10 | **0.6800** | **0.6629** | **0.6629** | **0.6629** |

#### Test Confusion Matrices
- **Logistic Regression**:
  $$\begin{bmatrix} 111 & 32 \\ 32 & 57 \end{bmatrix}$$
- **Random Forest (Selected Model)**:
  $$\begin{bmatrix} 113 & 30 \\ 30 & 59 \end{bmatrix}$$

> **Model Selection Justification**:
> Random Forest was selected over Logistic Regression because it achieved higher test precision (0.6629 vs. 0.6404) and test F1 (0.6629 vs. 0.6404), exceeding the $\ge 0.55$ requirement. In retail retention, the cost of missing a lapsing customer (false negative: losing future customer lifetime value of several thousand rupees) vastly outweighs the marginal marketing cost of offering a retention discount to a loyal customer (false positive). Random Forest minimized false negatives (30 vs 32) while accurately capturing non-linear feature interactions between delivery delays and order frequency.

---

### 6. Class-Balance Experiment

To address minority class under-representation (354 lapsed vs. 570 retained in train), training rows were oversampled with replacement (`random_state=42`) to 570 per class:

| Model Training | Test Precision | Test Recall | Test F1 |
| :--- | :--- | :--- | :--- |
| **Unbalanced Logistic Regression** | 0.6404 | 0.6404 | 0.6404 |
| **Balanced Logistic Regression** | 0.5966 | **0.7978** | **0.6827** |

> **Retention Team Takeaway**:
> Oversampling shifted the decision boundary toward the minority class, driving recall up by **+15.7 percentage points** (from 64.04% to 79.78%) while precision remained stable near 59.7%. For the retention team, this model configuration captures 8 out of 10 churning customers, making it ideal when aggressive retention campaigns are prioritized.

---

### 7. Feature Interpretability & Retention List

#### Top 5 Features (Random Forest)
1. `frequency` (0.2869): Historical order velocity is the strongest predictor of habituation and future ordering.
2. `monetary` (0.2685): Total cumulative spend reflects wallet-share and account stickiness.
3. `recency` (0.1145): Days elapsed since the last order indicates natural purchasing cadence.
4. `avg_delivery_minutes` (0.0605): Recurring delivery friction subtly drives customers to competitor platforms.
5. `cancellation_rate` (0.0506): Frequent order cancellations correlate strongly with customer dissatisfaction.

#### Logistic Regression Coefficients
- `recency` ($+0.0495$, positive): Customers who have not ordered recently have an increased log-odds of lapsing.
- `frequency` ($-1.6126$, negative): Higher order count strongly decreases the likelihood of customer churn.
- `monetary` ($+0.2326$, positive): Holding frequency fixed, higher basket value slightly elevates churn risk due to lower purchase repetition.

#### Retention List (`outputs/retention_list.csv`)
Persisted selected model to `outputs/lapse_model.joblib`. Verified that reloading reproduces exact test predictions. Extracted test-set customers in the Premium segment ranked by lapse probability:

| Customer ID | Predicted Lapse Probability | Actually Lapsed |
| :--- | :--- | :--- |
| `C0862` | 0.5074 | 0 |
| `C1112` | 0.4955 | 0 |
| `C0167` | 0.4903 | 0 |
| `C1000` | 0.4849 | 0 |
| `C0060` | 0.4780 | 0 |
| `C0267` | 0.4730 | 0 |
| `C0021` | 0.4686 | 0 |
| `C0133` | 0.4676 | 0 |
| `C0930` | 0.4613 | 0 |
| `C1008` | 0.4598 | 0 |

> **Important Disclaimer on Post-Hoc Premium Label**:
> The `is_premium` flag originates from the whole-window RFM analysis (Task 3), which includes September transactions; therefore, the retention list is illustrative of VIP targeting workflows and is not a strictly leakage-free evaluation. However, the lapse model's feature matrix strictly utilized orders prior to the `2025-09-01` cutoff.

---

## Part C — Support Ticket Triage with Guardrails

### 1. Guardrails & Privacy-Preserving Redaction Audit

Evaluated over all 164 tickets in `data/raw/tickets.csv`:
- **Input Guardrail (a)**: 4 tickets were empty or whitespace-only $\rightarrow$ assigned status `skipped_empty` (0 model calls).
- **Input Guardrail (b)**: 4 tickets exceeded 400 tokens ($\sim 734$ tokens each) $\rightarrow$ assigned status `flagged_long` (0 model calls).
- **PII Substitutions**:
  - Email addresses (`[\w.+-]+@[\w-]+\.[\w.]+` $\rightarrow$ `[EMAIL]`): **55**
  - Phone numbers (`\b[6-9]\d{9}\b` $\rightarrow$ `[PHONE]`): **91**
  - Order IDs (`ORD-\d{6}` $\rightarrow$ `[ORDER_ID]`): **132**
  - **Total PII Redactions**: **278** (0 remaining post-redaction assertion verified).
  - Prompts delivered to the model contained strictly the sanitized ticket text.

---

### 2. Resilient API Calls & Output Validation

- **Transient Error Handling**: `mock_llm.py` simulates network timeouts every 9th call. The retry wrapper implemented exponential backoff with jitter ($0.05 \times 2^n \times [0.9, 1.15]$ seconds). Across 156 eligible tickets, **19 TransientLLMErrors** were encountered and 100% recovered on retry 0 without any permanent failures.
- **Hierarchical Output Validation**:
  - Valid JSON with allowed category $\rightarrow$ `classified_json`: **135 tickets**
  - Malformed/chatty text parsed via single keyword extraction $\rightarrow$ `classified_keyword`: **12 tickets**
  - Out-of-schema category (e.g. `shipping`) $\rightarrow$ `flagged_invalid_output`: **9 tickets**
  - **Total Classified**: 147 tickets. Total Flagged for Human Review: 17 tickets (9 invalid + 4 empty + 4 long).

---

### 3. Zero-Shot vs. Few-Shot Performance Comparison

| Pipeline Run | Classified Tickets | Accuracy | Total Tokens | Tokens / Classified Ticket | Status Counts (json / kw / invalid / empty / long) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Zero-Shot** | 147 | **85.7%** (126/147) | 11,428 | 77.7 | 135 / 12 / 9 / 4 / 4 |
| **Few-Shot** | 147 | **100.0%** (147/147) | 19,228 | 130.8 | 135 / 12 / 9 / 4 / 4 |

#### Zero-Shot Confusion Matrix (Classified Tickets)
| True Category | Predicted billing | Predicted delivery | Predicted fraud | Predicted technical |
| :--- | :--- | :--- | :--- | :--- |
| **billing** | **38** | 0 | 0 | 0 |
| **delivery** | 0 | **36** | 0 | 0 |
| **fraud** | 11 | 0 | **23** | 0 |
| **technical** | 10 | 0 | 0 | **29** |

> **Cost vs. Accuracy Analysis (When Few-Shot is Justified)**:
> In the Zero-Shot configuration, 11 fraud tickets and 10 technical tickets were misclassified as billing because users mentioned financial keywords like 'refund' or 'deducted' while describing unauthorized account takeovers or app crashes. Few-Shot in-context learning cost 53.1 additional tokens per ticket (+68% token overhead). In mission-critical environments, misrouting security breaches or account takeovers to standard billing queues creates severe financial and reputational liability. The extra token expenditure is thoroughly justified because it achieves 100% routing accuracy and eliminates security escalations being overlooked.

---

### 4. Token Budgeting & Production Routing

- **Token Budget Demonstration (`outputs/triage_budget_demo.csv`)**:
  - Configured zero-shot run with `TOKEN_BUDGET = 6000`.
  - The pipeline processed initial tickets until cumulative tokens hit the ceiling, cleanly stopping model calls and routing remaining tickets to `human_review` with status `budget_exhausted`. Exactly **74 tickets** ended with status `budget_exhausted`.
- **Final Segment-Based Routing (`outputs/triage_results.csv`)**:
  - Generated from Few-Shot pipeline with `TOKEN_BUDGET = 100000` (164 rows total).
  - Routing distribution:
    - `support_queue` (Priority NORMAL): **102 tickets** (standard inquiries from non-premium users)
    - `security_team` (Priority HIGH): **34 tickets** (all validated fraud cases)
    - `human_review` (Priority NORMAL): **17 tickets** (4 empty, 4 long, 9 schema-invalid outputs)
    - `premium_desk` (Priority HIGH): **11 tickets** (non-fraud requests from validated VIP customers)

---

## Part D — Policy RAG Assistant and Refund Agent

### 1. Document Chunking & TF-IDF Retrieval

- Five policy documents in `data/policies/` were partitioned into 40-word chunks with a 10-word sliding overlap:
  - `delivery_policy.txt`: 3 chunks
  - `escalation_policy.txt`: 3 chunks
  - `payment_policy.txt`: 3 chunks
  - `privacy_policy.txt`: 2 chunks
  - `refund_policy.txt`: 4 chunks
  - **Total Chunks**: **15 chunks** (Exact match with reference).
- Vectorized using `TfidfVectorizer()` and cosine similarity ranking.
- Query test: `refund policy late delivery` $\rightarrow$ ranks 1, 2, and 3 all originate from `refund_policy.txt`.

> **TF, IDF, and Inherent Limitations**:
> Term Frequency (TF) measures how often a word occurs within a specific chunk, while Inverse Document Frequency (IDF) penalizes ubiquitous words across all chunks to emphasize discriminative terms. A primary limitation of lexical TF-IDF retrieval is the vocabulary mismatch problem: queries using synonyms (e.g. 'late arrival reimbursement') fail to match relevant chunks phrased with different vocabulary (e.g. 'delay refund') because exact token overlap is absent.

---

### 2. RAG Assistant Grounding & Overlap Ablation

Evaluated over the 14 questions in `data/raw/policy_eval.csv`:

| Evaluation Mode | Expected Doc in Top 3 | Expected Fact in Output | Unanswerable Fallback ('I could not find...') |
| :--- | :--- | :--- | :--- |
| **Standard (10-Word Overlap)** | **12/12** (100%) | **12/12** (100%) | **2/2** (100%) |
| **Ablation (0-Word Overlap)** | 12/12 (100%) | **10/12** (83.3%) | 2/2 (100%) |

#### Failure Mode Under Zero-Overlap
Under zero overlap, two questions failed:
1. *'What delivery time makes a completed order eligible for a full refund?'*: Expected fact `'more than 45 minutes'` was severed across a chunk boundary, causing the model to retrieve only the trailing clause `'receives a full refund of the order amount.'`
2. *'How soon is money returned when a payment fails but is deducted?'*: Expected fact `'48 hours'` was split from the opening condition, returning fragmented text.
This empirically demonstrates why boundary-overlapping chunking is vital for contextual integrity.

---

### 3. ReAct Refund Agent Evaluation

The refund agent operates with five registered tools: `get_order`, `search_policy`, `calculate_refund`, `issue_refund`, `escalate_to_human`. Evaluated across all 7 scenarios in `data/raw/agent_scenarios.csv`:

#### Seven Scenarios Results Table

| Scenario ID | Expected Outcome | Actual Status | Expected Refund (INR) | Actual Refund (INR) | Model Calls | Tokens Used |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **S1** | REFUND | REFUND | 128.25 | 128.25 | 5 | 897 |
| **S2** | REFUND | REFUND | 135.00 | 135.00 | 5 | 897 |
| **S3** | ESCALATE | ESCALATE | 1999.00 | 1999.00 | 5 | 884 |
| **S4** | DENIED | DENIED | 0.00 | 0.00 | 4 | 605 |
| **S5** | DENIED | DENIED | 0.00 | 0.00 | 3 | 337 |
| **S6** | NOT_FOUND | NOT_FOUND | 0.00 | 0.00 | 2 | 72 |
| **S7** | FALLBACK | FALLBACK_MAX_ITERATIONS | 0.00 | 0.00 | 6 | 3,000 |

- All refund amounts match expected values within INR 0.01.
- All executed refunds and escalations are atomically logged to `outputs/refund_ledger.jsonl`.
- Full execution trajectories with step observations are recorded in `outputs/agent_runs.jsonl`.

---

### 4. Stop-Safely Demonstrations

1. **Token Limit Demonstration (S1 with `token_limit=200`)**:
   - Status: **`FALLBACK_TOKEN_LIMIT`**
   - The agent aborted immediately upon exceeding the 200 token budget before executing the pending tool, safely returning a handoff summary.
2. **Timeout Demonstration (S1 with `time_limit_seconds=0`)**:
   - Status: **`FALLBACK_TIMEOUT`**
   - The agent verified elapsed monotonic time at the top of iteration 1 and stopped before initiating any model calls.

---

### 5. Crash-and-Resume Demonstration (S1)

Simulated a critical process termination immediately following the second tool execution (`search_policy`):
- **Tools executed prior to crash**: `['get_order', 'search_policy']`
- Intermediate state retrieved from `state/s1_crash_resume_test.json`.
- `resume_agent('s1_crash_resume_test')` loaded execution history and completed remaining tools: `['calculate_refund', 'issue_refund']`.
- **Ordered Sequence Across Both Parts**:
  `get_order` $\rightarrow$ `search_policy` $\rightarrow$ `calculate_refund` $\rightarrow$ `issue_refund`
- Exactly four unique tools executed without redundant repetition. Final status: **`REFUND` (INR 128.25)**.

---

### 6. Architectural Trade-Off: ReAct vs. Plan-and-Execute (Under 250 Words)

In this refund workflow, the ReAct paradigm required 2 to 5 iterative LLM calls for straightforward cases (2 for S6, 3 for S5, 4 for S4, 5 for S1–S3) and consumed between 72 and 897 tokens. Under a Plan-and-Execute architecture, the agent would generate a complete upfront plan (e.g. `[get_order, search_policy, calculate_refund, issue_refund]`) in a single initial call, consuming fewer planning tokens when workflows proceed nominally.

However, Plan-and-Execute breaks down when encountering unexpected intermediate observations. In Scenario S6 (`NOT_FOUND`), an upfront planner would have erroneously queued policy lookups and refund calculations for a nonexistent order, forcing an expensive mid-plan replanning call. Similarly, in Scenario S5 (cancelled order), ReAct immediately terminated processing upon reading `status: cancelled` during policy search, whereas Plan-and-Execute would require replanning logic to abort downstream refund issuance. Furthermore, in infinite loop prompts like Scenario S7, ReAct safely hit its maximum reasoning boundary.

For financial transaction workflows like automated refunds, **ReAct is the superior choice**. Its interleaved reasoning and action loop enforces strict validation at each step, guarantees that orders exist and qualify before money is moved, and simplifies safety stops. The minor increase in model calls is an acceptable trade-off for transactional correctness and zero unintended fund disbursements.

---

## Limitations and Next Steps

### Current Limitations
1. **Synthetic Data Artifacts**: The underlying dataset relies on Poisson-generated transactional frequencies and fixed Gaussian delivery times. While realistic, real quick-commerce logs contain complex real-world anomalies such as split-shipments, rider re-assignments, partial substitutions, and dynamic surge pricing.
2. **Static Snapshot RFM**: Recency, Frequency, and Monetary scores were calculated using a static snapshot date (`2025-10-01`). In live production, customer engagement shifts dynamically, necessitating rolling window RFM metrics.
3. **Illustrative VIP Retention List**: The retention list generated in Part B uses the post-hoc whole-window `is_premium` flag to simulate VIP queue targeting. While the model itself used only pre-cutoff features, a fully leakage-free production deployment requires segmenting customers strictly on historical windows.
4. **Lexical Retrieval Constraints**: The policy assistant uses TF-IDF, which cannot capture semantic relationships across synonyms or multilingual colloquialisms common in Indian metro customer support (e.g., Hinglish or regional slang).

### Next Steps for Production Deployment
1. **Migration to Hosted LLMs**: Replace `mock_llm.py` with enterprise LLM endpoints (e.g., Gemini 1.5 Pro / Flash via Google Cloud Vertex AI) using native OpenAI/Gemini tool calling schemas.
2. **Dense Semantic Embeddings**: Upgrade policy retrieval from TF-IDF to dense vector embeddings (e.g., `text-embedding-004`) stored in a managed vector database (e.g., Chroma, Pgvector, or BigQuery Vector Search) for semantic and multilingual similarity.
3. **Automated Human-in-the-Loop Queue**: Integrate the flagged tickets (`human_review`, `security_team`, `premium_desk`) with customer support platforms such as Zendesk, Freshdesk, or Jira Service Management via webhooks.
4. **Real-Time Feature Store**: Ingest live order events via Apache Kafka or Google Cloud Pub/Sub into a feature store (e.g., Feast), calculating continuous churn risk scores and dynamically triggering proactive retention coupons.
