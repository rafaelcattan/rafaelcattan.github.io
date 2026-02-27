# Data Scientist Coding Exam

**Time:** 90 minutes  
**Instructions:** Provide your answers in a Python script or Jupyter notebook. Include comments explaining your reasoning.

---

## Part 1: Data Manipulation & Analysis

### Question 1.1: Data Cleaning
You are given a CSV file `sales_data.csv` with the following columns: `date`, `product_id`, `category`, `price`, `quantity`, `customer_id`. The data has missing values, duplicates, and inconsistent categories.

Tasks:
1. Load the CSV into a pandas DataFrame.
2. Handle missing values:
   - For `price`, fill with the median price of the same category.
   - For `quantity`, drop rows where quantity is missing.
3. Remove duplicate rows (keeping the first occurrence).
4. Standardize the `category` column: convert to lowercase, strip whitespace, and replace any variant spellings (e.g., "Electronics" vs "electronics") with a canonical name.
5. Create a new column `revenue` = `price` * `quantity`.

Write Python code to accomplish the above.

### Question 1.2: Exploratory Data Analysis
Using the cleaned DataFrame from 1.1:
1. Compute summary statistics (mean, median, standard deviation) for `price`, `quantity`, and `revenue` per category.
2. Plot a bar chart showing total revenue per category.
3. Identify the top 5 customers by total revenue and display their IDs and total spend.

---

## Part 2: Machine Learning

### Question 2.1: Feature Engineering & Model Training
You have a dataset `customer_churn.csv` with features describing customer behavior and a binary target `churn` (1 if customer churned, 0 otherwise). The features include `tenure`, `monthly_charges`, `total_charges`, `contract_type`, `payment_method`, etc.

Tasks:
1. Perform train‑test split (80/20) with stratification on the target.
2. Encode categorical variables appropriately (one‑hot encoding for low‑cardinality, target encoding for high‑cardinality).
3. Scale numerical features using StandardScaler.
4. Train a logistic regression model and a random forest classifier.
5. Evaluate both models using accuracy, precision, recall, F1‑score, and ROC‑AUC.
6. Which model performs better? Justify your answer.

### Question 2.2: Hyperparameter Tuning
Using the better model from 2.1 (or random forest if tie):
1. Perform grid search over a reasonable hyperparameter space (e.g., `n_estimators`, `max_depth`, `min_samples_split` for random forest).
2. Use 5‑fold cross‑validation and optimize for ROC‑AUC.
3. Report the best parameters and the cross‑validated performance.

---

## Part 3: Time Series Forecasting

### Question 3.1: ARIMA Modeling
You are given a monthly time series of `unemployment_rate.csv` (two columns: `date`, `rate`).

Tasks:
1. Plot the series and inspect for trend, seasonality, and stationarity.
2. If non‑stationary, apply differencing and/or transformation.
3. Determine appropriate ARIMA orders (p,d,q) using ACF/PACF plots or auto‑ARIMA.
4. Fit an ARIMA model and forecast the next 12 months.
5. Plot the forecast with 95% confidence intervals.

---

## Part 4: SQL

### Question 4.1: Query Writing
Given the following tables:

- `orders(order_id, customer_id, order_date, total_amount)`
- `customers(customer_id, signup_date, country)`
- `order_items(order_id, product_id, quantity, price)`

Write SQL queries for:
1. Total revenue per month for the last 12 months.
2. Customers who have made at least 3 orders, along with their average order value.
3. The top‑selling product (by quantity) in each country.

---

## Part 5: Problem Solving & Communication

### Question 5.1: Interpretation
You trained a model that predicts customer lifetime value (CLV). The feature importance from a gradient boosting model shows that `number_of_complaints` is the most important predictor, with a negative relationship.

How would you explain this finding to a non‑technical business stakeholder? Provide a concise paragraph.

### Question 5.2: Ethics
Your model uses zip‑code‑based features that indirectly correlate with race, leading to potential disparate impact. What steps would you take to detect and mitigate bias?

---

## Submission

Please bundle your code, visualizations, and answers in a single notebook or script. Include a brief summary of your approach and any assumptions made.

Good luck!