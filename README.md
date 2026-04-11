# Credit Card Fraud Detection API

## Project Overview
This project aims to detect fraudulent transactions using a machine learning model trained on historical credit card transaction data. The API provides an interface to interact with the model and perform predictions on new transactions.

## Why Decisions Were Made
- **Choice of Algorithm**: After evaluating several algorithms, a decision was made to use XGBoost due to its high performance and efficiency in handling large datasets.
- **Data Sources**: The dataset was chosen based on its comprehensiveness and representative nature of real-world transactions.
- **API Design**: The RESTful architecture was chosen for its simplicity and ease of integration with various frontend technologies.

## How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/rohitnath-dev/credit-card-fraud-detection-api.git
   cd credit-card-fraud-detection-api
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app.py
   ```
4. Access the API at `http://localhost:5000`.

## API Examples
- **Predict Fraud**: To predict whether a transaction is fraudulent, send a POST request to `/predict` with the following JSON body:
   ```json
   {
       "amount": 100,
       "location": "New York",
       "transactionTime": "2026-04-11T06:27:31Z"
   }
   ```
   You will receive a response indicating whether the transaction is fraudulent or not.

## Project Limitations
- The model's accuracy depends heavily on the quality and size of the training data.
- The API may not perform well on unseen types of transactions that diverge significantly from the training data.
- Real-time detection may be limited by the processing time of incoming requests.

---
Last updated: 2026-04-11 06:27:31 UTC