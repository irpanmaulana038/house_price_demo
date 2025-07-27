# 🏡 House Price Prediction API with FastAPI

This project serves a trained regression model as an API using FastAPI. It predicts house prices based on engineered features from housing data.

## 🚀 Features
- CSV upload endpoint to predict multiple house prices
- Feature engineering with `create_features()` function
- Scikit-learn pipeline integration

## 📁 Folder Structure
- `app/`: Contains the FastAPI app and feature engineering code
- `models/`: Contains the serialized model (`.pkl`)
- `test.csv`: Example input
- `requirements.txt`: Python dependencies

## ▶️ How to Run Locally

```bash
# 1. Create virtual env
conda create -n house_api python=3.11 -y
conda activate house_api

# 2. Clone this repo
git clone https://github.com/yourusername/house_price_demo.git
cd house_price_demo

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the server
uvicorn app.main:app --reload

