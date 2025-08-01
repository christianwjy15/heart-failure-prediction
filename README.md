# heart-failure-prediction
Predict heart failure

python -m src.data.data_preprocessing
python -m src.models.train_model
python -m src.models.evaluate_model
mlflow ui

docker-compose up --build
buka localhost

atau

uvicorn src.app.api:app --reload

ubah code streamlit
streamlit run src/app/streamlit_app.py
buka localhost