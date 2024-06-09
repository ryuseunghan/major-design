from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from enum import Enum
import joblib
import pandas as pd

model = joblib.load('rf_clf.pkl')

app = FastAPI()

class rf_clf(BaseModel):
    연령 : int
    허리둘레 : int
    흡연상태 : int
    음주여부 : int
    BMI : float
    남성 : int
    여성 : int


@app.post("/predict")
def predict(application: rf_clf):
    # 입력 데이터를 DataFrame으로 변환
    input_df = pd.DataFrame([application.dict()])

    # 예측 수행
    prediction = model.predict(input_df)

    prediction_as_int = int(prediction[0])

    # 예측 결과 반환
    return {"predicted": prediction_as_int}
