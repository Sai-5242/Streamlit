import joblib

model = joblib.load('D:\IRIS_STREAMLIT\model\model.pkl')
print("model loaded sucessfully")
print(model.predict([[5.1, 3.5, 1.4, 0.2]]))