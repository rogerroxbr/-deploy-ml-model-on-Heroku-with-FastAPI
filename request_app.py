"""
Module that make a request to the inferencer
Author : Roger de Tarso
Date : 20th may 2023
"""
import requests

row = {
    "age": 47,
    "workclass": "Private",
    "fnlwgt": 51835,
    "education": "Prof-school",
    "education_num": 15,
    "marital_status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Wife",
    "race": "White",
    "sex": "Female",
    "capital_gain": 0,
    "capital_loss": 1902,
    "hours_per_week": 60,
    "native_country": "Honduras",
}

# response = requests.post(url="http://127.0.0.1:8000/inference", json=row)
response = requests.post(
    url="https://udacity-project.herokuapp.com/inference", json=row
)

print(response.status_code)
print(response.json())
