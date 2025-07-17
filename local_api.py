import requests

data = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}
try:
    response = requests.get("http://127.0.0.1:8000")
    print("Status Code:", response.status_code)
    print("Raw Response:", response.text)
    response = requests.post("http://127.0.0.1:8000/data/", json=data)
    print("Status Code:", response.status_code)
    print("Headers:", response.headers)
    print("Raw Response:", response.text)  # This will show what's actually being returned
    if response.status_code == 200:
        print("JSON Response:", response.json())
except Exception as e:
    print("Error:", str(e))