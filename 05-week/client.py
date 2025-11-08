import requests

client_2 = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0,
}

url = "http://localhost:9090/predict"

response = requests.post(url, json=client_2).json()
print("Converted probability:", f"{response['converted_probability']:.3f}")
