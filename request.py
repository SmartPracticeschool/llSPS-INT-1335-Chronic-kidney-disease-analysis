import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'bp':80, 'sg':1.02, 'al':1,'su':0})

print(r.json())
