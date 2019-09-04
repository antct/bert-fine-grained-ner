import requests

url = 'http://127.0.0.1:5000/'

data = {
    'sent': 'I love China, you like USA.'
}

res = requests.get(url, params=data).json()
print(res)
