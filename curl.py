import requests

url = 'http://127.0.0.1:3104/batch'

data = {
    'sent': ['I love China, you like USA.'] * 2
}

res = requests.post(url, data=data).json()
print(res)
