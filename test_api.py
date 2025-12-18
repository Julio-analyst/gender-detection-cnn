import requests

url = "http://localhost:8000/predict"
files = {"file": open("path_audio.wav", "rb")}
response = requests.post(url, files=files)
print(response.status_code)
print(response.json())
