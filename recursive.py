import requests
import time

URL = "https://placement-ranking-api.onrender.com/top-students?n=1"  # Your live URL

while True:
    try:
        response = requests.get(URL)
        print(f"Pinged API - Status: {response.status_code} | Time: {time.ctime()}")
    except Exception as e:
        print(f"Error: {e}")
    
    time.sleep(10)  # Wait 5 minutes (300 seconds)