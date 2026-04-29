import requests
import time

URL = "http://127.0.0.1:8000/ask"
DATA = {
    "username": "test_user",
    "question": "what is research in this paper about",
    "filename": "A_Comprehensive_Review_of_Artificial_Intelligence_.pdf",
    "mode": "student"
}

print(f"Sending request to {URL}...")
start = time.time()
try:
    r = requests.post(URL, json=DATA, timeout=300)
    end = time.time()
    print(f"Status Code: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"Answer snippet: {data.get('answer', '')[:200]}...")
        print(f"Execution time (from backend): {data.get('execution_time_ms', 0)}ms")
        print(f"Total time (from script): {end - start:.2f}s")
        print(f"Answer Word Count: {len(data.get('answer', '').split())}")
    else:
        print(f"Error: {r.text}")
except Exception as e:
    print(f"Request failed: {e}")
