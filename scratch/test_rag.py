import requests
import time

URL = "http://127.0.0.1:8000/ask"
payload = {
    "username": "gauri",
    "question": "What is the primary objective of this paper?",
    "filename": "A_Comprehensive_Review_of_Artificial_Intelligence_.pdf",
    "session_id": "test_session_123",
    "mode": "student"
}

print(f"Sending request to {URL}...")
start = time.time()
try:
    r = requests.post(URL, json=payload, timeout=300)
    end = time.time()
    print(f"Status: {r.status_code}")
    print(f"Time taken: {end - start:.2f}s")
    if r.status_code == 200:
        data = r.json()
        print(f"Answer: {data.get('answer')[:200]}...")
        print(f"Pipeline Keys: {list(data.keys())}")
    else:
        print(f"Error: {r.text}")
except Exception as e:
    print(f"Request failed: {e}")
