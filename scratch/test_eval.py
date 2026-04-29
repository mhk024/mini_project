import requests
import time

URL = "http://127.0.0.1:8000/evaluate"
payload = {
    "filename": "A_Comprehensive_Review_of_Artificial_Intelligence_.pdf",
    "questions": [
        "What is the primary objective of this paper?",
        "What methodology is used?"
    ]
}

print(f"Sending request to {URL}...")
start = time.time()
try:
    r = requests.post(URL, json=payload, timeout=600)
    end = time.time()
    print(f"Status: {r.status_code}")
    print(f"Time taken: {end - start:.2f}s")
    if r.status_code == 200:
        data = r.json()
        print(f"Metrics: {data.get('metrics')}")
        for res in data.get('results', []):
            print(f"\nQ: {res['query']}")
            print(f"Ref: {res['reference_answer'][:100]}...")
            print(f"Mod: {res['model_answer'][:100]}...")
            print(f"Sim: {res['similarity']}")
    else:
        print(f"Error: {r.text}")
except Exception as e:
    print(f"Request failed: {e}")
