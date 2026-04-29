import requests

BASE = "http://127.0.0.1:8000"
ACAD_EVAL = f"{BASE}/academic/evaluate-questions"

DATA = {
    "questions": ["What is AI?"],
    "filename": "A_Comprehensive_Review_of_Artificial_Intelligence_.pdf"
}

print(f"Checking {ACAD_EVAL}...")
try:
    r = requests.post(ACAD_EVAL, json=DATA)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        print("Success!")
    else:
        print(f"Error: {r.text}")
except Exception as e:
    print(f"Error: {e}")
