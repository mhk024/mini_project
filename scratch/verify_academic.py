import requests

BASE = "http://127.0.0.1:8000"
ACAD_FULL = f"{BASE}/academic/full-analysis"

print(f"Checking {ACAD_FULL}...")
try:
    # Just a HEAD or GET might return 405 Method Not Allowed, which is better than 404
    r = requests.post(ACAD_FULL, json={"filename": "none.pdf"})
    print(f"Status: {r.status_code}")
    print(f"Response: {r.json()}")
except Exception as e:
    print(f"Error: {e}")
