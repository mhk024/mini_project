import os, time, requests, json

BASE_URL = 'http://127.0.0.1:8000'

# upload sample file
files = {'file': ('sample.txt', b'This is a test document for RAG.')}
resp = requests.post(f'{BASE_URL}/upload', files=files)
print('Upload status:', resp.status_code, resp.text)

# wait a moment for processing
time.sleep(2)

payload = {
    'username': 'test_user',
    'question': 'What is the content?',
    'filename': 'sample.txt',
    'mode': 'student'
}
headers = {'Content-Type': 'application/json'}
resp = requests.post(f'{BASE_URL}/ask', json=payload, headers=headers)
print('Ask status:', resp.status_code)
print('Response:', resp.text)
