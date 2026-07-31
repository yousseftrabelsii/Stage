import requests

files = {'file': ('dummy.pdf', b'dummy content', 'application/pdf')}
response = requests.post('http://localhost:5000/api/upload', files=files)
print(response.status_code)
print(response.json())
