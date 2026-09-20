import requests

url = "http://localhost:8000/api/agent/chat"
payload = {
    "thread_id": "test_identity_123",
    "message": None
}
# Start session
response = requests.post(url, json=payload)

# Turn 1
payload["message"] = "Manuel"
response = requests.post(url, json=payload)
print("Turn 1:", response.json())

# Turn 2
payload["message"] = "Are you a human or a bot?"
response = requests.post(url, json=payload)
print("Turn 2:", response.json())
