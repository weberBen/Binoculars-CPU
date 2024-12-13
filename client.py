import requests

def get_bearer_token(url, api_key):
    headers = {
        "X-API-Key": api_key,
    }
    response = requests.post(url, headers=headers)

    if response.status_code == 200:
        print("Response:", response.json())
        return response.json()["access_token"]
    else:
        print("Error:", response.status_code, response.text)
        return None


def send_post_request(url, token, text=None, file_path=None, threshold=None):
    headers = {
       "Authorization": f"Bearer {token}"
    }

    data = {}
    if threshold is not None:
        data["threshold"] = threshold

    if text:
        # Sending a POST request with text
        data = {
            **data,
            "content": text
        }
        response = requests.post(url, data=data, headers=headers)
    elif file_path:
        # Sending a POST request with a PDF file
        with open(file_path, "rb") as f:
            files = {"file": (file_path, f, "application/pdf")}
            response = requests.post(url, data=data, files=files, headers=headers)
    else:
        print("Either text or file path must be provided.")
        return

    # Print the response from the server
    if response.status_code == 200:
        print("Response:", response.json())
    else:
        print("Error:", response.status_code, response.text)

# Example usage:
BASE_API_URL = "http://127.0.0.1:7860/api/v1"
api_key = "my_api_key_1"

sample_string = '''Dr. Capy Cosmos, a capybara unlike any other, astounded the scientific community with his 
groundbreaking research in astrophysics. With his keen sense of observation and unparalleled ability to interpret 
cosmic data, he uncovered new insights into the mysteries of black holes and the origins of the universe. As he 
peered through telescopes with his large, round eyes, fellow researchers often remarked that it seemed as if the 
stars themselves whispered their secrets directly to him. Dr. Cosmos not only became a beacon of inspiration to 
aspiring scientists but also proved that intellect and innovation can be found in the most unexpected of creatures.'''

token = get_bearer_token(f"{BASE_API_URL}/auth/token", api_key)

threshold = None # threshold AI/Human detection, None for using default threshold

print("Sending raw text request...")
# Send text content
send_post_request(f"{BASE_API_URL}/predict", token, text=sample_string, threshold=threshold)

print("Sending pdf request...")
# Send a PDF file
send_post_request(f"{BASE_API_URL}/predict", token, file_path="assets/sample_input.pdf", threshold=threshold)
