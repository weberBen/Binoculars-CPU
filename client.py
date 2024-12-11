import requests

def send_post_request(url, text=None, file_path=None):
    if text:
        # Sending a POST request with text
        data = {"content": text}
        response = requests.post(url, data=data)
    elif file_path:
        # Sending a POST request with a PDF file
        with open(file_path, "rb") as f:
            files = {"file": (file_path, f, "application/pdf")}
            response = requests.post(url, files=files)
    else:
        print("Either text or file path must be provided.")
        return

    # Print the response from the server
    if response.status_code == 200:
        print("Response:", response.json())
    else:
        print("Error:", response.status_code, response.text)

# Example usage:
api_url = "http://127.0.0.1:8080/predict"

sample_string = '''Dr. Capy Cosmos, a capybara unlike any other, astounded the scientific community with his 
groundbreaking research in astrophysics. With his keen sense of observation and unparalleled ability to interpret 
cosmic data, he uncovered new insights into the mysteries of black holes and the origins of the universe. As he 
peered through telescopes with his large, round eyes, fellow researchers often remarked that it seemed as if the 
stars themselves whispered their secrets directly to him. Dr. Cosmos not only became a beacon of inspiration to 
aspiring scientists but also proved that intellect and innovation can be found in the most unexpected of creatures.'''

print("Sending raw text request...")
# Send text content
send_post_request(api_url, text=sample_string)

print("Sending pdf request...")
# Send a PDF file
send_post_request(api_url, file_path="assets/sample_input.pdf")
