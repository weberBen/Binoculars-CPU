import requests
import argparse
import sys

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

#%%

def parse_args():
    parser = argparse.ArgumentParser(
        description='AI Content Detection API Client',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Example usage:
    # Using sample text:
    python script.py --api-key "my_api_key_1" --text --use-sample
    
    # Using sample PDF:
    python script.py --api-key "my_api_key_1" --pdf --use-sample
    
    # Using custom text:
    python script.py --api-key "my_api_key_1" --text "Your text here"
    
    # Using custom PDF:
    python script.py --api-key "my_api_key_1" --pdf "path/to/file.pdf"'''
    )
    
    # Required arguments
    parser.add_argument(
        '--api-key',
        default="my_api_key_1",
        help='API key for authentication'
    )
    
    # Optional arguments
    parser.add_argument(
        '--base-url',
        default='http://127.0.0.1:7860/api/v1',
        help='Base API URL (default: http://127.0.0.1:7860/api/v1)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        help='Threshold for AI/Human detection (default: None for using API default)'
    )
    
    # Input type group (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--text',
        help='Raw text input to analyze'
    )
    input_group.add_argument(
        '--pdf',
        help='Path to PDF file to analyze'
    )

    return parser.parse_args()
    

# Sample data as constants
SAMPLE_TEXT = '''Dr. Capy Cosmos, a capybara unlike any other, astounded the scientific community with his
groundbreaking research in astrophysics. With his keen sense of observation and unparalleled ability to interpret
cosmic data, he uncovered new insights into the mysteries of black holes and the origins of the universe. As he
peered through telescopes with his large, round eyes, fellow researchers often remarked that it seemed as if the
stars themselves whispered their secrets directly to him. Dr. Cosmos not only became a beacon of inspiration to
aspiring scientists but also proved that intellect and innovation can be found in the most unexpected of creatures.'''

SAMPLE_PDF_PATH = "assets/sample_input.pdf"

def main():
    args = parse_args()
    
    # Get authentication token
    token = get_bearer_token(f"{args.base_url}/auth/token", args.api_key)
    
    # Process based on input type
    if args.text is not None :
        if len(args.text.strip()) == 0:
            print("Using sample text about Dr. Capy Cosmos...")
            print("Text:", SAMPLE_TEXT[:100] + "...")  # Preview first 100 chars
            text_content = SAMPLE_TEXT
        else:
            print("Using provided text...")
            text_content = args.text
            
        send_post_request(
            f"{args.base_url}/predict",
            token,
            text=text_content,
            threshold=args.threshold
        )
    else:  # PDF mode
        if len(args.pdf.strip()) == 0:
            print(f"Using sample PDF: {SAMPLE_PDF_PATH}")
            pdf_path = SAMPLE_PDF_PATH
        else:
            print(f"Using provided PDF: {args.pdf}")
            pdf_path = args.pdf
            
        send_post_request(
            f"{args.base_url}/predict",
            token,
            file_path=pdf_path,
            threshold=args.threshold
        )

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("\n", "*** Example usages ***")
        print("\t", "*", 'python script.py --text "" for using default demo text')
        print("\t", "*", 'python script.py --pdf "" for using default demo pdf')
        print("")

    main()
