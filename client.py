import requests
import argparse
import sys
from contextlib import ExitStack

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


def send_post_request(url, token, texts=None, file_paths=None, threshold=None):
    headers = {
       "Authorization": f"Bearer {token}"
    }


    data = {}
    if threshold is not None:
        data["threshold"] = threshold

    if texts:      
        # Sending a POST request with text
        texts = [texts] if isinstance(texts, str) else texts

        data = {
            **data,
            "contents": texts
        }
        response = requests.post(url, data=data, headers=headers)
    
    elif file_paths:
        # Sending a POST request with a PDF file
        file_paths = [file_paths] if isinstance(file_paths, str) else file_paths

        with ExitStack() as stack:
            # Open all files at once and ensure they're properly closed
            files = [
                ("files", (path, stack.enter_context(open(path, "rb")), "application/pdf"))
                for path in file_paths
            ]
            
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
        nargs='+',  # This means "one or more arguments"
        help='Path to PDF file(s) to analyze'
    )
    
    return parser.parse_args()
    

# Sample data as constants
SAMPLE_TEXT = '''Dr. Capy Cosmos, a capybara unlike any other, astounded the scientific community with his
groundbreaking research in astrophysics. With his keen sense of observation and unparalleled ability to interpret
cosmic data, he uncovered new insights into the mysteries of black holes and the origins of the universe. As he
peered through telescopes with his large, round eyes, fellow researchers often remarked that it seemed as if the
stars themselves whispered their secrets directly to him. Dr. Cosmos not only became a beacon of inspiration to
aspiring scientists but also proved that intellect and innovation can be found in the most unexpected of creatures.'''

SAMPLE_PDFS = {
    "1": "assets/sample_input_1.pdf",
    "2": "assets/sample_input_2.pdf",
}

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
            texts=text_content,
            threshold=args.threshold
        )
    else:  # PDF mode
        if len(args.pdf) == 1 and args.pdf[0].strip() in SAMPLE_PDFS:
            pdf_path = SAMPLE_PDFS[args.pdf[0].strip()]
            print(f"Using sample PDF: {pdf_path}")

        elif len(args.pdf) == 2 and {args.pdf[0].strip(), args.pdf[1].strip()} == {"1", "2"}:
            pdf_path = [SAMPLE_PDFS[args.pdf[0].strip()], SAMPLE_PDFS[args.pdf[1].strip()]]
            print(f"Using samples PDFs: {pdf_path}")
        else:
            print(f"Using provided PDF: {args.pdf}")
            pdf_path = args.pdf
            
        send_post_request(
            f"{args.base_url}/predict",
            token,
            file_paths=pdf_path,
            threshold=args.threshold
        )

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("\n", "*** Example usages ***")
        print("\t", "*", 'python script.py --text "" for using default demo text')
        print("\t", "*", 'python script.py --pdf "1" for using default demo pdf 1')
        print("\t", "*", 'python script.py --pdf "2" for using default demo pdf 2')
        print("\t", "*", 'python script.py --pdf "1" "2" for using both demo pdf')
        print("")

    main()
