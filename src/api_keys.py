from dotenv import load_dotenv
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = os.path.join(os.path.dirname(script_dir), 'SECRETS.env')

load_dotenv(script_dir)

oai_key = os.getenv('OPENAI_API_KEY')
gem_key = os.getenv('GEMINI_API_KEY')
gcp_key = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
azure_key = os.getenv('AZURE_KEY')
azure_endpoint = os.getenv('AZURE_ENDPOINT')
LOCATION = os.getenv('LOCATION')
PROJECT_ID = os.getenv('PROJECT_ID')