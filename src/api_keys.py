from dotenv import load_dotenv
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = os.path.join(os.path.dirname(script_dir), 'SECRETS.env')

load_dotenv(script_dir)

oai_key = os.getenv('OPENAI_API_KEY')
gem_key = os.getenv('GEMINI_API_KEY')