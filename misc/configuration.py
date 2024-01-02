from dotenv import load_dotenv

load_dotenv()
import os
from pathlib import Path

# Directories
ROOT_DIR = Path(__file__).parent.parent.absolute()
EFS_DIR = Path("data")


# default index name
DEFAULT_INDEX_NAME = "index-start"

# Mappings
EMBEDDING_DIMENSIONS = {
    "thenlper/gte-base": 768,
    "thenlper/gte-large": 1024,
    "local:BAAI/bge-small-en-v1.5": 384,
    "local:BAAI/bge-base-en-v1.5": 768,
    "local:BAAI/bge-large-en-v1.5": 1024,
    "local:BAAI/bge-large-en": 1024,
    "local:BAAI/bge-base-en": 768,
    "local:BAAI/bge-small-en": 384,
    "text-embedding-ada-002": 1536,
    "gte-large-fine-tuned": 1024,
    "sentence-transformers/all-mpnet-base-v2": 768,
}
MAX_CONTEXT_LENGTHS = {
    "gpt-4": 8192,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-4-1106-preview": 128000,
    "meta-llama/Llama-2-7b-chat-hf": 4096,
    "meta-llama/Llama-2-13b-chat-hf": 4096,
    "meta-llama/Llama-2-70b-chat-hf": 4096,
    "codellama/CodeLlama-34b-Instruct-hf": 16384,
    "mistralai/Mistral-7B-Instruct-v0.1": 65536,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 32768,
}


EMBEDDING_MODEL_NAMES_TO_INDEX_NAMES = {
    "text-embedding-ada-002": "index-start-ada-002",
    "local:BAAI/bge-small-en": "index-start-bge-small-en",
    "local:BAAI/bge-base-en": "index-start-bge-base-en",
    "local:BAAI/bge-large-en": "index-start-bge-large-en",
    "local:BAAI/bge-large-en-v1.5": "index-start-bge-large-en-v1.5",
    "local:BAAI/bge-base-en-v1.5": "index-start-bge-base-en-v1.5",
    "local:BAAI/bge-small-en-v1.5": "index-start-bge-small-en-v1.5",
    "sentence-transformers/all-mpnet-base-v2": "index-start-mpnet-base-v2",
    "thenlper/gte-base": "index-start-gte-base",
    "thenlper/gte-large": "index-start-gte-large",
    "gte-large-fine-tuned": "index-start-gte-large-fine-tuned",
    # Add more embedding models and index names as needed
}

# Text Processing Prompt Templates
SYSTEM_PROMPT_TEXT_PROCESSING = """You are an expert in reading text from PDF file. 
            You will be given two texts. The first text is a raw text containing both texts and tables.
            The second text contains only tables from the first text, but with better formatting.
            Your job is to merge the second text into the right places in the first text and omit the tables in the first text suitably. 
            For example:
            ## First Text:
            ---
            here are the important items in the table.
            Item Description Function
            1. Apple iPhone 
            13 Pro Max (256GB, Graphite)
            Used
            2. Apple iPhone 
            14 Pro Max (512GB, Gold)
            Unlocked (Renewed)

            Caution: One cannot refund after 7 days of purchase.  
            ---
            ## Second Text:
            ---
            Item: Apple iphone, Description: 13 Pro Max (256GB, Graphite), Function: Used;
            Item: Apple iphone, Description: 14 Pro Max (512GB, Gold), Function: Unlocked (Renewed);
            ---

            ### DESIRED OUTPUT:
            here are the important items in the table.
            Item: Apple iphone, Description: 13 Pro Max (256GB, Graphite), Function: Used;
            Item: Apple iphone, Description: 14 Pro Max (512GB, Gold), Function: Unlocked (Renewed);

            Caution: One cannot refund after 7 days of purchase. 
            """


USER_PROMPT_TEXT_PROCESSING = """First Text:\n{first_text}\nSecond Text:\n{second_text}\n###
Please provide your output starting with the capital words "DESIRED OUTPUT:"
"""

SYSTEM_PROMPT_TEXT_PROCESSING_V0 = """You are an expert in reading text from PDF file. 
You will be given a text where reading order may not be natural as human reading. 
Your job is to reorganize the text naturally. For example:
Original text:
---
Item Description Function
1. Apple iPhone 
13 Pro Max (256GB, Graphite)
Used
2. Apple iPhone 
13 Pro Max (256GB, Graphite)
Unlocked (Renewed)  
---
Reorganized text:
---
Apple iphone, description is 13 Pro Max (256GB, Graphite), and function is used.
Apple iphone, description is 13 Pro Max (256GB, Graphite), and function is unlocked.

"""

USER_PROMPT_TEXT_PROCESSING_V0 = """Here is the text: '''{document} '''\n###
Please output ONLY the new text, nothing else. 
Start with the word 'TEXT:'"""

### API KEYS
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_API_BASE = os.environ["OPENAI_API_BASE"]
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = os.environ["PINECONE_ENVIRONMENT"]
ANYSCALE_API_BASE = os.environ["ANYSCALE_API_BASE"]
ANYSCALE_API_KEY = os.environ["ANYSCALE_API_KEY"]
COHERE_API_KEY = os.environ["COHERE_API_KEY"]
QDRANT_URI = os.environ["QDRANT_URI"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]