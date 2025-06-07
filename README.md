## Features

- Loads documents from the `data/` directory
- Uses Hugging Face’s `sentence-transformers/all-MiniLM-L6-v2` model to create embeddings
- Uses `distilgpt2` for text generation as the LLM backend
- Interactive command-line prompt for questions and answers
- Securely accepts the Hugging Face API token as user input at runtime

## Requirements
- Python 3.7 or higher
- Install dependencies: pip install requirements.txt

## Setup
- Clone this repository or save the script llama_hf_textgen.py locally.
- Create a folder named data in the same directory as the script.
- Add plain text documents you want to index into the data/ folder.
- Get your Hugging Face API token from https://huggingface.co/settings/tokens
- Make sure your token has read access.

## Troubleshooting
- If you get authentication errors, verify your API token and permissions.
- If the vector index building fails, check that your documents are correctly formatted text files.
- If the query returns no answers, try expanding or cleaning your documents.
