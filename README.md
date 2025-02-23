# RAG PoC

A Retrieval-Augmented Generation (RAG) PoC built with Haystack that combines document retrieval with language model 
generation for context-aware responses.

## Description

This project implements a RAG pipeline that:
- Uses a NN for layout parsing of the documents
- Uses vector store for efficient document retrieval
- Leverages language models for generating responses
- Combines retrieved context with queries for improved accuracy

## Installation

### Prerequisites
- Python 3.12+
- pip or conda

### Setting up the environment

#### Option 1: Using venv

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

#### Option 2: Using Conda
1. Create a new conda environment:
```bash
conda create -n rag_env python=3.12
conda activate rag_env
```

2. Install the requirements:
```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR:
```bash
sudo apt install tesseract-ocr -y
```

or for macOS:
```bsah
brew install tesseract
```

4. Install libjpeg-dev:
```bash
sudo apt-get install libjpeg-dev
```

or for macOS:
```bash
brew install libjpeg
```

5. Install the following github repo to use a NN for layout detection:
```bash
pip install "git+https://github.com/facebookresearch/detectron2.git@v0.5#egg=detectron2"
```

## Usage

The system is designed to run as an interactive command-line application. To use the system:

1. Ensure you have set up your environment (using either venv or conda as described above)

2. Run the main application:
```bash
python main.py
```

3. The system will:
    - Check for an existing vector store
    - Create a new vector store if none exists
      - Trigger the ingestion pipeline
    - Start an interactive session where you can:
        - Enter questions/queries
        - Get AI-generated responses based on the documents
        - Type 'quit' or 'exit' to end the session
