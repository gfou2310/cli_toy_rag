# RAG PoC

A Retrieval-Augmented Generation (RAG) PoC built with Haystack that combines document retrieval with language model 
generation for accurate and context-aware responses.

## Description

This project implements a RAG pipeline that:
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

2. Install the requirements:
```bash
pip install -r requirements.txt
```

#### Option 2: Using Conda
1. Create a new conda environment:
```bash
conda create -n rag_env python=3.12
conda activate rag_env
```

2. Install pip requirements in conda environment:
```bash
conda install pip  # Ensure pip is available in conda env
pip install -r requirements.txt
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
    - Start an interactive session where you can:
        - Enter questions/queries
        - Get AI-generated responses based on the documents
        - Type 'quit' or 'exit' to end the session

## Configuration

The system can be configured through:
- `CHAT_MODEL_CONFIG`: Controls the language model parameters
- `RETRIEVER_CONFIG`: Manages document retrieval settings

Key configurations include:
- `TOP_K`: Number of documents to retrieve
- `MAX_NEW_TOKENS`: Maximum length of generated responses
- `DO_SAMPLE`: Controls whether to use greedy or sampling-based generation
- `REPETITION_PENALTY`: Adjusts the penalty for repetitive text

## Requirements

Main dependencies:
- torch
- transformers
- sentence-transformers
- qdrant-client
- haystack
- farm-haystack
- faiss-cpu

For a complete list of dependencies, see `requirements.txt`.

A Retrieval-Augmented Generation (RAG) system built with Haystack that combines document retrieval with language model generation for accurate and context-aware responses.

## Description

This project implements a RAG pipeline that:
- Uses vector store for efficient document retrieval
- Leverages language models for generating responses
- Combines retrieved context with queries for improved accuracy

## Installation

### Prerequisites
- Python 3.12+
- pip




## Requirements

Main dependencies:
- torch
- transformers
- sentence-transformers
- qdrant-client
- haystack
- farm-haystack
- faiss-cpu

For a complete list of depe

ndencies, see `requirements.txt`.# rag_assignment

2. Create a `.pre-commit-config.yaml` file in your project root with the following content:
```yaml
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files
    -   id: check-merge-conflict

-   repo: https://github.com/psf/black
    rev: 24.2.0
    hooks:
    -   id: black
        language_version: python3

-   repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
    -   id: isort
        args: ["--profile", "black"]

-   repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
    -   id: flake8
        additional_dependencies: [flake8-docstrings]

-   repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
    -   id: mypy
        additional_dependencies: [types-all]
```

3. Install the pre-commit hooks:
```bash
pip install pre-commit
```

4. (Optional) Run pre-commit on all files:
```bash
pre-commit run --all-files
```

The pre-commit configuration includes:
- `black`: Code formatting
- `isort`: Import sorting
- `flake8`: Code linting
- `mypy`: Static type checking
- Basic file checks (trailing whitespace, yaml validation, etc.)

Pre-commit hooks will run automatically on every commit. If any hooks fail, the commit will be aborted and issues will need to be fixed before committing again.
