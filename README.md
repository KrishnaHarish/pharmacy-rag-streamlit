# Pharmacy RAG Streamlit Application

A Retrieval-Augmented Generation (RAG) application for pharmacy-related queries built with Streamlit and LangChain.

## Features

- **Question Answering**: Ask questions about pharmacy topics and get accurate responses
- **Source Citations**: View the source documents used to generate answers
- **Interactive UI**: User-friendly Streamlit interface
- **Vector Search**: Efficient document retrieval using ChromaDB
- **LLM Integration**: Powered by GitHub-hosted language models (gpt-4o)

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/KrishnaHarish/pharmacy-rag-streamlit.git
   cd pharmacy-rag-streamlit
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the Application**
   
   The application uses GitHub-hosted models and does not require OpenAI API keys.
   
   If running in GitHub Codespaces or a GitHub-authenticated environment, the models will be automatically available.
   
   For local development, you may need to configure alternative model providers in the code.

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Usage

1. Launch the application using the command above
2. The app will automatically use GitHub-hosted models (no API key needed in GitHub Codespaces)
3. The app will automatically load sample pharmacy documents
4. Type your pharmacy-related question in the text area
5. Click "Get Answer" to receive a response with source citations

## Project Structure

```
pharmacy-rag-streamlit/
├── app.py                 # Main Streamlit application
├── rag_utils.py          # RAG pipeline utilities
├── data/                 # Sample pharmacy documents
│   └── pharmacy_info.txt
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## Sample Questions

- What are the side effects of aspirin?
- How should insulin be stored?
- What is the recommended dosage for ibuprofen?
- What are drug interactions with warfarin?

## Technology Stack

- **Streamlit**: Web interface
- **LangChain**: RAG pipeline orchestration
- **GitHub Models**: AI models hosted by GitHub (gpt-4o)
- **ChromaDB**: Vector database for document storage
- **Python-dotenv**: Environment variable management

## Configuration

This application is configured to use GitHub-hosted models through the Copilot infrastructure:
- **Model**: gpt-4o (GitHub-hosted)
- **No API keys required** when running in GitHub Codespaces or authenticated GitHub environments
- Configuration file: `.github/copilot/coding_agents.yml`

## License

MIT License