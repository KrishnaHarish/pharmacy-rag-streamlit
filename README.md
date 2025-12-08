# Pharmacy RAG Streamlit Application

A Retrieval-Augmented Generation (RAG) application for pharmacy-related queries built with Streamlit and LangChain.

## Features

- **Question Answering**: Ask questions about pharmacy topics and get accurate responses
- **Source Citations**: View the source documents used to generate answers
- **Interactive UI**: User-friendly Streamlit interface
- **Vector Search**: Efficient document retrieval using ChromaDB
- **LLM Integration**: Powered by OpenAI's language models

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

3. **Configure API Keys**
   
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

   Or configure through the Streamlit sidebar when running the app.

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Usage

1. Launch the application using the command above
2. Enter your OpenAI API key in the sidebar (if not set in .env)
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
- **OpenAI**: Language model for generation
- **ChromaDB**: Vector database for document storage
- **Python-dotenv**: Environment variable management

## License

MIT License