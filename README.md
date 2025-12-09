# Pharmacy RAG Streamlit Application

A fully self-contained Retrieval-Augmented Generation (RAG) application for pharmacy-related queries that runs entirely locally on Streamlit Community Cloud with no external API keys required.

## Features

- **✅ No API Keys Required**: Runs completely locally using GGUF models
- **🚀 Self-Contained**: All models and data bundled or auto-downloaded
- **💊 Pharmacy Knowledge**: Pre-loaded drug database with 10+ medications
- **🔍 Semantic Search**: FAISS vector search with sentence-transformers embeddings
- **🤖 Local LLM**: llama-cpp-python with quantized TinyLlama model
- **📝 Citation Format**: Answers include [S#] citations for transparency
- **⚡ Cached Loading**: Fast reloads with Streamlit's @st.cache_resource

## Quick Start

### Deploy to Streamlit Community Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy from your forked repository
4. On first run, the app will download the TinyLlama GGUF model (~640MB)
5. Subsequent runs load from cache instantly

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/KrishnaHarish/pharmacy-rag-streamlit.git
   cd pharmacy-rag-streamlit
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Configuration

### Model Configuration

The app uses environment variables for model configuration:

- **MODEL_URL**: URL to download the GGUF model (default: TinyLlama-1.1B Q4_K_M)
- **MODEL_PATH**: Local path to save/load the model (default: `models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`)

#### Using a Different Model

Set environment variables to use a different GGUF model:

```bash
# Example: Using Phi-2 Q4 model
export MODEL_URL="https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf"
export MODEL_PATH="models/phi-2.Q4_K_M.gguf"
streamlit run app.py
```

#### Recommended Small Models for Streamlit Cloud

Due to Streamlit Community Cloud resource limits, use small quantized models:

- **TinyLlama-1.1B-Chat-Q4_K_M** (~640MB) - Default, best for free tier
- **Phi-2-Q4_K_M** (~1.6GB) - Better quality, may work on free tier
- **OpenHermes-2.5-Mistral-7B-Q2_K** (~2.5GB) - Requires more resources

⚠️ **Note**: Models >1GB may cause OOM errors on Streamlit Community Cloud free tier.

#### Pre-bundling the Model

To avoid download on first run (faster startup):

1. Download your GGUF model
2. Place it in `models/` directory
3. Update MODEL_PATH to match
4. Commit to repository (if size permits, <1GB recommended)

### Customizing the Drug Database

Edit `drugs.csv` to add/modify drug information:

```csv
drug,description
Aspirin,"Aspirin (acetylsalicylic acid) is a nonsteroidal anti-inflammatory drug..."
```

Required columns:
- `drug`: Drug name
- `description`: Detailed information (dosage, side effects, interactions, storage)

## Project Structure

```
pharmacy-rag-streamlit/
├── app.py                 # Main Streamlit application (local RAG)
├── drugs.csv             # Drug knowledge base
├── models/               # Directory for GGUF models (auto-created)
├── requirements.txt      # Minimal dependencies
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## Sample Questions

- What are the side effects of aspirin?
- How should insulin be stored?
- What is the recommended dosage for ibuprofen?
- What are drug interactions with warfarin?
- Tell me about metformin for diabetes

## Technology Stack

- **Streamlit**: Web interface and caching
- **llama-cpp-python**: Local GGUF model inference
- **sentence-transformers**: Text embeddings (all-MiniLM-L6-v2)
- **FAISS**: Vector similarity search (CPU version)
- **Pandas**: Drug data management
- **NumPy**: Numerical operations

## Model Hyperparameters

Configurable in the sidebar:

- **Context Window**: 512-4096 tokens (default: 2048)
- **CPU Threads**: 1-8 threads (default: 4)
- **Top-K Results**: 1-10 sources (default: 3)
- **Max Tokens**: 100-800 tokens (default: 400)
- **Temperature**: 0.0-1.0 (default: 0.7)

## Free Tier Considerations

**Streamlit Community Cloud Limits:**
- RAM: ~1GB available
- Storage: Limited, recommend models <1GB
- CPU: Shared resources
- Startup timeout: ~10 minutes

**Optimization Tips:**
- Use Q4 or Q2 quantized models
- Keep context window ≤2048
- Limit max_tokens to reduce generation time
- Use @st.cache_resource for all heavy operations

## Troubleshooting

**Model download fails:**
- Check MODEL_URL is accessible
- Verify internet connectivity
- Try pre-bundling the model

**Out of memory errors:**
- Use smaller quantized model (Q2_K instead of Q4_K_M)
- Reduce context window (n_ctx)
- Reduce number of CPU threads

**Slow inference:**
- Increase CPU threads (if RAM allows)
- Reduce max_tokens
- Use smaller model

## Important Disclaimer

⚠️ This application is for **informational purposes only** and should not replace professional medical advice. Always consult with a qualified healthcare provider or pharmacist for medical decisions.

## License

MIT License