# Deployment Guide for Streamlit Community Cloud

This guide explains how to deploy the Pharmacy RAG Assistant to Streamlit Community Cloud.

## Prerequisites

- A GitHub account
- This repository forked to your account

## Deployment Steps

### 1. Fork the Repository

1. Go to https://github.com/KrishnaHarish/pharmacy-rag-streamlit
2. Click "Fork" in the top-right corner
3. Wait for the fork to complete

### 2. Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your forked repository
4. Set the following:
   - **Branch**: `main` (or your working branch)
   - **Main file path**: `app.py`
   - **App URL**: Choose a custom subdomain (e.g., `pharmacy-rag-assistant`)

### 3. Configure Environment Variables (Optional)

If you want to use a different model or enable verbose logging:

1. Click "Advanced settings" before deploying
2. Add environment variables:
   ```
   MODEL_URL=https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
   MODEL_PATH=models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
   LLM_VERBOSE=false
   ```

### 4. Deploy

1. Click "Deploy!"
2. Wait for the app to build (first deployment takes ~5-10 minutes)
3. The model will download on first run (~640MB, takes a few minutes)

## First Run Behavior

**Expected Timeline:**
- Build time: 5-10 minutes (installing dependencies)
- Model download: 2-5 minutes (downloading TinyLlama ~640MB)
- Initialization: 30-60 seconds (loading embeddings, building FAISS index)
- **Total first run: ~10-15 minutes**

**Subsequent Runs:**
- Load time: 30-60 seconds (everything cached)
- No model download needed

## Model Size Considerations

### Free Tier Limits

Streamlit Community Cloud free tier has:
- **RAM**: ~1GB available
- **Storage**: Limited disk space
- **CPU**: Shared resources

### Recommended Models

1. **TinyLlama-1.1B-Q4_K_M** (~640MB) - **Default, best for free tier**
   - Fast inference
   - Fits in memory easily
   - Good quality for simple queries

2. **Phi-2-Q4_K_M** (~1.6GB) - May work on free tier
   - Better quality responses
   - Slower inference
   - Higher memory usage

3. **OpenHermes-2.5-Mistral-7B-Q2_K** (~2.5GB) - Requires more resources
   - Best quality
   - May cause OOM errors on free tier
   - Consider paid tier

### Changing the Model

To use a different model:

1. Find a GGUF model on [Hugging Face](https://huggingface.co/models?search=gguf)
2. Copy the download URL
3. Update environment variables in Streamlit Cloud:
   - Go to your app settings
   - Update `MODEL_URL` to the new model URL
   - Update `MODEL_PATH` to a matching local path
4. Restart the app

**Important**: Choose quantized models (Q2, Q4, Q5) to stay within memory limits.

## Pre-bundling the Model (Advanced)

To avoid downloading on first run, you can commit the model to your repository:

### ⚠️ Warning
- GitHub has a 100MB file size limit for regular files
- Use Git LFS for files >100MB
- Total repository size should be <1GB

### Steps

1. **Install Git LFS**:
   ```bash
   git lfs install
   ```

2. **Track GGUF files**:
   ```bash
   git lfs track "*.gguf"
   git add .gitattributes
   ```

3. **Download and commit model**:
   ```bash
   mkdir -p models
   cd models
   wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
   cd ..
   git add models/
   git commit -m "Add pre-bundled GGUF model"
   git push
   ```

4. **Verify in Streamlit Cloud**:
   - App will detect existing model
   - No download needed
   - Faster first run

## Troubleshooting

### Out of Memory (OOM) Errors

**Symptoms**: App crashes with "Killed" or "Out of memory" message

**Solutions**:
1. Use a smaller model (Q2 instead of Q4)
2. Reduce context window (n_ctx) to 1024 or 512
3. Reduce number of CPU threads to 2
4. Consider Streamlit Cloud paid tier

### Model Download Fails

**Symptoms**: Network error or timeout during download

**Solutions**:
1. Check MODEL_URL is accessible
2. Try a different model source
3. Pre-bundle the model (see above)
4. Check Streamlit Cloud logs for specific error

### Slow Inference

**Symptoms**: Response takes >30 seconds

**Solutions**:
1. Reduce max_tokens to 200-300
2. Increase CPU threads (if RAM allows)
3. Use smaller context window
4. Consider Q2 quantization for faster inference

### FAISS Index Issues

**Symptoms**: "Index not built" or search errors

**Solutions**:
1. Verify drugs.csv exists and has correct format
2. Check Streamlit Cloud logs
3. Clear cache: Settings → Clear cache → Rerun

## Monitoring

### Check App Logs

1. Go to your app in Streamlit Cloud
2. Click "Manage app"
3. View "Logs" tab
4. Look for errors or warnings

### Resource Usage

Monitor resource usage in logs:
- Memory usage
- CPU usage
- Disk space

If approaching limits, optimize (see troubleshooting above).

## Updating the App

### Update Code

1. Make changes in your forked repository
2. Commit and push to GitHub
3. Streamlit Cloud auto-deploys on push

### Update Dependencies

1. Edit `requirements.txt`
2. Commit and push
3. App rebuilds automatically

### Clear Cache

If cached resources cause issues:
1. Go to app settings
2. Click "Clear cache"
3. Rerun the app

## Custom Domain (Optional)

Streamlit Cloud Pro allows custom domains:
1. Upgrade to Pro plan
2. Go to app settings
3. Add custom domain
4. Update DNS records

## Security Notes

✅ **No secrets required**: App runs without API keys
✅ **All local**: No external API calls
✅ **Open source**: Code is fully auditable

⚠️ **User input**: Sanitize if displaying user-generated content
⚠️ **Model output**: LLM may generate incorrect information
⚠️ **Medical advice**: Add disclaimers (already included)

## Support

For issues:
1. Check [Streamlit Community Forum](https://discuss.streamlit.io/)
2. Open issue on GitHub repository
3. Check Streamlit Cloud [status page](https://status.streamlit.io/)

## License

MIT License - See LICENSE file for details
