# 🚀 Quick Start Guide

Get your RAG system running in 5 minutes!

---

## 📦 Step 1: Install Dependencies

```bash
cd rag_system
pip install -r requirements.txt
```

**Note:** This will install ~2GB of packages. Make sure you have enough disk space.

---

## 📚 Step 2: Test with Sample Document

We've included a sample document about Machine Learning. Let's index it!

```bash
python build_index.py --documents data/documents --output data/vector_store
```

**Expected output:**
```
================================================================================
BUILDING RAG INDEX
================================================================================

1️⃣  Processing documents from: data/documents
✅ Loaded 1 documents from data/documents
...
✅ INDEX BUILT SUCCESSFULLY
```

---

## 🧪 Step 3: Test the System

Interactive testing:

```bash
python test_rag.py --vector-store data/vector_store
```

**Try these queries:**
- "What is machine learning?"
- "What are the types of machine learning?"
- "What is overfitting?"
- "What libraries are popular for ML?"

**Example interaction:**
```
❓ Your question: What is machine learning?

🤔 Thinking...

────────────────────────────────────────────────────────────
📝 ANSWER
────────────────────────────────────────────────────────────
Machine Learning (ML) is a subset of artificial intelligence
that enables systems to learn and improve from experience
without being explicitly programmed. Instead of following
hard-coded rules, ML algorithms build mathematical models
based on sample data to make predictions or decisions.

────────────────────────────────────────────────────────────
📚 RETRIEVED CONTEXT
────────────────────────────────────────────────────────────

[1] Score: 0.856
Machine Learning (ML) is a subset of artificial intelligence
that enables systems to learn and improve from experience...

⏱️  Processing time: 2.34s
```

---

## 🌐 Step 4: Run Web UI (Optional)

For a beautiful graphical interface:

```bash
streamlit run ui/streamlit_app.py
```

Then open http://localhost:8501 in your browser.

**In the Web UI:**
1. Click "🚀 Initialize System"
2. Enter vector store path: `data/vector_store`
3. Click initialize and wait
4. Ask questions in the text box!

---

## 🔧 Step 5: Add Your Own Documents

1. **Add documents:**
   ```bash
   cp /path/to/your/pdfs/* data/documents/
   ```

2. **Rebuild index:**
   ```bash
   python build_index.py --documents data/documents --output data/vector_store
   ```

3. **Query your documents:**
   ```bash
   python test_rag.py --vector-store data/vector_store
   ```

---

## 🎯 Common Commands

### Build index with custom settings:
```bash
python build_index.py \
    --documents data/documents \
    --output data/vector_store \
    --chunk-size 512 \
    --chunk-overlap 50
```

### Test with different model:
```bash
# Faster (7B model)
python test_rag.py \
    --vector-store data/vector_store \
    --model mistralai/Mistral-7B-Instruct-v0.2

# More capable (141B model) - requires ~45GB VRAM
python test_rag.py \
    --vector-store data/vector_store \
    --model mistralai/Mixtral-8x22B-Instruct-v0.1
```

### Run API server:
```bash
python -m src.api.server

# Or with uvicorn
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

---

## 💡 Tips

1. **Start small:** Test with a few documents first
2. **GPU recommended:** Much faster for embeddings and generation
3. **4-bit quantization:** Enabled by default, saves 75% VRAM
4. **Chunk size:** 512 tokens works well for most documents
5. **Context chunks (k):** Start with 5, increase if needed

---

## 🐛 Troubleshooting

### "CUDA out of memory"
```bash
# Use smaller model
python test_rag.py --model mistralai/Mistral-7B-Instruct-v0.2

# Or use CPU (slow)
# Edit configs/config.yaml: device: "cpu"
```

### "No module named 'src'"
```bash
# Make sure you're in the rag_system directory
cd rag_system
python test_rag.py --vector-store data/vector_store
```

### "No documents found"
```bash
# Check documents directory
ls -la data/documents/

# Make sure files are .pdf, .txt, .md, or .docx
```

---

## 📊 Expected Performance

On NVIDIA L40S (48GB VRAM):

| Task | Time | Memory |
|------|------|--------|
| Index 100 docs | ~1 min | ~8 GB |
| Index 1,000 docs | ~10 min | ~8 GB |
| Query (Mistral Nemo) | ~2-3 sec | ~8 GB |
| Query (Mistral 7B) | ~1-2 sec | ~5 GB |

---

## ✅ Next Steps

Once you have the system running:

1. **Add more documents** - The more context, the better
2. **Customize prompts** - Edit `src/generation/prompts.py`
3. **Tune retrieval** - Adjust semantic/keyword weights
4. **Deploy API** - Use FastAPI for production
5. **Build applications** - Chatbots, Q&A systems, documentation search

---

## 📚 More Information

- Full documentation: [README.md](README.md)
- Configuration guide: [configs/config.yaml](configs/config.yaml)
- API reference: http://localhost:8000/docs (when server is running)

---

**Happy RAG building! 🚀**
