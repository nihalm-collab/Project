# 📚 Kitap Asistanı Chatbot

Bu proje, **Akbank GenAI Bootcamp** kapsamında geliştirilmiş bir **RAG (Retrieval-Augmented Generation)** tabanlı Türkçe kitap asistanıdır.  
Veri seti olarak [Kitapyurdu Yorumlar](https://huggingface.co/datasets/alibayram/kitapyurdu_yorumlar) kullanılmıştır.

## 🎯 Amaç
Kullanıcılara kitaplar hakkında yorumlara dayalı özet ve öneriler sunmak.

## 🧩 Kullanılan Teknolojiler
- Gemini API (LLM)
- Sentence Transformers
- ChromaDB (vektör veritabanı)
- Hugging Face Datasets
- Flask (ilerleyen sürümde arayüz)

## ⚙️ Çalıştırma
```bash
pip install -r requirements.txt
python app.py
