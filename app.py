"""
📚 Kitap Asistanı Chatbot (Akbank GenAI Bootcamp)
--------------------------------------------------
Bu proje, Kitapyurdu yorum verisetine dayalı olarak
RAG (Retrieval Augmented Generation) mimarisiyle
çalışan bir kitap asistanı chatbotudur.
"""

import os
import time
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

# -----------------------------#
# Ortam Değişkenleri
# -----------------------------#
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./persist")

# -----------------------------#
# Yardımcı Fonksiyonlar
# -----------------------------#
def simple_clean(text):
    if not text:
        return ""
    return " ".join(text.split()).strip()

def print_divider():
    print("=" * 60)

# -----------------------------#
# Veri Seti Yükleme
# -----------------------------#
print("📥 Veri seti yükleniyor...")
dataset = load_dataset("alibayram/kitapyurdu_yorumlar")

split = list(dataset.keys())[0]
columns = dataset[split].column_names
print_divider()
print("📊 Veri seti kolonları:", columns)

# Yorum metin kolonu belirleme
comment_col = next(
    (c for c in columns if "yorum" in c.lower() or "comment" in c.lower() or "text" in c.lower()),
    None,
)
if not comment_col:
    raise ValueError("Yorum metni içeren bir kolon bulunamadı!")

# -----------------------------#
# Veri Hazırlama
# -----------------------------#
MAX_DOCS = 2000
docs, metadatas, ids = [], [], []
for i, row in enumerate(dataset[split]):
    if i >= MAX_DOCS:
        break
    text = simple_clean(row.get(comment_col, ""))
    if len(text) < 5:
        continue
    docs.append(text)
    meta = {k: row[k] for k in row if k in ["kitap_adi", "yazar_adi", "puan", "kategori"]}
    metadatas.append(meta)
    ids.append(f"doc_{i}")

print(f"📚 Toplam {len(docs)} yorum yüklendi.")

# -----------------------------#
# Embedding Model
# -----------------------------#
print("🧠 Embedding modeli yükleniyor...")
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

print("📈 Embedding'ler hesaplanıyor...")
BATCH = 64
embeddings = np.vstack([
    embed_model.encode(docs[i:i + BATCH], show_progress_bar=False)
    for i in range(0, len(docs), BATCH)
])
print("✅ Embedding tamamlandı:", embeddings.shape)

# -----------------------------#
# ChromaDB (Vektör Veritabanı)
# -----------------------------#
print("💾 ChromaDB oluşturuluyor...")
client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIR))
COLLECTION_NAME = "kitapyurdu_yorumlar"

if COLLECTION_NAME in [c.name for c in client.list_collections()]:
    client.delete_collection(COLLECTION_NAME)

collection = client.create_collection(COLLECTION_NAME)
collection.add(
    documents=docs,
    metadatas=metadatas,
    ids=ids,
    embeddings=embeddings.tolist(),
)
client.persist()
print("✅ Koleksiyon kaydedildi.")

# -----------------------------#
# Retrieval Fonksiyonu
# -----------------------------#
def retrieve(query, n_results=5):
    q_emb = embed_model.encode([query], show_progress_bar=False)
    res = collection.query(query_embeddings=q_emb.tolist(), n_results=n_results, include=["documents", "metadatas", "distances"])
    results = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        results.append({"doc": doc, "meta": meta, "distance": dist})
    return results

# -----------------------------#
# Gemini API (Stub)
# -----------------------------#
def generate_answer_with_gemini(prompt):
    """
    Bu fonksiyon Gemini API ile gerçek bağlantı kurmak içindir.
    Gerçek bağlantı örneği:

    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
    return response.text

    Şimdilik demo amaçlı sahte yanıt döner.
    """
    return "💡 Tahmini yanıt:\n\n" + prompt[:600]

# -----------------------------#
# CLI Chat Döngüsü
# -----------------------------#
def ask_loop():
    print_divider()
    print("🤖 Kitap Asistanı Chatbot (çıkmak için 'exit' yaz)")
    while True:
        query = input("\n🔎 Soru veya kitap adı: ").strip()
        if query.lower() in ["exit", "quit"]:
            break
        retrieved = retrieve(query, n_results=5)
        print_divider()
        print("📖 En alakalı yorumlar:")
        for i, r in enumerate(retrieved, 1):
            meta = ", ".join(f"{k}: {v}" for k, v in r["meta"].items()) if r["meta"] else ""
            print(f"[{i}] {r['doc'][:200]}... | {meta}")
        context = "\n\n".join([r["doc"] for r in retrieved])
        prompt = f"Kullanıcının sorusu: {query}\n\nAşağıdaki yorumlara göre kısa bir özet veya tavsiye üret:\n{context}"
        print_divider()
        print(generate_answer_with_gemini(prompt))
        print_divider()

if __name__ == "__main__":
    print("🚀 Uygulama başlatılıyor...")
    ask_loop()
