import streamlit as st
import torch
import numpy as np
import nltk
import sys
import os
import nltk
nltk.download('punkt')


sys.path.append(os.path.abspath("./src"))
from models.model_build import ExtSummarizer
# nltk.download('punkt')

def tokenize_sentences(text):
    return nltk.sent_tokenize(text)

def summarize(text, model, device, num_sentences=3):
    """ Melakukan inferensi ringkasan pada teks input. """
    sentences = tokenize_sentences(text)
    if len(sentences) == 0:
        return "Teks terlalu pendek untuk diringkas."

    src_txt = [sentences]  # Model memproses dalam bentuk batch
    mask_cls = torch.ones((1, len(sentences)), dtype=torch.bool, device=device)

    with torch.no_grad():
        sent_scores, _ = model(src_txt, mask_cls)

    sent_scores = sent_scores.cpu().numpy().flatten()
    sorted_indices = np.argsort(-sent_scores)  # Urutkan skor dari tertinggi ke terendah
    selected_indices = sorted_indices[:num_sentences]  # Pilih N kalimat terbaik
    selected_indices.sort()  # Urutkan berdasarkan urutan aslinya dalam teks

    summary = " ".join([sentences[i] for i in selected_indices])
    return summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./models/sbertsumext_model/s2_xlsum_sbert_base_p1_ext/model_step_45000.pt"  # Pastikan path benar
checkpoint = torch.load(model_path, map_location=device)
model = ExtSummarizer(None, device, checkpoint)
model.eval()  # Set model ke mode evaluasi


st.title("📜 SBERTSumExt - Summarization Web App")
st.write("Masukkan teks panjang untuk diringkas:")


input_text = st.text_area("Teks Dokumen", height=200)

num_sentences = st.slider("Jumlah kalimat dalam ringkasan", min_value=1, max_value=5, value=2)

# Tombol untuk menjalankan model**
if st.button("Ringkas!"):
    if input_text.strip():
        summary = summarize(input_text, model, device, num_sentences)
        st.subheader("Hasil Ringkasan:")
        st.write(summary)
    else:
        st.warning("Harap masukkan teks terlebih dahulu.")
