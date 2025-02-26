import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from models.model_build import ExtSummarizer
import nltk

nltk.download('punkt')

def tokenize_sentences(text):
    """ Membagi teks input menjadi daftar kalimat. """
    return nltk.sent_tokenize(text)

def summarize(text, model, device, num_sentences=3):
    """ Melakukan inferensi ringkasan pada teks input. """
    sentences = tokenize_sentences(text)  # Tokenisasi teks menjadi kalimat
    if len(sentences) == 0:
        return "Teks terlalu pendek untuk diringkas."

    src_txt = [sentences]  # Model memproses dalam bentuk batch
    mask_cls = torch.ones((1, len(sentences)), dtype=torch.bool, device=device)

    with torch.no_grad():
        sent_scores, _ = model(src_txt, mask_cls)  # Dapatkan skor kalimat

    sent_scores = sent_scores.cpu().numpy().flatten()
    sorted_indices = np.argsort(-sent_scores)  # Urutkan skor dari tertinggi ke terendah
    selected_indices = sorted_indices[:num_sentences]  # Pilih N kalimat terbaik
    selected_indices.sort()  # Urutkan berdasarkan urutan kemunculan dalam teks

    summary = " ".join([sentences[i] for i in selected_indices])
    return summary

if __name__ == "__main__":
    # Konfigurasi perangkat (GPU jika tersedia)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Path model SBertSumExt yang telah dilatih
    model_path = "../models/sbertsumext_model/model_step_12000.pt"

    # Load model SBertSumExt
    checkpoint = torch.load(model_path, map_location=device)
    model = ExtSummarizer(None, device, checkpoint)
    model.eval()  # Set model ke mode evaluasi

    # **Input teks langsung dalam kode**
    input_text = """
    song zuying terkenal dengan lagu-lagu patriotiknya . media resmi pemerintah melaporkan song zuying tampil bersama penyanyi lain dalam pentas keliling ke kepulauan spratlys , yang diakui sebagai milik cina . di kawasan itu -yang antara lain juga diklaim malaysia , vietnam , dan filipina- cina melakukan reklamasi untuk membuat pulau baru dan membangun landasan pacu serta sarana militer . para wartawan melaporkan foto-foto pentas song zuying yang diterbitkan xinhua sekaligus pula menampilkan foto yang jarang tentang pembangunan secara meluas yang dilakukan cina di kawasan pulau karang tersebut . cina melakukan reklamasi untuk membangun pulau baru di kawasan pulau karang yang masih jadi sengketa . pelabuhan , mercusuar , beberapa bangunan , serta kapal amfibi besar jenis 071 -yang mampu membawa empat helikopter dan 800 pasukan- tampak di beberapa foto tersebut . dalam acara itu , song antara lain membawakan lagi yang berjudul 'ode untuk para pembela laut selatan ' . vietnam juga mengklaim sebagian wilayah di laut cina selatan . terkenal dengan lagu-lagu patriotik , song -yang pernah tampil bersama celine dion di stasiun tv cina- dilaporkan memiliki penggemar meluas di kalangan pekerja bangunan . pentasnya tersebut disiarkan langsung di stasiun tv pemerintah , cctv . selain menggelar pentas ke pulau yang diberi nama yongshu oleh cina , song juga tampil di pulau lain yang lebih kecil , cuarteron .
    """

    # **Jumlah kalimat yang ingin dipilih untuk ringkasan**
    num_sentences = 2

    # **Jalankan inferensi**
    summary = summarize(input_text, model, device, num_sentences)

    print("\n📌 Ringkasan:\n", summary)

