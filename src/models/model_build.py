import torch
import torch.nn as nn
from models.encoder import Classifier, ExtTransformerEncoder
from sentence_transformers import SentenceTransformer #SBERT
import os



class ExtSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.device = device

        # Load IndoSBERT sebagai Sentence Encoder
        # model_name = "../models/indobert-base-p1"
        model_name = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/indobert-base-p1"))
        self.sbert = SentenceTransformer(model_name).to(device)
        self.hidden_size = 768  # Dimensi output IndoSBERT

        # Layer Transformer untuk menentukan kalimat penting
        self.ext_layer = ExtTransformerEncoder(
            d_model=self.hidden_size, 
            d_ff=2048, 
            heads=8, 
            dropout=0.2, 
            num_inter_layers=2
        )

        # Load model yang telah dilatih
        # if checkpoint is not None:
        #     self.load_state_dict(checkpoint['model'], strict=True)

        # self.to(device)
        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=False)  # Pastikan strict=False untuk mencegah error
        else:
            print("Error: Tidak ada checkpoint yang dimuat. Pastikan model sudah dilatih sebelumnya.")
            exit(1)  # Keluar dari program jika tidak ada model yang dimuat

        self.to(device)

       

    def forward(self, src_txt, mask_cls):
        """
        Parameters:
            src_txt: List of sentences (input teks per dokumen)
            mask_cls: Mask untuk menandai keberadaan kalimat dalam batch
        
        Returns:
            sent_scores: Skor kepentingan setiap kalimat
        """
        batch_size = len(src_txt)
        
        # Encode setiap dokumen menggunakan IndoSBERT
        sents_vec_list = [
            self.sbert.encode(doc, convert_to_tensor=True, show_progress_bar=False).to(self.device)
            for doc in src_txt
        ]

        # Temukan jumlah kalimat maksimum untuk padding
        max_sents = max(vec.shape[0] for vec in sents_vec_list)
        embedding_dim = sents_vec_list[0].shape[-1]  # Biasanya 768

        # Buat tensor padded untuk batch processing
        padded_sents_vec = torch.zeros((batch_size, max_sents, embedding_dim), device=self.device)
        for i, vec in enumerate(sents_vec_list):
            padded_sents_vec[i, :vec.shape[0], :] = vec  

        # Pastikan mask_cls cocok dengan dimensi embedding
        min_sents = min(padded_sents_vec.shape[1], mask_cls.shape[1])
        padded_sents_vec = padded_sents_vec[:, :min_sents, :]
        mask_cls = mask_cls[:, :min_sents]

        # Terapkan mask_cls pada representasi kalimat
        sents_vec = padded_sents_vec * mask_cls[:, :, None].float()

        # Dapatkan skor penting untuk setiap kalimat
        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls
