import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertForMaskedLM, BertConfig
from torch.nn import CrossEntropyLoss

from models.at import Block

class ContrastiveAligner(nn.Module):
    def __init__(self, visual_feature_dim, text_feature_dim, aligned_dim, temperature=0.1):
        super(ContrastiveAligner, self).__init__()
        self.visual_transform = nn.Linear(visual_feature_dim, aligned_dim)
        self.text_transform = nn.Linear(text_feature_dim, aligned_dim)
        self.temperature = temperature

    def forward(self, visual_embedded, text_embedded):
        aligned_visual = F.normalize(self.visual_transform(visual_embedded), dim=-1)
        aligned_text = F.normalize(self.text_transform(text_embedded), dim=-1)

        similarity_matrix = torch.matmul(aligned_visual, aligned_text.t()) / self.temperature

        labels = torch.arange(similarity_matrix.size(0)).to(similarity_matrix.device)
        loss = (F.cross_entropy(similarity_matrix, labels) +
                F.cross_entropy(similarity_matrix.t(), labels)) / 2
        return loss

class MultimodalAlignNet(nn.Module):
    def __init__(self, visual_feature_dim, text_feature_dim, audio_feature_dim, aligned_dim, temperature=0.1):
        super(MultimodalAlignNet, self).__init__()
        self.visual_transform = nn.Linear(visual_feature_dim, aligned_dim)
        self.text_transform = nn.Linear(text_feature_dim, aligned_dim)
        self.audio_transform = nn.Linear(audio_feature_dim, aligned_dim)
        self.temperature = temperature
        self.config = BertConfig.from_pretrained('bert-base-uncased')
        self.block = Block(aligned_dim,8)
        model = BertForMaskedLM.from_pretrained('bert-base-uncased',config=self.config)
        self.cls = model.cls

    def forward(self, visual_embedded, text_embedded, audio_embedded, label = None):
        aligned_visual = F.normalize(self.visual_transform(visual_embedded), dim=-1)
        aligned_text = F.normalize(self.text_transform(text_embedded), dim=-1)
        aligned_audio = F.normalize(self.audio_transform(audio_embedded), dim=-1)

        visual_text_similarity = torch.matmul(aligned_visual, aligned_text.t()) / self.temperature
        visual_audio_similarity = torch.matmul(aligned_visual, aligned_audio.t()) / self.temperature
        text_audio_similarity = torch.matmul(aligned_text, aligned_audio.t()) / self.temperature

        labels = torch.arange(visual_text_similarity.size(0)).to(visual_text_similarity.device)
        loss_vtc = (F.cross_entropy(visual_text_similarity, labels) +
                F.cross_entropy(visual_text_similarity.t(), labels)) / 2
        loss_vac = (F.cross_entropy(visual_audio_similarity, labels) +
                F.cross_entropy(visual_audio_similarity.t(), labels) ) / 2
        loss_tac = (F.cross_entropy(text_audio_similarity, labels) +
                F.cross_entropy(text_audio_similarity.t(), labels)) / 2
        return  loss_vtc + loss_vac + loss_tac

    def forward2(self, visual_embedded, text_embedded, label = None):
        aligned_visual = F.normalize(self.visual_transform(visual_embedded), dim=-1)
        aligned_text = F.normalize(self.text_transform(text_embedded), dim=-1)

        visual_text_similarity = torch.matmul(aligned_visual, aligned_text.t()) / self.temperature

        labels = torch.arange(visual_text_similarity.size(0)).to(visual_text_similarity.device)
        loss_vtc = (F.cross_entropy(visual_text_similarity, labels) +
                F.cross_entropy(visual_text_similarity.t(), labels)) / 2

        return  loss_vtc

    def mlm(self, visual_embedded, audio_embedded, text_embedded, label = None):
        aligned_visual = F.normalize(self.visual_transform(visual_embedded), dim=-1)
        aligned_text = F.normalize(self.text_transform(text_embedded), dim=-1)
        aligned_audio = F.normalize(self.audio_transform(audio_embedded), dim=-1)        
        f_f = torch.cat([aligned_visual, aligned_text, aligned_audio], dim=0)
        f_f = self.block(f_f.unsqueeze(0)).squeeze(0) # len , 768
        f_f_f = self.cls(f_f) # len, 30522
        # MLM
        if label is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            loss_mlm = loss_fct(f_f_f.view(-1, self.config.vocab_size), label.view(-1))
        else:
            loss_mlm = None
        return  loss_mlm, f_f[2,:]
    
    def mlm_vt(self, visual_embedded, text_embedded, label = None):
        aligned_visual = F.normalize(self.visual_transform(visual_embedded), dim=-1)
        aligned_text = F.normalize(self.text_transform(text_embedded), dim=-1)     
        f_f = torch.cat([aligned_visual, aligned_text], dim=0)
        f_f = self.block(f_f.unsqueeze(0)).squeeze(0) # len , 768
        f_f_f = self.cls(f_f)   # len, 30522
        # MLM
        if label is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            loss_mlm = loss_fct(f_f_f.view(-1, self.config.vocab_size), label.view(-1))
        else:
            loss_mlm = None
        return  loss_mlm, f_f[1,:]
    

