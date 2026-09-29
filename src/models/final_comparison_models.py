import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

ONT_INPUT_DIM = 24


class ViSoBERT_Baseline(nn.Module):
    """
    ViSoBERT Backbone with standard classification head.
    """

    def __init__(self, n_classes=3, model_name="uitnlp/visobert"):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(768, n_classes)

    def forward(self, input_ids, attention_mask, ontology_features=None):
        outputs = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        )
        cls_rep = self.dropout(outputs.last_hidden_state[:, 0, :])
        return self.classifier(cls_rep)


class ALDONAr_Model(nn.Module):
    """
    ALDONAr: Dual-Gated Attentive Fusion between contextual text representation and projected ontology features.
    """

    def __init__(
        self,
        n_classes=3,
        ont_dim=ONT_INPUT_DIM,
        hidden_dim=768,
        model_name="uitnlp/visobert",
    ):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=0.3)

        self.ont_proj = nn.Sequential(
            nn.Linear(ont_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.attn_weight = nn.Linear(hidden_dim * 2, 2)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, input_ids, attention_mask, ontology_features):
        out = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        )
        h_text = self.dropout(out.last_hidden_state[:, 0, :])
        h_ont = self.ont_proj(ontology_features.float())

        combined = torch.cat([h_text, h_ont], dim=-1)
        attn_scores = F.softmax(self.attn_weight(combined), dim=-1)
        w_t = attn_scores[:, 0:1]
        w_o = attn_scores[:, 1:2]

        fused = w_t * h_text + w_o * h_ont
        return self.classifier(fused)


class KEAHT_Model(nn.Module):
    """
    KEAHT: Knowledge-Enriched Attention Hybrid Transformer using token-level multi-head cross-attention.
    """

    def __init__(
        self,
        n_classes=3,
        ont_dim=ONT_INPUT_DIM,
        hidden_dim=768,
        n_heads=8,
        model_name="uitnlp/visobert",
    ):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=0.3)

        self.proj_emo = nn.Linear(7, hidden_dim)
        self.proj_app = nn.Linear(9, hidden_dim)
        self.proj_pol = nn.Linear(3, hidden_dim)
        self.proj_neg = nn.Linear(5, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, input_ids, attention_mask, ontology_features):
        ont = ontology_features.float()
        text_seq = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        c1 = self.proj_emo(ont[:, 0:7]).unsqueeze(1)
        c2 = self.proj_app(ont[:, 7:16]).unsqueeze(1)
        c3 = self.proj_pol(ont[:, 19:22]).unsqueeze(1)
        c4 = self.proj_neg(
            torch.cat([ont[:, 16:19], ont[:, 22:24]], dim=1)
        ).unsqueeze(1)
        k_concepts = torch.cat([c1, c2, c3, c4], dim=1)

        attn_out, _ = self.cross_attn(
            query=text_seq, key=k_concepts, value=k_concepts
        )
        fused_seq = self.norm(text_seq + self.dropout(attn_out))
        cls_rep = self.dropout(fused_seq[:, 0, :])
        return self.classifier(cls_rep)


class CombViSA_Model(nn.Module):
    """
    CombViSA (Proposed Model): Multi-branch architecture combining Transformer contextual representations,
    Bidirectional LSTM sequential modeling, and non-linear Ontology representations.
    """

    def __init__(
        self,
        n_classes=3,
        ont_dim=ONT_INPUT_DIM,
        hidden_dim=768,
        lstm_hidden=256,
        model_name="uitnlp/visobert",
    ):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=0.3)

        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.ont_branch = nn.Sequential(
            nn.Linear(ont_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        fused_dim = (lstm_hidden * 2) + 128
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )

    def forward(self, input_ids, attention_mask, ontology_features):
        seq = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        lstm_out, _ = self.bilstm(seq)
        lstm_feat, _ = torch.max(lstm_out, dim=1)
        lstm_feat = self.dropout(lstm_feat)

        ont_feat = self.ont_branch(ontology_features.float())
        combined = torch.cat([lstm_feat, ont_feat], dim=1)
        return self.classifier(combined)
