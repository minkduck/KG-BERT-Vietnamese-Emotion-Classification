import torch
import torch.nn as nn
from transformers import AutoModel


class ViSoBERT_Fusion(nn.Module):
    """
    ViSoBERT fusion network supporting baseline ('none'), raw ontology features,
    dense adapter integration, and dynamic text-to-ontology gating ('gate').
    """

    def __init__(
        self,
        n_classes=3,
        fusion_type="none",
        ontology_dim=None,
        ont_input_dim=24,
        model_name="uitnlp/visobert",
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.visobert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=0.3)

        if ontology_dim:
            self.ont_adapter = nn.Sequential(
                nn.Linear(ont_input_dim, ontology_dim),
                nn.BatchNorm1d(ontology_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            self.curr_dim = ontology_dim
        else:
            self.ont_adapter = nn.Identity()
            self.curr_dim = ont_input_dim

        if self.fusion_type == "gate":
            self.gate = nn.Linear(768, self.curr_dim)

        inp_dim = 768 + self.curr_dim if self.fusion_type != "none" else 768
        self.fc = nn.Linear(inp_dim, n_classes)

    def forward(self, input_ids, attention_mask, ontology_features):
        outputs = self.visobert(input_ids=input_ids, attention_mask=attention_mask)
        txt = outputs.last_hidden_state[:, 0, :]
        txt = self.dropout(txt)

        if self.fusion_type == "none":
            return self.fc(txt)

        ont_emb = self.ont_adapter(ontology_features)
        if self.fusion_type == "gate":
            g = torch.sigmoid(self.gate(txt))
            ont_emb = ont_emb * g

        comb = torch.cat([txt, ont_emb], dim=1)
        return self.fc(comb)


# TODO: ViSoBERT Deep Ontology architectural variants
# - ViSoBERT_DeepOntology: Used in VSMEC with BatchNorm1d in ontology projection pipeline (24 -> 64 -> 768).
# - ViSoBERT_DeepOntology_V2: Used in VSFC with LayerNorm in ontology projection pipeline (24 -> 64 -> 768).
# - ViSoBERT_Residual_Fusion: Used in VSFC-Ekman with deep 2-step projection (24 -> 256 -> 768) and residual gating.
# All three are preserved below to maintain exact experimental fidelity.


class ViSoBERT_DeepOntology(nn.Module):
    """
    Deep residual gated fusion architecture mapping ontology features to hidden dimension (VSMEC variant).
    Uses BatchNorm1d in the ontology projection pipeline.
    """

    def __init__(
        self,
        n_classes=7,
        fusion_type="residual_gate",
        ont_input_dim=24,
        hidden_dim=64,
        model_name="uitnlp/visobert",
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.visobert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=0.3)

        self.ont_pipeline = nn.Sequential(
            nn.Linear(ont_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        if self.fusion_type in ["gate", "residual_gate"]:
            self.gate_fc = nn.Linear(768 * 2, 768)

        final_dim = 768 * 2 if self.fusion_type == "concat" else 768
        self.fc = nn.Linear(final_dim, n_classes)

    def forward(self, input_ids, attention_mask, ontology_features):
        outputs = self.visobert(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = self.dropout(outputs.last_hidden_state[:, 0, :])
        ont_emb = self.ont_pipeline(ontology_features)

        if self.fusion_type == "concat":
            combined = torch.cat([text_emb, ont_emb], dim=1)
            return self.fc(combined)
        elif self.fusion_type == "residual_gate":
            concat_feat = torch.cat([text_emb, ont_emb], dim=1)
            z = torch.sigmoid(self.gate_fc(concat_feat))
            ont_info = self.dropout(z * ont_emb)
            fused_emb = text_emb + ont_info
            return self.fc(fused_emb)
        elif self.fusion_type == "gate":
            concat_feat = torch.cat([text_emb, ont_emb], dim=1)
            z = torch.sigmoid(self.gate_fc(concat_feat))
            fused_emb = (z * text_emb) + ((1 - z) * ont_emb)
            return self.fc(fused_emb)


class ViSoBERT_DeepOntology_V2(nn.Module):
    """
    Deep residual gated fusion architecture mapping ontology features to text embeddings (VSFC variant).
    Uses LayerNorm in the ontology projection pipeline.
    """

    def __init__(
        self,
        n_classes=3,
        fusion_type="residual_gate",
        ont_input_dim=24,
        hidden_dim=64,
        model_name="uitnlp/visobert",
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.visobert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=0.3)

        self.ont_pipeline = nn.Sequential(
            nn.Linear(ont_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        if self.fusion_type in ["gate", "residual_gate"]:
            self.gate_fc = nn.Linear(768 * 2, 768)

        final_input_dim = 768 * 2 if self.fusion_type == "concat" else 768
        self.fc = nn.Linear(final_input_dim, n_classes)

    def forward(self, input_ids, attention_mask, ontology_features):
        outputs = self.visobert(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = outputs.last_hidden_state[:, 0, :]
        text_emb = self.dropout(text_emb)

        if self.fusion_type == "none":
            return self.fc(text_emb)

        ont_emb = self.ont_pipeline(ontology_features)

        if self.fusion_type == "concat":
            combined = torch.cat([text_emb, ont_emb], dim=1)
            return self.fc(combined)
        elif self.fusion_type == "residual_gate":
            concat_feat = torch.cat([text_emb, ont_emb], dim=1)
            z = torch.sigmoid(self.gate_fc(concat_feat))
            ont_info = self.dropout(z * ont_emb)
            fused_emb = text_emb + ont_info
            return self.fc(fused_emb)
        elif self.fusion_type == "gate":
            concat_feat = torch.cat([text_emb, ont_emb], dim=1)
            z = torch.sigmoid(self.gate_fc(concat_feat))
            fused_emb = (z * text_emb) + ((1 - z) * ont_emb)
            return self.fc(fused_emb)


class ViSoBERT_Residual_Fusion(nn.Module):
    """
    Deep gated residual fusion between ViSoBERT representations and projected ontology features (VSFC-Ekman variant).
    Uses 2-step projection (24 -> 256 -> 768) and residual gating.
    """

    def __init__(
        self,
        n_classes=7,
        ont_input_dim=24,
        hidden_dim=768,
        dropout_p=0.3,
        model_name="uitnlp/visobert",
    ):
        super().__init__()
        self.visobert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=dropout_p)

        self.ont_proj = nn.Sequential(
            nn.Linear(ont_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, hidden_dim),
        )

        self.res_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid()
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, input_ids, attention_mask, ontology_features):
        outputs = self.visobert(input_ids=input_ids, attention_mask=attention_mask)
        h_txt = outputs.last_hidden_state[:, 0, :]
        h_txt = self.dropout(h_txt)

        h_ont = self.ont_proj(ontology_features)
        gate = self.res_gate(torch.cat([h_txt, h_ont], dim=1))

        h_fused = self.layer_norm(h_txt + (gate * h_ont))
        h_fused = self.dropout(h_fused)

        return self.fc(h_fused)


class OntologyConceptProjector(nn.Module):
    """
    Projects 24-dimensional continuous ontology features into 5 discrete conceptual tokens:
    [c_emo, c_app, c_int, c_pol, c_neg] in hidden_dim space.
    """

    def __init__(self, hidden_dim=768, dropout_p=0.1):
        super().__init__()
        self.proj_emo = nn.Sequential(
            nn.Linear(7, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()
        )
        self.proj_app = nn.Sequential(
            nn.Linear(9, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()
        )
        self.proj_int = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()
        )
        self.proj_pol = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()
        )
        self.proj_neg = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()
        )
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, ont_vector):
        v_emo = ont_vector[:, 0:7]
        v_app = ont_vector[:, 7:16]
        v_int = ont_vector[:, 16:19]
        v_pol = ont_vector[:, 19:22]
        v_neg = ont_vector[:, 22:24]

        c_emo = self.proj_emo(v_emo).unsqueeze(1)
        c_app = self.proj_app(v_app).unsqueeze(1)
        c_int = self.proj_int(v_int).unsqueeze(1)
        c_pol = self.proj_pol(v_pol).unsqueeze(1)
        c_neg = self.proj_neg(v_neg).unsqueeze(1)

        concepts = torch.cat([c_emo, c_app, c_int, c_pol, c_neg], dim=1)
        return self.dropout(concepts)


class ViSoBERT_Ontology_CrossAttention(nn.Module):
    """
    Neuro-symbolic architecture using token-level Multi-Head Cross-Attention over projected
    ontology concept tokens (OCA - Ontology-Guided Cross Attention).
    """

    def __init__(
        self,
        model_name="uitnlp/visobert",
        n_classes=3,
        n_heads=8,
        hidden_dim=768,
        dropout_p=0.3,
    ):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.ont_projector = OntologyConceptProjector(hidden_dim=hidden_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True,
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, input_ids, attention_mask, ontology_features):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        text_seq = outputs.last_hidden_state

        ont_concepts = self.ont_projector(ontology_features)

        attn_out, _ = self.cross_attention(
            query=text_seq, key=ont_concepts, value=ont_concepts
        )

        fused_seq = self.layer_norm(text_seq + self.dropout(attn_out))
        cls_rep = self.dropout(fused_seq[:, 0, :])

        return self.classifier(cls_rep)
