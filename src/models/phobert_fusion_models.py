import torch
import torch.nn as nn
from transformers import AutoModel


class PhoBERT_Fusion_V2(nn.Module):
    """
    PhoBERTv2 Backbone supporting baseline ('none'), raw ontology ('raw'), dense projection adapter ('dense'),
    and dynamic gating ('gate').
    """

    def __init__(
        self,
        n_classes,
        fusion_type="none",
        ontology_dim=None,
        ont_input_dim=24,
        model_name="vinai/phobert-base-v2",
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.phobert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=0.3)

        # Ontology Adapter
        if ontology_dim:  # Dense Mode (e.g., 64d)
            self.ont_adapter = nn.Sequential(
                nn.Linear(ont_input_dim, ontology_dim),
                nn.BatchNorm1d(ontology_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            self.curr_dim = ontology_dim
        else:  # Raw Mode (xAI - 24d)
            self.ont_adapter = nn.Identity()
            self.curr_dim = ont_input_dim

        # Gating mechanism
        if self.fusion_type == "gate":
            self.gate = nn.Linear(768, self.curr_dim)

        inp_dim = 768 + self.curr_dim if self.fusion_type != "none" else 768
        self.fc = nn.Linear(inp_dim, n_classes)

    def forward(self, input_ids, attention_mask, ontology_features):
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )
        text_emb = outputs[1] if isinstance(outputs, tuple) else outputs.pooler_output
        text_emb = self.dropout(text_emb)

        if self.fusion_type == "none":
            return self.fc(text_emb)

        ont_emb = self.ont_adapter(ontology_features)
        if self.fusion_type == "gate":
            g = torch.sigmoid(self.gate(text_emb))
            ont_emb = ont_emb * g

        comb = torch.cat([text_emb, ont_emb], dim=1)
        return self.fc(comb)


class PhoBERT_DeepOntology(nn.Module):
    """
    Deep Ontology Projection network with multi-layer non-linear transformations:
    Input (ont_input_dim) -> Hidden (hidden_dim) -> 768 -> Fusion (Concat / Gate / Add).
    """

    def __init__(
        self,
        n_classes,
        fusion_type="concat",
        ont_input_dim=24,
        hidden_dim=64,
        model_name="vinai/phobert-base-v2",
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.phobert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=0.3)

        # Deep Projection Pipeline
        self.ont_pipeline = nn.Sequential(
            nn.Linear(ont_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        if self.fusion_type == "gate":
            self.gate_fc = nn.Linear(768 * 2, 768)

        final_input_dim = 768 * 2 if self.fusion_type == "concat" else 768
        self.fc = nn.Linear(final_input_dim, n_classes)

    def forward(self, input_ids, attention_mask, ontology_features):
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )
        text_emb = outputs[1] if isinstance(outputs, tuple) else outputs.pooler_output
        text_emb = self.dropout(text_emb)

        ont_emb = self.ont_pipeline(ontology_features)

        if self.fusion_type == "concat":
            combined = torch.cat([text_emb, ont_emb], dim=1)
            return self.fc(combined)
        elif self.fusion_type == "gate":
            concat_feat = torch.cat([text_emb, ont_emb], dim=1)
            z = torch.sigmoid(self.gate_fc(concat_feat))
            fused_emb = (z * text_emb) + ((1 - z) * ont_emb)
            return self.fc(fused_emb)
        else:
            return self.fc(text_emb + ont_emb)


class PhoBERT_WideProjection(nn.Module):
    """
    Direct Wide Projection from ontology features into 768-dim space (24 -> 768) with dynamic gating.
    """

    def __init__(
        self,
        n_classes,
        fusion_type="gate",
        ont_input_dim=24,
        model_name="vinai/phobert-base-v2",
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.phobert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=0.3)

        self.ont_pipeline = nn.Sequential(
            nn.Linear(ont_input_dim, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        if self.fusion_type == "gate":
            self.gate_fc = nn.Linear(768 * 2, 768)

        final_input_dim = 768 * 2 if self.fusion_type == "concat" else 768
        self.fc = nn.Linear(final_input_dim, n_classes)

    def forward(self, input_ids, attention_mask, ontology_features):
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )
        text_emb = outputs[1] if isinstance(outputs, tuple) else outputs.pooler_output
        text_emb = self.dropout(text_emb)

        ont_emb = self.ont_pipeline(ontology_features)

        if self.fusion_type == "concat":
            combined = torch.cat([text_emb, ont_emb], dim=1)
            return self.fc(combined)
        elif self.fusion_type == "gate":
            concat_feat = torch.cat([text_emb, ont_emb], dim=1)
            z = torch.sigmoid(self.gate_fc(concat_feat))
            fused_emb = (z * text_emb) + ((1 - z) * ont_emb)
            return self.fc(fused_emb)
        else:
            return self.fc(text_emb + ont_emb)
