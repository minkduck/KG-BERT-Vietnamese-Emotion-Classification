import torch
import torch.nn as nn
from transformers import AutoModel

VNEMOLEX_DIM = 7
ONTOLOGY_DIM = 24


class Model_Baseline(nn.Module):
    """
    Standard Transformer Backbone (PhoBERT or ViSoBERT) with Classification Head.
    """

    def __init__(self, n_classes, model_name="uitnlp/visobert"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(768, n_classes)

    def forward(self, input_ids, attention_mask, extra_feat=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "last_hidden_state"):
            cls_rep = outputs.last_hidden_state[:, 0, :]
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            cls_rep = outputs.pooler_output
        else:
            cls_rep = outputs[0][:, 0, :]
        cls_rep = self.dropout(cls_rep)
        return self.fc(cls_rep)


class Model_VnEmoLex_Concat(nn.Module):
    """
    Transformer Backbone concatenated with raw VnEmoLex lexicon feature vector (7-dimensional).
    """

    def __init__(self, n_classes, model_name="uitnlp/visobert", lex_dim=VNEMOLEX_DIM):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(768 + lex_dim, n_classes)

    def forward(self, input_ids, attention_mask, lex_vector):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "last_hidden_state"):
            cls_rep = outputs.last_hidden_state[:, 0, :]
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            cls_rep = outputs.pooler_output
        else:
            cls_rep = outputs[0][:, 0, :]
        cls_rep = self.dropout(cls_rep)
        comb = torch.cat([cls_rep, lex_vector], dim=1)
        return self.fc(comb)


class Model_Ontology_RawGate(nn.Module):
    """
    Transformer Backbone integrated with full 24-dimensional Ontology vector via Dynamic Text Gating.
    """

    def __init__(self, n_classes, model_name="uitnlp/visobert", ont_dim=ONTOLOGY_DIM):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=0.3)
        self.gate = nn.Linear(768, ont_dim)
        self.fc = nn.Linear(768 + ont_dim, n_classes)

    def forward(self, input_ids, attention_mask, ont_vector):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "last_hidden_state"):
            cls_rep = outputs.last_hidden_state[:, 0, :]
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            cls_rep = outputs.pooler_output
        else:
            cls_rep = outputs[0][:, 0, :]
        cls_rep = self.dropout(cls_rep)

        g = torch.sigmoid(self.gate(cls_rep))
        gated_ont = ont_vector * g

        comb = torch.cat([cls_rep, gated_ont], dim=1)
        return self.fc(comb)
