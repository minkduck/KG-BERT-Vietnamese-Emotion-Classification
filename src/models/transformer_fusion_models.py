"""Language-agnostic Transformer–ontology fusion models for the IJMLC extension v2.

The architecture is intentionally unchanged with respect to the manuscript RawGate idea.
The v2 package adds richer auxiliary outputs so that experiments can quantify the role
of the ontology branch rather than merely compare end-point classification metrics.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class TransformerOntologyFusion(nn.Module):
    """Generic Transformer + ontology fusion network.

    Supported fusion modes
    ----------------------
    none
        Text-only baseline.
    rawconcat
        Concatenate the raw ontology vector to the text representation.
    rawgate
        Compute ``g=sigmoid(W h + b)`` and concatenate ``[h; g * o]``.
    denseconcat
        Project the ontology vector to a dense space before concatenation.
    densegate
        Project the ontology vector, gate it from the text representation, then
        concatenate the gated ontology representation with the text vector.
    flatconcat
        Alias of raw concatenation used by the 7-d NRC-flat control.
    """

    VALID_MODES = {
        "none",
        "rawconcat",
        "rawgate",
        "denseconcat",
        "densegate",
        "flatconcat",
    }

    def __init__(
        self,
        model_name: str,
        n_classes: int,
        fusion_mode: str = "none",
        ont_input_dim: int = 24,
        dense_dim: int = 64,
        text_dropout: float = 0.3,
        ontology_dropout: float = 0.1,
    ):
        super().__init__()
        self.model_name = model_name
        self.fusion_mode = str(fusion_mode).lower()
        if self.fusion_mode not in self.VALID_MODES:
            raise ValueError(
                f"Unsupported fusion_mode={fusion_mode}; "
                f"expected one of {sorted(self.VALID_MODES)}"
            )

        config = AutoConfig.from_pretrained(model_name)
        self.hidden_size = int(config.hidden_size)
        self.backbone = AutoModel.from_pretrained(model_name, config=config)
        self.text_dropout = nn.Dropout(float(text_dropout))

        use_dense = self.fusion_mode in {"denseconcat", "densegate"}
        if use_dense:
            self.ont_adapter = nn.Sequential(
                nn.Linear(int(ont_input_dim), int(dense_dim)),
                nn.BatchNorm1d(int(dense_dim)),
                nn.ReLU(),
                nn.Dropout(float(ontology_dropout)),
            )
            self.ont_dim = int(dense_dim)
        else:
            self.ont_adapter = nn.Identity()
            self.ont_dim = int(ont_input_dim)

        gated = self.fusion_mode in {"rawgate", "densegate"}
        self.gate = (
            nn.Linear(self.hidden_size, self.ont_dim)
            if gated
            else None
        )

        classifier_in = (
            self.hidden_size
            if self.fusion_mode == "none"
            else self.hidden_size + self.ont_dim
        )
        self.classifier = nn.Linear(classifier_in, int(n_classes))

    def encode_text(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        if hasattr(outputs, "last_hidden_state"):
            h = outputs.last_hidden_state[:, 0, :]
        elif (
            hasattr(outputs, "pooler_output")
            and outputs.pooler_output is not None
        ):
            h = outputs.pooler_output
        else:
            h = outputs[0][:, 0, :]
        return self.text_dropout(h)

    def forward(
        self,
        input_ids,
        attention_mask,
        ontology_features: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        h = self.encode_text(input_ids, attention_mask)

        if self.fusion_mode == "none":
            logits = self.classifier(h)
            if return_aux:
                return logits, {
                    "gate": None,
                    "ontology_input": None,
                    "ontology_pre_gate": None,
                    "ontology_representation": None,
                }
            return logits

        if ontology_features is None:
            raise ValueError(
                "ontology_features must be supplied for non-baseline fusion modes"
            )

        ontology_input = ontology_features
        ontology_pre_gate = self.ont_adapter(ontology_features)
        ontology_representation = ontology_pre_gate
        gate_values = None

        if self.gate is not None:
            gate_values = torch.sigmoid(self.gate(h))
            ontology_representation = ontology_pre_gate * gate_values

        fused = torch.cat([h, ontology_representation], dim=1)
        logits = self.classifier(fused)

        if return_aux:
            return logits, {
                "gate": gate_values,
                "ontology_input": ontology_input,
                "ontology_pre_gate": ontology_pre_gate,
                "ontology_representation": ontology_representation,
            }
        return logits
