from pathlib import Path
import torch
import torch.nn as nn
from torchcrf import CRF

class BiLSTMCRFNER(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embedding_dim: int,
        hidden_dim: int,
        pad_token_id: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_token_id,
        )

        self.dropout = nn.Dropout(dropout)

        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Linear(hidden_dim, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | list[list[int]]:
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        lstm_out, _ = self.bilstm(embeddings)
        lstm_out = self.dropout(lstm_out)

        emissions = self.classifier(lstm_out)

        mask = attention_mask.bool()

        if labels is not None:
            loss = -self.crf(emissions, labels, mask=mask, reduction="mean")
            return loss

        decoded = self.crf.decode(emissions, mask=mask)
        return decoded

    def save_checkpoint(
        self,
        path: Path,
        token_vocab: dict[str, int],
        label_to_id: dict[str, int],
        id_to_label: dict[int, str],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "token_vocab": token_vocab,
                "label_to_id": label_to_id,
                "id_to_label": id_to_label,
            },
            path,
        )