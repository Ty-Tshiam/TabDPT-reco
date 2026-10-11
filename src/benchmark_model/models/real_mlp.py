"""
RealMLP: Modernized Tabular Neural Network Baseline.

Implements a high-capacity, regularized Tabular MLP with:
- Linear input projection with LayerNorm
- Residual MLP blocks with Mish non-linearity and Dropout
- Cosine learning rate scheduling and AdamW optimization
- Standard BaseRecommender API (fit, predict_proba, recommend)
"""

import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.benchmark_model.base import BaseRecommender


class ResidualMLPBlock(nn.Module):
    """Residual Block for tabular representations."""

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.act = nn.Mish()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return residual + out


class RealMLPArchitecture(nn.Module):
    """Deep residual tabular neural architecture."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        num_classes: int = 16,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
        )
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        for block in self.blocks:
            out = block(out)
        out = self.final_norm(out)
        return self.head(out)


class RealMLPRecommender(BaseRecommender):
    """RealMLP Tabular Recommender wrapper."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        epochs: int = 15,
        batch_size: int = 512,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str | None = None,
    ):
        super().__init__(name="RealMLP", family="Neural Tabular")
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: RealMLPArchitecture | None = None
        self.input_dim: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RealMLPRecommender":
        start_time = time.time()
        self.input_dim = X.shape[1]

        self.model = RealMLPArchitecture(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks,
            num_classes=16,
        ).to(self.device)

        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )

        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            scheduler.step()

        self.training_time_sec = round(time.time() - start_time, 2)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("RealMLP must be fitted before predict_proba.")

        self.model.eval()
        x_tensor = torch.tensor(X, dtype=torch.float32)

        # Batch evaluation to manage memory
        all_probs = []
        batch_size = 1024
        with torch.no_grad():
            for i in range(0, len(x_tensor), batch_size):
                batch = x_tensor[i : i + batch_size].to(self.device)
                logits = self.model(batch)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)
