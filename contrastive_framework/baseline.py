"""
baseline.py – Baseline "Random Init" do porównań w pracy naukowej.

RandomInitModel ma identyczną architekturę kodera co pozostałe modele,
ale pretrain_step() jest celowo pustą operacją (zwraca loss=0, bez backward).
Dzięki temu mierzy dokładnie ile daje samo uczenie kontrastowe — różnica
wyników względem RandomInit to "zysk z pre-treningu".
"""

from typing import Any, Dict

import torch

from .models.base import BaseContrastiveModel


class RandomInitModel(BaseContrastiveModel):
    """
    Baseline bez pre-treningu.

    Koder jest inicjalizowany losowo i nie jest aktualizowany w fazie pre-treningu.
    Użycie w pętli eksperymentalnej jest identyczne jak SSL — uczciwe porównanie.
    """

    def pretrain_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        """Zwraca loss=0 bez requires_grad — pretrain loop pomija backward."""
        device = next(self.parameters()).device
        return {"loss": torch.tensor(0.0, device=device)}
