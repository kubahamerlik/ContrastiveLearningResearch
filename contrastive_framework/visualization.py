"""
visualization.py – t-SNE i UMAP wizualizacja przestrzeni embeddingów.

Standard w artykułach SSL: figura "embedding space przed i po pre-treningu"
pokazuje że reprezentacje SSL grupują klasy lepiej niż losowa inicjalizacja.

Użycie:
    from contrastive_framework.visualization import plot_embeddings

    fig = plot_embeddings(
        model_before, model_after, dataloader, device,
        title="SCARF embeddings – Credit Card Fraud",
        save_path="embeddings_scarf.png",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int = 5000,
):
    """Przepuszcza dane przez model i zbiera embeddingi + etykiety.

    Returns
    -------
    embeddings : (N, D) numpy array
    labels     : (N,) numpy array
    """
    model.eval()
    all_emb = []
    all_lbl = []
    n = 0

    with torch.no_grad():
        for batch in loader:
            if n >= max_samples:
                break
            x = batch[0].to(device)
            y = batch[1].cpu().numpy() if len(batch) > 1 else None

            # forward() zwraca embeddingi (przed klasyfikatorem)
            h = model(x).cpu().numpy()
            all_emb.append(h)
            if y is not None:
                all_lbl.append(y)
            n += h.shape[0]

    import numpy as np
    embeddings = np.concatenate(all_emb, axis=0)[:max_samples]
    labels = np.concatenate(all_lbl, axis=0)[:max_samples] if all_lbl else None
    return embeddings, labels


def plot_embeddings(
    model_before: nn.Module,
    model_after: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    method: str = "tsne",
    title: str = "Embedding space",
    save_path: Optional[str | Path] = None,
    max_samples: int = 3000,
    class_names: Optional[list] = None,
    perplexity: int = 30,
    random_state: int = 42,
):
    """
    Rysuje dwa wykresy: embedding space przed i po pre-treningu.

    Parameters
    ----------
    model_before : model przed pre-treningiem (np. RandomInitModel lub model po reset)
    model_after  : model po pre-treningu SSL
    loader       : DataLoader ze zbiorem (train lub test) — bez augmentacji
    device       : urządzenie
    method       : "tsne" (domyślnie) lub "umap" (wymaga pakietu umap-learn)
    title        : tytuł figury
    save_path    : jeśli podany, zapisuje figurę do pliku PNG/PDF
    max_samples  : maksymalna liczba próbek (t-SNE jest wolne dla N>5000)
    class_names  : lista nazw klas (do legendy); None = etykiety numeryczne
    perplexity   : perplexity dla t-SNE (ignorowane przy UMAP)
    random_state : seed dla powtarzalności

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "plot_embeddings wymaga matplotlib. Zainstaluj: pip install matplotlib"
        ) from e

    print(f"[Visualization] Ekstrakcja embeddingów (max {max_samples} próbek)...")
    emb_before, labels = _extract_embeddings(model_before, loader, device, max_samples)
    emb_after,  _      = _extract_embeddings(model_after,  loader, device, max_samples)

    print(f"[Visualization] Redukcja wymiarowości ({method.upper()})...")
    coords_before = _reduce(emb_before, method, perplexity, random_state)
    coords_after  = _reduce(emb_after,  method, perplexity, random_state)

    # Kolorowanie
    unique_labels = sorted(set(labels.tolist())) if labels is not None else [0]
    n_classes = len(unique_labels)
    cmap = plt.cm.get_cmap("tab10" if n_classes <= 10 else "tab20", n_classes)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, coords, subtitle in [
        (axes[0], coords_before, "Przed pre-treningiem (Random Init)"),
        (axes[1], coords_after,  f"Po pre-treningu SSL ({method.upper()})"),
    ]:
        if labels is not None:
            for i, c in enumerate(unique_labels):
                mask = labels == c
                lbl = class_names[c] if class_names and c < len(class_names) else str(c)
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=[cmap(i)], s=6, alpha=0.6, label=lbl, rasterized=True,
                )
            ax.legend(markerscale=3, fontsize=7, loc="best",
                      title="Klasy", title_fontsize=8)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], s=6, alpha=0.5)
        ax.set_title(subtitle, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Visualization] Zapisano figurę → {save_path}")

    return fig


def _reduce(
    embeddings,
    method: str,
    perplexity: int,
    random_state: int,
):
    """Redukuje wymiarowość do 2D przez t-SNE lub UMAP."""
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ImportError as e:
            raise ImportError(
                "t-SNE wymaga scikit-learn. Zainstaluj: pip install scikit-learn"
            ) from e
        reducer = TSNE(
            n_components=2,
            perplexity=min(perplexity, embeddings.shape[0] - 1),
            random_state=random_state,
            n_iter=1000,
            init="pca",
            learning_rate="auto",
        )
        return reducer.fit_transform(embeddings)

    elif method == "umap":
        try:
            import umap
        except ImportError as e:
            raise ImportError(
                "UMAP wymaga pakietu umap-learn. Zainstaluj: pip install umap-learn"
            ) from e
        reducer = umap.UMAP(
            n_components=2,
            random_state=random_state,
            n_neighbors=15,
            min_dist=0.1,
        )
        return reducer.fit_transform(embeddings)

    else:
        raise ValueError(f"Nieznana metoda redukcji: '{method}'. Wybierz 'tsne' lub 'umap'.")
