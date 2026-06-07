# Contrastive Framework

Framework badawczy do porównywania metod **samo-nadzorowanego uczenia kontrastowego** (SSL) z baseline'em `RandomInit` na trzech modalnościach: szeregach czasowych, danych tabelarycznych i obrazach medycznych.

Praca inżynierska – pełna pętla badawcza z HPO (Optuna), 10 seedami, stratyfikowanymi splitami, early stopping, LR schedulerem (warmup + cosine) i testem statystycznym Wilcoxona.

---

## Spis treści

1. [Architektura](#architektura)
2. [Instalacja](#instalacja)
3. [Szybki start (smoke test)](#szybki-start-smoke-test)
4. [Pełne eksperymenty](#pełne-eksperymenty)
5. [Datasety](#datasety)
6. [Wszystkie opcje CLI](#wszystkie-opcje-cli)
7. [Struktura wyjścia](#struktura-wyjścia)
8. [Modele i modalności](#modele-i-modalności)
9. [Diagnostyka problemów](#diagnostyka-problemów)
10. [Struktura projektu](#struktura-projektu)

---

## Architektura

| Modalność         | Dataset             | Baseline      | Algorytmy SSL          |
|-------------------|---------------------|---------------|------------------------|
| Szeregi czasowe   | UCI HAR             | `RandomInit`  | `TSTCC`                |
| Tabelaryczne      | Credit Card Fraud   | `RandomInit`  | `SCARF`, `SubTab`      |
| Obrazy medyczne   | PathMNIST           | `RandomInit`  | `MoCo`, `BYOL`         |

**Pipeline dla każdego algorytmu:**
1. **Pre-trening** samo-nadzorowany na zbiorze train (bez etykiet) – z warmup + cosine LR scheduler i gradient clipping (max_norm=1.0).
2. **Zamrożenie kodera** – enkoder pozostaje w trybie `eval()` przez cały fine-tuning (BN running stats się nie aktualizują).
3. **Liniowa ewaluacja** na kolejnych frakcjach etykiet: 6.25%, 12.5%, 25%, 50%, 100% – z early stopping na zbiorze val.
4. **Ewaluacja** na ukrytym zbiorze test – 4 metryki: Accuracy, Macro F1, Balanced Accuracy, Macro AUC-ROC.
5. Powtórz dla `N` seedów (domyślnie 10) → **mean ± std**.
6. **Test Wilcoxona** (jednostronny na macro_f1) vs `RandomInit` → istotność statystyczna.
7. **Tabela LaTeX** gotowa do wklejenia.

Opcjonalnie: **fine-tuning end-to-end** (`--run-finetune`) obok linear probe.

---

## Instalacja

Wymagane: Python 3.10+ (najlepiej 3.11).

```bash
# 1. Sklonuj/przejdź do katalogu projektu
cd framework

# 2. (Opcjonalnie) wirtualne środowisko
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate     # Linux/Mac

# 3. Zainstaluj zależności
pip install -r requirements.txt
```

`requirements.txt`:
```
torch>=2.0
torchvision>=0.15
optuna>=3.0
scipy>=1.10
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
kagglehub>=0.2
medmnist>=3.0
matplotlib>=3.7
```

**Uwaga – GPU (zalecane dla pełnych eksperymentów):** zainstaluj `torch` zgodny z Twoim CUDA z https://pytorch.org/get-started/locally/. Bez GPU pełne eksperymenty zajmą wiele godzin – używaj `--subsample`.

**Uwaga – Kaggle (tylko dla Credit Card Fraud):** przy pierwszym pobraniu `kagglehub` poprosi interaktywnie o token z https://www.kaggle.com/settings/account (sekcja "API"). Token jest zapamiętany.

**Uwaga – wizualizacje:** flaga `--visualize` wymaga `scikit-learn` (t-SNE) lub `umap-learn` (UMAP). Scikit-learn jest w `requirements.txt`, UMAP należy zainstalować osobno: `pip install umap-learn`.

---

## Szybki start (smoke test)

Każdy z trzech datasetów na CPU w <2 minuty:

```bash
python run_experiments.py --dataset uci_har --seeds 0 \
    --subsample 1000 --n-hpo-trials 2 \
    --pretrain-epochs 3 --finetune-epochs 3

python run_experiments.py --dataset credit_card --seeds 0 \
    --subsample 5000 --n-hpo-trials 2 \
    --pretrain-epochs 3 --finetune-epochs 3

python run_experiments.py --dataset pathmnist --seeds 0 \
    --subsample 2000 --n-hpo-trials 2 \
    --pretrain-epochs 3 --finetune-epochs 3
```

Po zakończeniu zobaczysz:
- konsolowy `summary` DataModule z rozkładem klas per frakcja,
- logi `[Pretrain]` z losses i learning rate,
- logi `[Finetune]` z val accuracy + early stopping,
- tabele wyników Accuracy, Macro F1 (i opcjonalnie Balanced Accuracy, Macro AUC-ROC),
- gotowy snippet LaTeX,
- plik `results_{dataset}.json` z surowymi wynikami i konfiguracją CLI.

---

## Pełne eksperymenty

**UCI HAR** – ~30 min na CPU, ~5 min na GPU:
```bash
python run_experiments.py --dataset uci_har
```

**Credit Card Fraud** – z 30k podsamplem stratyfikowanym (zachowuje wszystkie 492 fraudy):
```bash
python run_experiments.py --dataset credit_card --subsample 30000
```

**PathMNIST** – z 20k podsamplem stratyfikowanym (9 klas):
```bash
python run_experiments.py --dataset pathmnist --subsample 20000
```

**Pełen dataset bez subsamplingu** (wymaga GPU; CC Fraud = 284 807 wierszy, PathMNIST = 107 180 obrazów):
```bash
python run_experiments.py --dataset credit_card
python run_experiments.py --dataset pathmnist
```

**Z wizualizacją przestrzeni embeddingów (t-SNE):**
```bash
python run_experiments.py --dataset uci_har --seeds 0 --visualize --viz-dir figures/
```

**Z fine-tuningiem end-to-end obok linear probe:**
```bash
python run_experiments.py --dataset uci_har --run-finetune
```

---

## Datasety

| Dataset            | Źródło            | Rozmiar     | Shape (per próbka) | Klasy |
|--------------------|-------------------|-------------|---------------------|-------|
| UCI HAR            | UCI Archive       | ~10 300     | `(9, 128)`          | 6     |
| Credit Card Fraud  | Kaggle            | 284 807     | `(30,)`             | 2     |
| PathMNIST          | MedMNIST (Zenodo) | 107 180     | `(2352,)` (flatten) | 9     |

**Pierwsze uruchomienie automatycznie pobiera datasety do `data/`:**
- UCI HAR – ZIP ~62 MB z `archive.ics.uci.edu`
- Credit Card – CSV ~144 MB przez `kagglehub` (wymaga tokena Kaggle)
- PathMNIST – NPZ ~205 MB z Zenodo przez `medmnist`

Wszystkie loadery wspierają `--subsample N` (stratyfikowany po klasach). Subsample jest losowany niezależnie dla każdego seedu, żeby różne seedy widziały różne wycinki danych.

**Uwaga – PathMNIST:** framework scala oficjalne splity MedMNIST i tworzy własne podziały 70/15/15%. Wyniki nie są porównywalne z oryginalnym benchmarkiem MedMNIST.

---

## Wszystkie opcje CLI

```bash
python run_experiments.py --help
```

| Argument                  | Domyślnie           | Opis                                                            |
|---------------------------|---------------------|-----------------------------------------------------------------|
| `--dataset`               | **wymagany**        | `uci_har` \| `credit_card` \| `pathmnist`                       |
| `--data-dir`              | `data`              | Katalog z pobranymi datasetami                                  |
| `--subsample N`           | `None` (pełny)      | Stratyfikowany subsample N próbek (losowany per seed)           |
| `--seeds 0 1 2 ...`       | `0 1 2 ... 9`       | Lista seedów (domyślnie 10; min. 5 do testu Wilcoxona)          |
| `--n-hpo-trials`          | `20`                | Liczba prób Optuny per algorytm                                 |
| `--pretrain-epochs`       | `100`               | Epoki pre-treningu (finalny run)                                |
| `--finetune-epochs`       | `50`                | Epoki fine-tuningu (z early stopping)                           |
| `--hpo-pretrain-epochs`   | `20`                | Skrócone epoki pre-treningu podczas HPO                         |
| `--hpo-finetune-epochs`   | `15`                | Skrócone epoki fine-tuningu podczas HPO                         |
| `--batch-size`            | `256`               | Batch size dla wszystkich loaderów                              |
| `--splits-dir`            | `splits`            | Katalog z plikami `{dataset}_seed_{seed}.json`                  |
| `--results-file`          | `results_{ds}.json` | Plik z surowymi wynikami i konfiguracją CLI                     |
| `--visualize`             | `False`             | Generuj t-SNE/UMAP wizualizacje przestrzeni embeddingów         |
| `--viz-method`            | `tsne`              | `tsne` \| `umap`                                                |
| `--viz-dir`               | `figures`           | Katalog zapisu figur                                            |
| `--run-finetune`          | `False`             | Fine-tuning end-to-end obok linear probe                        |

---

## Struktura wyjścia

Po pełnym uruchomieniu:
```
framework/
├── data/                                # Pobrane datasety (auto-cache)
│   ├── uci_har/UCI HAR Dataset/...
│   └── medmnist/pathmnist.npz
├── splits/
│   ├── uci_har_seed_0.json              # Stratyfikowane indeksy train/val/test
│   ├── uci_har_seed_1.json
│   └── ...
├── figures/                             # Wizualizacje t-SNE/UMAP (tylko z --visualize)
│   └── embeddings_{dataset}_{algo}.png
├── results_uci_har.json                 # Surowe wyniki + konfiguracja CLI
├── results_credit_card.json
└── results_pathmnist.json
```

Tabela wyników w konsoli wygląda tak:
```
==================================================================
              Accuracy (mean ± std)
==================================================================
Algorytm       6.25%       12.5%       25.0%       50.0%      100.0%
------------------------------------------------------------------
RandomInit  0.612±0.028 0.681±0.020 0.741±0.015 0.789±0.011 0.823±0.008
TSTCC       0.728±0.022*0.789±0.017*0.831±0.012*0.866±0.009*0.889±0.006*
==================================================================
  * = istotnie lepszy od RandomInit (Wilcoxon jednostronny na macro_f1, p < 0.05)
```

Drukowane są tabele dla: Accuracy, Macro F1 (i odpowiadające im snippety LaTeX).

Plik JSON zawiera pełną konfigurację CLI (`"config": {...}`) i surowe wartości per seed – wystarczające do pełnej reprodukcji wyników.

---

## Modele i modalności

| Model       | Modalność       | Strategia kontrastowa                                                    |
|-------------|-----------------|--------------------------------------------------------------------------|
| `RandomInit`| dowolna         | Brak pre-treningu – baseline (random init kodera + linear readout)       |
| `TSTCC`     | szeregi         | Temporal + Contextual Contrasting (weak/strong aug + Transformer)        |
| `SCARF`     | tabelaryczne    | Random Feature Corruption (zamiana wartości kolumn na inne z batcha)     |
| `SubTab`    | tabelaryczne    | Subsetting kolumn + szum Gaussa, NT-Xent między podzbiorami              |
| `MoCo`      | obrazy          | Momentum contrast z kolejką negatywnych próbek (InfoNCE)                 |
| `BYOL`      | obrazy          | Bootstrap your own latent – bez negatywów, predyktor + target EMA, asymetryczne augmentacje |

Każdy algorytm SSL ma dedykowaną przestrzeń hiperparametrów w [contrastive_framework/hpo.py](contrastive_framework/hpo.py).

---

## Diagnostyka problemów

**`ModuleNotFoundError: No module named 'torch'`**
→ `pip install -r requirements.txt`. Dla CUDA przejdź na https://pytorch.org/get-started/locally/.

**Credit Card Fraud: "Could not find kaggle.json"**
→ Pierwsze uruchomienie `kagglehub` poprosi interaktywnie o token. Jeśli runuje w trybie nieinteraktywnym, umieść token z https://www.kaggle.com/settings/account w `~/.kaggle/kaggle.json`.

**`Zapisany podział dla seed=X ma N próbek, ale dataset ma M`**
→ Zmieniłeś `--subsample` po wygenerowaniu splitów. Usuń `splits/{dataset}_seed_*.json` i uruchom ponownie.

**Bardzo wolne uruchomienie na CPU (Credit Card / PathMNIST)**
→ Użyj `--subsample 5000` dla smoke testów, `--subsample 30000`/`20000` dla porządnych eksperymentów. Pełny dataset wymaga GPU.

**`UWAGA: za mało seedów (n) do testu Wilcoxona — potrzeba min. 5`**
→ Używasz za małej liczby seedów (np. `--seeds 0`). Test Wilcoxona wymaga min. 5 seedów – przy mniejszej liczbie gwiazdki istotności są pomijane.

**`RuntimeError: test_loader() dostępny dopiero po HPO`**
→ Wewnętrzny błąd – zgłoś jako bug. Normalnie nie powinien wystąpić.

**`--visualize` nie generuje figur / ImportError**
→ Sprawdź czy `scikit-learn` jest zainstalowany (`pip install scikit-learn`). Dla `--viz-method umap` potrzebny jest `pip install umap-learn`.

---

## Struktura projektu

```
framework/
├── contrastive_framework/
│   ├── __init__.py             # Publiczne API
│   ├── data_module.py          # DataModule + SplitManager + 3 fabryki per modalność
│   ├── datasets.py             # Loadery: UCI HAR, Credit Card, PathMNIST
│   ├── experiment.py           # pretrain / finetune_linear / finetune_end_to_end / run_experiment
│   │                           # + EarlyStopping + warmup_cosine scheduler + gradient clipping
│   ├── hpo.py                  # optimize() – Optuna HPO per algorytm (catch exceptions)
│   ├── baseline.py             # RandomInitModel
│   ├── imputation.py           # Imputacja brakujących danych (DAE, KNN, Mean)
│   ├── visualization.py        # plot_embeddings() – t-SNE / UMAP przestrzeni embeddingów
│   └── models/
│       ├── base.py             # BaseContrastiveModel + build_encoder (mlp/cnn/resnet_like)
│       │                       # + freeze_encoder() z encoder.eval() + train() override
│       ├── image_models.py     # MoCo (zawijanie kolejki bez asserta), BYOL (asymetryczne aug)
│       ├── tabular_models.py   # SCARF (brak self-corruption), SubTab (per-seed subsets)
│       └── timeseries_models.py # TS-TCC (dynamiczny max_seq_len z seq_len)
├── run_experiments.py              # Główny skrypt: HPO → trening → agregacja → Wilcoxon → LaTeX
├── requirements.txt
└── README.md
```

---
