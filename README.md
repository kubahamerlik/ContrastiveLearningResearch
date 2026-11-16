# Contrastive Framework

Framework badawczy do porównywania metod **samo-nadzorowanego uczenia kontrastowego** (SSL) z baseline'em `RandomInit` na trzech modalnościach: szeregach czasowych, danych tabelarycznych i obrazach medycznych.

Praca inżynierska – pełna pętla badawcza z HPO (Optuna), 10 seedami, stratyfikowanymi splitami, early stopping, LR schedulerem (warmup + cosine) oraz statystyką 4 metryk (mean ± std ± 95% CI, różnice sparowane po seedzie).

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
9. [Zgodność z oryginalnymi implementacjami](#zgodność-z-oryginalnymi-implementacjami)
10. [Kaggle, 2×T4](#kaggle-2t4)
11. [Diagnostyka problemów](#diagnostyka-problemów)
12. [Struktura projektu](#struktura-projektu)

---

## Architektura

| Modalność         | Dataset             | Baseline(y)                               | Algorytmy SSL          |
|-------------------|---------------------|-------------------------------------------|------------------------|
| Szeregi czasowe   | UCI HAR             | `RandomInit` (arch. CNN = TS-TCC)         | `TSTCC`                |
| Tabelaryczne      | Credit Card Fraud   | `RandomInit`, `RandomInit-SubTabArch`     | `SCARF`, `SubTab`      |
| Obrazy medyczne   | PathMNIST           | `RandomInit`                              | `MoCo`, `BYOL`         |

**Pipeline dla każdego algorytmu:**
1. **Pre-trening** samo-nadzorowany na zbiorze train (bez etykiet) – z warmup + cosine LR scheduler i gradient clipping (max_norm=1.0).
2. **Zamrożenie kodera** – enkoder pozostaje w trybie `eval()` przez cały fine-tuning (BN running stats się nie aktualizują).
3. **Liniowa ewaluacja** na kolejnych frakcjach etykiet: 6.25%, 12.5%, 25%, 50%, 100% – z early stopping na zbiorze val.
4. **Ewaluacja** na ukrytym zbiorze test – 4 metryki: Accuracy, Macro F1, Balanced Accuracy, Macro AUC-ROC.
5. Powtórz dla `N` seedów (domyślnie 10) → **mean ± std ± 95% CI t-Studenta**.
6. **Różnice sparowane** (SSL − baseline, parowane po seedzie) dla każdej metryki: mean\_diff, std, 95% CI, wins/N. Opcjonalnie: test Wilcoxona (`--wilcoxon`).
7. **Tabela LaTeX** i wykresy label-efficiency gotowe do wklejenia (`scripts/summarize_results.py`).

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
| `--algorithms NAME ...`   | wszystkie dla datasetu | Uruchom tylko podane algorytmy, np. `--algorithms MoCo BYOL` |
| `--device`                | auto (cuda/cpu)     | Jawna karta GPU, np. `cuda:0`; nadpisywane przez `CUDA_VISIBLE_DEVICES` |
| `--prepare-only`          | `False`             | Wygeneruj splity i zakończ (bez treningu) — do przygotowania danych przed `--resume` |
| `--resume`                | `False`             | Pomiń seedy już zapisane w pliku wyników (bezpieczne wznawianie po przerwaniu) |
| `--correction`            | `holm`              | Korekcja wielokrotnych porównań Wilcoxona: `holm` \| `bh`       |
| `--checkpoints-dir`       | `None`              | Katalog do zapisywania checkpointów modeli po pre-treningu      |
| `--monitor-metrics`       | `False`             | Loguj dodatkowe metryki diagnostyczne w trakcie treningu         |

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

| Model                    | Modalność       | Strategia kontrastowa                                                                   |
|--------------------------|-----------------|-----------------------------------------------------------------------------------------|
| `RandomInit`             | dowolna         | Brak pre-treningu – baseline (random init kodera + linear readout)                      |
| `RandomInit-SubTabArch`  | tabelaryczne    | Jak `RandomInit`, ale z **identyczną architekturą AE** co SubTab – porównanie arch-to-arch |
| `RandomInit` (HAR)       | szeregi czasowe | Jak `RandomInit`, ale z **identyczną architekturą CNN** co TS-TCC                       |
| `TSTCC`                  | szeregi         | Temporal + Contextual Contrasting (weak/strong aug + Transformer)                       |
| `SCARF`                  | tabelaryczne    | Random Feature Corruption – dokładnie ⌊c·M⌋ cech na wiersz, wartości z rozkładu brzegowego X_train |
| `SubTab`                 | tabelaryczne    | Subsetting kolumn + szum, NT-Xent między podzbiorami (jeden wspólny enkoder AE)         |
| `MoCo`                   | obrazy          | Momentum contrast v2 – kolejka negatywów, głowica MLP bez BN, temperatura 0.2          |
| `BYOL`                   | obrazy          | Bootstrap your own latent – bez negatywów, predyktor + target EMA z cosine τ schedule  |

**Flaga `--algorithms`** pozwala uruchomić tylko wybrane algorytmy zamiast całego zestawu:

```bash
# Uruchom tylko MoCo i BYOL na PathMNIST (z pominięciem RandomInit)
python run_experiments.py --dataset pathmnist --algorithms MoCo BYOL

# Uruchom tylko SCARF na Credit Card
python run_experiments.py --dataset credit_card --algorithms SCARF

# Sprawdź dostępne algorytmy per dataset — jeśli podasz błędną nazwę, skrypt wypisze dostępne
python run_experiments.py --dataset uci_har --algorithms foo
```

Nazwy algorytmów muszą dokładnie pasować do identyfikatorów wewnętrznych:
`RandomInit`, `RandomInit-SubTabArch`, `TSTCC`, `SCARF`, `SubTab`, `MoCo`, `BYOL`.

Każdy algorytm SSL ma dedykowaną przestrzeń hiperparametrów w [contrastive_framework/hpo.py](contrastive_framework/hpo.py).

---

## Zgodność z oryginalnymi implementacjami

Każda metoda SSL jest zaimplementowana z zachowaniem wierności oryginalnej publikacji lub kodowi
referencyjnemu. Poniżej zestawienie repozytoriów użytych jako referencja:

| Metoda | Repozytorium referencyjne | Licencja | Gałąź/commit |
|--------|--------------------------|----------|--------------|
| MoCo v2 | [facebookresearch/moco](https://github.com/facebookresearch/moco) | CC BY-NC 4.0 | `KaimingHe-patch-2`, `moco/builder.py` |
| BYOL | [deepmind/deepmind-research/byol](https://github.com/deepmind/deepmind-research/tree/master/byol) | Apache 2.0 | `master`, `byol_experiment.py`, `utils/` |
| SCARF | arXiv:2106.15147 (Bahri et al., ICLR 2022) | — (brak oficjalnego kodu) | implementacja z artykułu |
| SubTab | [AstraZeneca/SubTab](https://github.com/AstraZeneca/SubTab) | Apache 2.0 | `main`, `utils/model_utils.py`, `src/model.py` |
| TS-TCC | [emadeldeen24/TS-TCC](https://github.com/emadeldeen24/TS-TCC) | MIT | `master`, `models/`, `dataloader/` |

Repozytoria sklonowane lokalnie do `third_party_reference/` (w `.gitignore`).

Pełna lista świadomych odstępstw od protokołów oryginalnych (optymalizator, architektura,
augmentacje, strata, sprzęt) znajduje się w **[docs/DEVIATIONS.md](docs/DEVIATIONS.md)**.

Licencje kodu zewnętrznego: **[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)**.

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

## Kaggle, 2×T4

Kaggle udostępnia dwie karty T4 16 GB w trybie „GPU T4 x2" (limit 12 h/sesję).
Modele są małe — zamiast DDP uruchamiamy **dwa niezależne procesy**, po jednym na kartę,
z rozdzielonymi seedami.

### Wybór urządzenia

Dwa równoważne sposoby:

```bash
# Sposób 1 – flaga --device (jawna karta)
python run_experiments.py --dataset pathmnist --device cuda:0 ...
python run_experiments.py --dataset pathmnist --device cuda:1 ...

# Sposób 2 – CUDA_VISIBLE_DEVICES (zalecany na Kaggle; pomija --device)
CUDA_VISIBLE_DEVICES=0 python run_experiments.py --dataset pathmnist ...
CUDA_VISIBLE_DEVICES=1 python run_experiments.py --dataset pathmnist ...
```

`CUDA_VISIBLE_DEVICES=0` sprawia że PyTorch widzi tylko pierwszą kartę jako `cuda:0`
— prostsze w notebookach Kaggle gdzie `%%bash` i `subprocess` dobrze obsługują zmienne
środowiskowe, a `--device` wymagałoby modyfikacji komendy.

### Gotowa komórka notebooka Kaggle

```bash
%%bash
# Krok 0: pobierz dane i wygeneruj splity dla wszystkich seedów (raz, sekwencyjnie)
python run_experiments.py \
    --dataset pathmnist --subsample 20000 \
    --prepare-only --seeds 0 1 2 3 4 5 6 7 8 9

# Krok 1: uruchom dwa procesy równolegle, każdy na swojej karcie
CUDA_VISIBLE_DEVICES=0 python run_experiments.py \
    --dataset pathmnist --subsample 20000 \
    --seeds 0 2 4 6 8 \
    --results-file results_pathmnist_gpu0.json \
    --resume > log_gpu0.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python run_experiments.py \
    --dataset pathmnist --subsample 20000 \
    --seeds 1 3 5 7 9 \
    --results-file results_pathmnist_gpu1.json \
    --resume > log_gpu1.txt 2>&1 &

wait  # czekaj aż oba procesy skończą

# Krok 2: scal wyniki w jeden plik i wypisz tabele
python scripts/merge_results.py \
    results_pathmnist_gpu0.json results_pathmnist_gpu1.json \
    --out results_pathmnist.json
```

Analogicznie dla pozostałych datasetów (zamień `pathmnist` → `uci_har` / `credit_card`
i dostosuj `--subsample`).

### Podgląd postępu

```bash
# W osobnej komórce Kaggle lub terminalu:
tail -f log_gpu0.txt log_gpu1.txt

# Użycie GPU:
nvidia-smi -l 5    # odświeżaj co 5 s
```

### Wznawianie po przerwaniu (--resume)

Flaga `--resume` pomija seedy już zapisane w pliku wyników — bezpieczne przy
wznowieniu po przekroczeniu limitu 12 h lub awarii:

```bash
# Uruchom ponownie — automatycznie pominie ukończone seedy
CUDA_VISIBLE_DEVICES=0 python run_experiments.py \
    --dataset pathmnist --subsample 20000 \
    --seeds 0 2 4 6 8 \
    --results-file results_pathmnist_gpu0.json \
    --resume
```

Wyniki są zapisywane **po każdym seedzie** (checkpoint), więc wznowienie traci
co najwyżej jeden seed pracy.

### Spójność wyników między procesami

Każdy seed jest seedowany niezależnie (`torch`, `numpy`, `random`, `optuna`)
przed wywołaniem `run_one_seed`. Wynik dla seeda N nie zależy od tego,
na której karcie ani w jakiej kolejności był liczony.

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
│       ├── image_models.py     # MoCo v2 (SplitBN, MLP bez BN, τ=0.2), BYOL (cosine τ, train() target)
│       ├── tabular_models.py   # SCARF (floor(c·M) korupcja, marginale X_train, InfoNCE τ=1), SubTab+RandomInit-SubTabArch
│       └── timeseries_models.py # TS-TCC (dynamiczny max_seq_len z seq_len)
├── run_experiments.py              # Główny skrypt: HPO → trening → agregacja → Wilcoxon → LaTeX
├── scripts/
│   └── merge_results.py            # Scala wyniki z wielu procesów/kart GPU w jeden plik
├── tests/                          # Testy pytest (CPU, kilka sekund)
├── docs/
│   └── DEVIATIONS.md               # Tabela odstępstw od oryginalnych implementacji
├── THIRD_PARTY_LICENSES.md         # Licencje kodu zewnętrznego
├── requirements.txt
└── README.md
```

---
