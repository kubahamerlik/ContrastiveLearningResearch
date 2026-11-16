# Odstępstwa od oryginalnych protokołów

Dokument rejestruje **świadome różnice** między naszą implementacją a oryginalnym
kodem źródłowym lub artykułem każdej metody. Przeznaczony do podrozdziału
„Odstępstwa od protokołu" w pracy inżynierskiej — każdy wpis zawiera: co robi oryginał,
co robimy my i dlaczego.

---

## Spis treści

1. [Wspólne dla wszystkich metod](#1-wspólne-dla-wszystkich-metod)
2. [MoCo v2](#2-moco-v2)
3. [BYOL](#3-byol)
4. [SCARF](#4-scarf)
5. [SubTab](#5-subtab)
6. [TS-TCC](#6-ts-tcc)

---

## 1. Wspólne dla wszystkich metod

| Element | Oryginały | U nas | Powód |
|---------|-----------|-------|-------|
| **Optymalizator** | Różne: SGD (MoCo), LARS (BYOL), Adam (SubTab, TS-TCC) | Adam z warmup-cosine LR schedulerem dla wszystkich metod | Spójność frameworku; LARS wymaga dużych batchy (≥4096), SGD z lr=0.03 nie konwerguje stabilnie na naszym pipeline. Parametry Adam (lr, wd) są strojone przez HPO. |
| **Liczba epok pre-treningu** | MoCo: 200 epok; BYOL: 1000 epok; SubTab: 100; TS-TCC: 40 | Domyślnie 100 epok; HPO używa 20 epok | Ograniczenia czasowe GPU T4 (12 h/sesję na Kaggle). HPO z 20 epokami wykrywa sensowny obszar hiperparametrów. |
| **Liczba epok fine-tuningu** | BYOL: 80 epok; TS-TCC: 10; pozostałe różnie | Domyślnie 50 epok z early stopping (patience=10, monitor: val macro_f1) | Spójność. Early stopping zapobiega przeuczeniu przy małych frakcjach etykiet (6.25%). |
| **Gradient clipping** | Brak w większości oryginałów | `max_norm=1.0` (global norm) przed każdym optimizer.step() | Zapobiega dywergencji numerycznej (stwierdzono NaN w BYOL bez clipping). |
| **Batch size** | MoCo: 256; BYOL: 4096; SubTab: 256; TS-TCC: 128 | Domyślnie 256 dla wszystkich; strojone przez HPO | GPU T4 ma 16 GB VRAM — batch 4096 nie mieści się dla modeli obrazowych. |
| **Sprzęt (GPU)** | Oryginały: 8× V100 (MoCo), 512 TPU (BYOL) | Kaggle GPU T4 x2 (2×16 GB), 12 h/sesję | Ograniczenia infrastruktury akademickiej. Dwa procesy (jeden per karta) z rozdzielonymi seedami zamiast DDP. |
| **Subsample PathMNIST** | PathMNIST: pełne 107 180 obrazów | `--subsample 20000` stratyfikowany (9 klas proporcjonalnie) | Pełny dataset na T4 zajmuje ~3 h/seed × 10 seedów = 30 h, nie mieści się w 12 h sesji. Subsample zachowuje proporcje klas. |
| **Subsample Credit Card Fraud** | Pełne 284 807 wierszy | `--subsample 30000` stratyfikowany | Subsample 30k **zachowuje wszystkie 492 fraudy** (klasa pozytywna), co **zmienia proporcję klas**: w pełnym zbiorze klasa=1 stanowi 0.17%, w subsampled ~1.64%. Wyniki są tym samym trudniejsze do porównania z literaturą na pełnym zbiorze. |
| **Liczba seedów** | Większość prac: 1–3 seedy | 10 seedów (0–9) dla każdego datasetu (UCI HAR, Credit Card, PathMNIST); aktualna liczba zapisana w polu `config.seeds` pliku wyników i wypisywana w podsumowaniu | Stabilność wyników — 10 seedów daje rozsądny przedział ufności t-Studenta (df=9) i rzetelne różnice sparowane. |
| **Liniowa ewaluacja** | Większość prac: 1–2 frakcje etykiet | 5 frakcji: 6.25%, 12.5%, 25%, 50%, 100% | Pełna krzywa SSL vs baseline. |

---

## 2. MoCo v2

**Referencja:** He et al., „Improved Baselines with Momentum Contrastive Learning" (2020).
Kod źródłowy: `third_party_reference/moco/` gałąź `KaimingHe-patch-2`, plik `moco/builder.py`.
Licencja: CC BY-NC 4.0.

| Element | Oryginał (`moco/builder.py`) | U nas | Powód |
|---------|------------------------------|-------|-------|
| **Shuffle BN** | `_batch_shuffle_ddp` / `_batch_unshuffle_ddp` (Alg. 3): shuffle próbek między GPU przed forward encoder_k, unshuffle po → każde GPU widzi różny podzbiór batcha w BN | `SplitBatchNorm2d` w **obu** enkoderach (q i k) + single-GPU `_batch_shuffle_single_gpu` / `_batch_unshuffle_single_gpu` wokół forward encoder_k (port z notebooka MoCo CIFAR `colab/moco_cifar10_demo.ipynb`). Tasowanie: losowa permutacja → encoder_k → argsort → oryginalna kolejność. | Shuffle BN DDP wymaga wielu GPU. CIFAR notebook pokazuje implementację single-GPU z SplitBN w obu enkoderach — para (anchor, key) trafia do różnych podgrup SplitBN; tasowanie uniemożliwia enkodowi kluczy „rozpoznanie" par przez statystyki batcha. |
| **encoder_k tryb BN** | Enkoder kluczy wywoływany bez `.eval()` — domyślnie `train()`, BN aktualizuje running stats | encoder_k w `train()` przez cały pre-trening; `eval()` dopiero gdy cały model jest w eval (linear probing) | Oryginał nie zamraża gałęzi kluczy. `eval()` powodowałoby używanie statystyk 0/1 (brak aktualizacji), de facto inna funkcja niż gałąź query. |
| **Głowica MLP (v2): bez BN** | `moco/builder.py` L.31–32: `nn.Sequential(Linear(dim_mlp,dim_mlp), ReLU(), encoder_q.fc)` — brak BatchNorm | `_build_moco_v2_projector`: Linear→ReLU→Linear, bez BN | MoCo v2 celowo pomija BN w głowicy (w przeciwieństwie do SimCLR). `build_mlp` z `base.py` zawiera BN — nie może być użyty. |
| **Rozmiar kolejki K** | `moco/main_moco.py` default: `--moco-k 65536` | Domyślnie `queue_size=4096`; HPO ogranicza K ≤ n_train/4; aktualny K logowany przy init | K=65536 przekracza n_train dla naszych subsampled datasetów → wycieki przez recyklowanie batchy w kolejce. |
| **Inicjalizacja kolejki** | `moco/builder.py` L.39–40: losowe znorm. wektory | Losowe znorm. wektory (identycznie); opcjonalne `prefill_queue()` domyślnie wyłączone | Domyślna inicjalizacja zgodna z oryginałem. Prefill dostępny jako opcja dla stabilizacji 1. epoki. |
| **Augmentacja: VerticalFlip** | `moco/loader.py` (aug-plus): tylko `RandomHorizontalFlip` | Opcjonalny `vertical_flip=True` (domyślnie włączony dla PathMNIST) | Histopatologia ma symetrię pionową (skrawki tkankowe bez orientacji anatomicznej). |
| **Optymalizator** | `main_moco.py`: SGD lr=0.03, momentum=0.9, wd=1e-4, cosine schedule | Adam z warmup-cosine (patrz sekcja wspólna) | Patrz sekcja wspólna §1. |

---

## 3. BYOL

**Referencja:** Grill et al., „Bootstrap Your Own Latent" (NeurIPS 2020).
Kod źródłowy: `third_party_reference/deepmind-research/byol/` (JAX/Haiku).
Pliki referencyjne: `byol_experiment.py`, `utils/augmentations.py`, `utils/networks.py`, `configs/byol.py`.
Licencja: Apache 2.0.

| Element | Oryginał (JAX/Haiku) | U nas (PyTorch) | Powód |
|---------|----------------------|-----------------|-------|
| **Sieć target: `is_training=True`** | `byol_experiment.py` L.236: `forward.apply(..., is_training=True)` — Haiku BN liczy statystyki batcha w obu gałęziach | encoder_t i projector_t w `train()` podczas pre-treningu (bez gradientów — tylko forward pass) | Bezpośrednie odwzorowanie. Brak `train()` zamrażało BN na 0/1 (jak w `eval()`), co dawało błędne cele. |
| **MLP: ostatnia warstwa bez biasu** | `utils/networks.py` L.44: `Linear(output_size, with_bias=False)` | `_build_byol_mlp`: Linear(hidden, bias=True) → BN → ReLU → Linear(out, **bias=False**) | Port 1:1. Poprzedni `build_mlp()` miał bias=True wszędzie — odchylenie od referencji. |
| **Harmonogram τ** | `utils/schedules.py` L.21–24: `decay = 0.5×(1+cos(π×t/T)); τ = 1−(1−τ_base)×decay` — τ rośnie od τ_base do 1 | `update_momentum(step, total_steps)`: identyczny wzór | Poprzednia implementacja miała odwrócony cos — τ malało zamiast rosnąć. Naprawione. |
| **Strata: sposób uśredniania** | `byol_experiment.py` L.243–250: `repr_loss = loss_v1 + loss_v2; jnp.mean(repr_loss)` | `loss = (loss_v1_per_sample + loss_v2_per_sample).mean()` — sum per sample, mean po batchu | Poprzednia implementacja liczyła mean każdego widoku osobno — matematycznie identyczne, ale różny gradient przy niestandardowym reduction. |
| **Augmentacje: przypisanie solarize/blur** | `utils/augmentations.py` augment_config: **view1** = blur p=1.0 + solarize p=0.0; **view2** = blur p=0.1 + solarize p=0.2 | `_build_byol_augmentation(view=1)`: blur_p=1.0, solarize_p=0.0; `(view=2)`: blur_p=0.1, solarize_p=0.2 | Poprzedni kod miał solarize_p=0.2 na view=1 — odwrotne przypisanie. Naprawione. |
| **ColorJitter: parametry** | `utils/augmentations.py`: brightness=0.4, contrast=0.4, **saturation=0.2**, hue=0.1 | `_build_byol_augmentation`: brightness=0.4·js, contrast=0.4·js, **saturation=0.2·js**, hue=0.1·js (js=1.0 domyślnie) | Poprzedni kod kopiował SimCLR (saturation=0.8·js=0.4). Artykuł BYOL ma saturation=0.2. |
| **Augmentacja: VerticalFlip** | `utils/augmentations.py`: tylko `random_flip=True` (poziomy) w obu widokach | Opcjonalny `vertical_flip=True` (domyślnie dla PathMNIST) dodany do obu widoków | Histopatologia — symetria pionowa (patrz §2 MoCo). |
| **Zakres skali RandomResizedCrop** | `byol/utils/dataset.py`: `_decode_and_random_crop` z `area_range=(0.08, 1.0)` | `crop_scale=(0.08, 1.0)` — identycznie jak oryginał | Port 1:1 wartości z kodu źródłowego. Poprzednie zdanie „referencja nie ujawnia" było nieprawdziwe; oryginał jawnie definiuje `area_range` w `dataset.py`. |
| **Optymalizator** | `configs/byol.py` L.57–66: LARS, lr=0.2×(batch/256), wd=1.5e-6, warmup+cosine | Adam z warmup-cosine (patrz sekcja wspólna) | LARS wymaga batch ≥ 4096 (niedostępny na T4). Patrz §1. |

---

## 4. SCARF

**Referencja:** Bahri et al., „SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption"
(ICLR 2022, arXiv:2106.15147). Brak oficjalnego kodu źródłowego — implementujemy z artykułu.

| Element | Artykuł (arXiv:2106.15147) | U nas | Powód |
|---------|---------------------------|-------|-------|
| **Strata: InfoNCE (nie NT-Xent)** | Algorithm 1, linia 8: `L = -log(e^{s_{ii}/τ} / Σ_k e^{s_{ik}/τ})` — asymetryczny InfoNCE, kotwica→pozytyw | `info_nce_loss(z_anchor, z_positive)`: `sim = z_anchor @ z_positive.T / τ`; labels = diagonala | Poprzednia implementacja NT-Xent concat [z1,z2] przesuwa etykiety o B — semantycznie różna funkcja. |
| **Korupcja: dokładnie floor(c·M) cech** | Algorithm 1, linia 3–5: `q=⌊c·M⌋`, losuj `q` cech **bez powtórzeń** per wiersz (`I_i ⊂ {1,…,M}, |I_i|=q`) | `_corrupt()`: `q = int(c*M)`, `torch.randperm(M)[:q]` — dokładnie q cech bez powtórzeń | Poprzednia wersja używała Bernoulli per komórka → zmienna liczba cech (E[k]=c·M). Artykuł wymaga dokładnie ⌊c·M⌋. |
| **Wartości zastępcze: empiryczny rozkład X_train** | Algorithm 1, linia 2: `X̂_j = uniform over {x_j : x ∈ X_train}` — cały zbiór treningowy | `set_marginals(train_X)` zapisuje (N_train, M); `_corrupt()` próbkuje wiersz z `self._marginals[:,j]`; wywoływane automatycznie w `experiment.run_experiment()` | Poprzednia wersja próbkowała z bieżącego batcha (aproksymacja ~B/N_train). Artykuł precyzuje rozkład na całym X_train. `set_marginals` przyjmuje wyłącznie train split — brak wycieku val/test. |
| **Temperatura domyślna τ=1** | Section 4: *"we use a temperature τ of 1"* | `temperature: float = 1.0` | Poprzednia wersja miała τ=0.5. |
| **Enkoder: 3 warstwy zamiast 4** | Section 4: *"f consists of 4 layers ReLU networks with hidden dimension 256"* | `build_encoder("mlp")` → `build_mlp(in, hidden, out, num_layers=3)` → 2 ukryte + 1 wyjście | Zmiana wymagałaby modyfikacji `build_encoder` lub `BaseContrastiveModel` — naruszałoby to zamrożony interfejs bazowy. `hidden_dim=256` zachowane. |

---

## 5. SubTab

**Referencja:** Ucar et al., „SubTab: Subsetting Features of Tabular Data
for Self-Supervised Representation Learning" (NeurIPS 2021).
Kod źródłowy: `third_party_reference/SubTab/` (Apache 2.0).
Pliki referencyjne: `utils/model_utils.py`, `utils/loss_functions.py`, `src/model.py`, `config/mnist.yaml`.

| Element | Oryginał | U nas | Powód |
|---------|----------|-------|-------|
| **Jeden wspólny AEWrapper** | `SubTab/utils/model_utils.py`: jeden `Encoder`/`Decoder`/projektor wspólny dla wszystkich podzbiorów | `_AEWrapper` z jednym `_Encoder`, `_Decoder` i 2-warstwowym projektorem | Port bezpośredni. Poprzednia implementacja tworzyła osobne enkodery per podzbiór — błąd architektury. |
| **Architektura: `_Encoder`/`_Decoder`** | `config/mnist.yaml`: `shallow_architecture: true` → `ShallowEncoder` (tylko `HiddenLayers`, brak osobnej warstwy latent) + `ShallowDecoder` (jeden `Linear`) | `_HiddenLayers` + `_Encoder(enc_dims)` + `_Decoder(dec_dims)` z osobną warstwą latent i symetrycznym dekodem (odpowiednik trybu `shallow_architecture: false`) | Świadoma decyzja: wariant głęboki daje silniejszą reprezentację dla Credit Card (30 cech). `shallow_architecture: true` z MNIST zostało pominięte z powodu różnicy w złożoności datasetu; nie jest to port 1:1. |
| **Domyślne `dims=[256,256]`** | `config/mnist.yaml`: `dims: [784, 784]` (dla MNIST 784-dim) | `dims=[hidden_dim, hidden_dim]` z `hidden_dim=256` → `[256, 256]` | Wartość 784 jest specyficzna dla MNIST. Dla Credit Card (30 cech) używamy [256,256]; HPO dostraja `hidden_dim`. |
| **`overlap` jako ułamek [0,1)** | `config/mnist.yaml`: `overlap: 0.75` (ułamek); `src/model.py`: `n_overlap = int(overlap * n_column_subset)` | `overlap: float = 0.75` | Port 1:1. Poprzednia wersja używała `overlap: int` (bezwzględna liczba) — błąd. |
| **`tau=0.1` domyślnie** | `config/mnist.yaml`: `tau: 0.1` | `tau=0.1` | Wartość z konfiguracji referencyjnej. |
| **`cosine_similarity=False` domyślnie** | `config/mnist.yaml`: `cosine_similarity: false` | `cosine_similarity=False` | Wartość z konfiguracji referencyjnej. Tryb kosinusowy dostępny przez flagę. |
| **Szum: wektoryzacja GPU** | `src/model.py`: numpy + pętla per próbka, CPU | `_add_noise()`: wektoryzowane na tensorach PyTorch, GPU-compatible | Port semantyki — identyczny rozkład, lepsza wydajność. |
| **`n_subsets` musi być dzielnikiem `num_features`** | Oryginalny `subset_generator` dzieli kolumny całkowitoliczbowo — reszta `num_features % n_subsets` kolumn nie jest pokryta przez żaden podzbiór i znika z treningu/ewaluacji. Dla Credit Card (30 cech) w zakresie HPO 2–8: dozwolone n_subsets ∈ {2, 3, 5, 6} | HPO sugeruje wyłącznie dzielniki `num_features` w zakresie 2–8. Przy inicjalizacji model automatycznie koryguje n_subsets do największego dzielnika ≤ żądanej wartości i emituje ostrzeżenie. | Nieuwzględnienie dzielności powoduje ciche gubienie cech — n_subsets=4 dla 30 cech pomijałoby Amount i V28. |
| **Ewaluacja bez szumu na zbiorze testowym** | `eval.py` → `subset_generator` dodaje szum (`swap_noise`) również na zbiorze testowym | Liniowe próbkowanie (fine-tuning) działa na czystych cechach bez szumu; brak szumu na val/test | Szum na teście zamazuje ocenę jakości reprezentacji. Nasz protokół jest standardem w SSL: clean eval. |
| **`RandomInit-SubTabArch` baseline** | Brak w oryginale | `RandomInitSubTabArchModel` z identycznym `_AEWrapper` co `SubTabModel`, bez pre-treningu | Zapewnia uczciwe porównanie arch-to-arch (SubTab vs identyczna architektura z losowymi wagami). `RandomInitModel` (MLP) pozostaje jako wspólny baseline SCARF+SubTab. |

---

## 6. TS-TCC

**Referencja:** Eldele et al., „Time-Series Representation Learning via Temporal and Contextual Contrasting"
(IJCAI 2021). Kod źródłowy: `third_party_reference/TS-TCC/` (MIT License).
Pliki referencyjne: `models/attention.py`, `models/model.py`, `dataloader/augmentations.py`, `trainer/trainer.py`.

| Element | Oryginał (`TS-TCC/`) | U nas | Powód |
|---------|----------------------|-------|-------|
| **Seq_Transformer: wierny port** | `models/attention.py`: własny `Transformer` z `Residual(PreNorm(Attention + FeedForward))`, `scale=dim**-0.5`, `to_qkv bias=False`, `dim_feedforward=mlp_dim` | Port 1:1: klasy `_Residual`, `_PreNorm`, `_FeedForward`, `_Attention`, `_Transformer`; `einops` zastąpione natywnym `view/permute/einsum`; wszystkie hiperparametry i topologia identyczne | Poprzednia wersja używała `nn.TransformerEncoderLayer` (Post-LN, `dim_feedforward=mlp_dim*4`) — semantycznie różna architektura. `einops` nie jest zależnością projektu. |
| **Augmentacja weak: wektoryzacja** | `dataloader/augmentations.py`: pętla po próbkach (numpy, CPU); scale `(B, T)` broadcast po kanałach | Pełna wektoryzacja na tensorach; scale `(B, 1, T)` — broadcast po C, identyczny rozkład, GPU-friendly | Oryginał numpy+pętla; wersja wektoryzowana zachowuje semantykę (jeden czynnik na próbkę i krok czasowy, wspólny dla wszystkich kanałów). |
| **Augmentacja strong: port 1:1** | `dataloader/augmentations.py` `permutation(seg_mode="random")`: per-sample `num_segs ∈ [1, max_seg-1]`; dla `num_segs>1`: losowe punkty podziału (`np.random.choice(T-2, num_segs-1, replace=False)`), sort, `np.split`, losowa permutacja segmentów; `num_segs=1` → próbka niezmieniona; jitter N(0, jitter_ratio) po permutacji | Pętla per-próbka na numpy (CPU), taki sam algorytm co oryginał; wynik przenoszony do tensora PyTorch. Per-sample `num_segs` odzwierciedla oryginalne `randint(1, max_segments, size=(B,))`. | Poprzednia wersja używała `round(mean(n_segs_per))` → wszyscy w batchu mieli tę samą liczbę segmentów i identyczną długość (podział całkowitoliczbowy). Oryginał ma zmienną liczbę segmentów i zmienne długości (losowe punkty podziału). |
| **`RandomInitHARModel` zamiast `RandomInitModel`** | Brak dedykowanego baseline w oryginale | `RandomInitHARModel` z tą samą architekturą CNN (`_BaseModel`) co `TSTCCModel` | Uczciwe porównanie arch-to-arch: baseline i SSL mają identyczny enkoder. `RandomInitModel` (MLP) używa innego enkodera. |
| **Wagi straty: stałe λ1=1, λ2=0.7** | `trainer/trainer.py`: `loss = (TC1+TC2)×1 + NTXent×0.7` — stałe wartości | `_lambda1=1.0, _lambda2=0.7` — stałe pola modelu; usunięte z przestrzeni HPO | Oryginał używa stałych wag. Usunięto `tc_weight` z HPO dla spójności. |
