# Przewodnik po ponownym uruchomieniu eksperymentów

Dokument wskazuje, które eksperymenty wymagają przeliczenia po zmianach
w gałęzi `fix/faithful-implementations`, podaje gotowe komendy Kaggle
i szacunki czasu na sesję GPU T4 x2 (12 h limit).

---

## Co się zmieniło i które wyniki są nieważne

| Dataset | Algorytm | Zmiana | Wyniki nieważne? |
|---------|----------|--------|-----------------|
| UCI HAR | **TS-TCC** | `RandomInitHARModel` zastąpił `RandomInitModel` jako baseline; wagi straty λ1=1, λ2=0.7 jako stałe (usunięte z HPO) | **TAK** — baseline zmienił architekturę, HPO zmienił przestrzeń |
| UCI HAR | **RandomInit** (HAR) | Zmieniony na `RandomInitHARModel` (arch. CNN identyczna z TS-TCC) | **TAK** |
| Credit Card | **SubTab** | Jeden wspólny `_AEWrapper` zamiast osobnych enkoderów; `overlap` jako ułamek | **TAK** |
| Credit Card | **RandomInit-SubTabArch** | Nowy model (poprzednio nie istniał) | **TAK** (brak poprzednich wyników) |
| Credit Card | **SCARF** | InfoNCE zamiast NT-Xent; dokładnie ⌊c·M⌋ cech; marginale X_train; τ=1.0 | **TAK** |
| Credit Card | **RandomInit** (MLP) | Bez zmian | nie |
| PathMNIST | **MoCo** | MLP głowica bez BN; SplitBN; τ=0.2; kolejka K dostosowana do n_train | **TAK** |
| PathMNIST | **BYOL** | Poprawiony τ schedule (rosnący); poprawione augmentacje (solarize/blur, saturation); target w train() | **TAK** |
| PathMNIST | **RandomInit** (MLP) | Bez zmian | nie |

**Podsumowanie: nie trzeba przeliczać** `RandomInit` (MLP) dla Credit Card i PathMNIST.

---

## Szacunki czasu na GPU T4 (jedna karta, 10 seedów)

Wartości szacunkowe na podstawie smoke testów i typowych proporcji epok → pełny run.
HPO: 20 trials × 20 epok pre-treningu + 15 epok finetune per algo.

| Dataset | Algorytmy | Szacunek (10 seedów, 2×T4) |
|---------|-----------|---------------------------|
| UCI HAR | RandomInit-HAR + TS-TCC | ~1.5–2 h (cały zestaw) |
| Credit Card | RandomInit-SubTabArch + SubTab + SCARF | ~3–5 h |
| PathMNIST | MoCo + BYOL | ~8–10 h |

> PathMNIST nie mieści się pewnie w jednej 12 h sesji z marginesem — plan na **2 sesje**
> z `--resume` opisany poniżej.

---

## Komendy Kaggle — UCI HAR

Cały zestaw HAR mieści się wygodnie w jednej sesji (~2 h).

```bash
%%bash
# Przygotuj splity (raz)
python run_experiments.py \
    --dataset uci_har \
    --prepare-only --seeds 0 1 2 3 4 5 6 7 8 9

# Dwa procesy równolegle
CUDA_VISIBLE_DEVICES=0 python run_experiments.py \
    --dataset uci_har \
    --seeds 0 2 4 6 8 \
    --results-file results_uci_har_gpu0.json \
    --resume > log_har_gpu0.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python run_experiments.py \
    --dataset uci_har \
    --seeds 1 3 5 7 9 \
    --results-file results_uci_har_gpu1.json \
    --resume > log_har_gpu1.txt 2>&1 &

wait

python scripts/merge_results.py \
    results_uci_har_gpu0.json results_uci_har_gpu1.json \
    --out results_uci_har.json
```

Aby uruchomić **tylko zmienione algorytmy** (HAR: TS-TCC + RandomInit):

```bash
CUDA_VISIBLE_DEVICES=0 python run_experiments.py \
    --dataset uci_har \
    --algorithms TSTCC RandomInit \
    --seeds 0 2 4 6 8 \
    --results-file results_uci_har_gpu0.json \
    --resume > log_har_gpu0.txt 2>&1 &
```

---

## Komendy Kaggle — Credit Card Fraud

Credit Card mieści się w jednej sesji (~3–5 h). Subsample 30k zachowuje
wszystkie 492 fraudy, ale zmienia proporcję klas (patrz DEVIATIONS.md §1).

```bash
%%bash
# Przygotuj splity
python run_experiments.py \
    --dataset credit_card --subsample 30000 \
    --prepare-only --seeds 0 1 2 3 4 5 6 7 8 9

# Dwa procesy równolegle
CUDA_VISIBLE_DEVICES=0 python run_experiments.py \
    --dataset credit_card --subsample 30000 \
    --seeds 0 2 4 6 8 \
    --results-file results_cc_gpu0.json \
    --resume > log_cc_gpu0.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python run_experiments.py \
    --dataset credit_card --subsample 30000 \
    --seeds 1 3 5 7 9 \
    --results-file results_cc_gpu1.json \
    --resume > log_cc_gpu1.txt 2>&1 &

wait

python scripts/merge_results.py \
    results_cc_gpu0.json results_cc_gpu1.json \
    --out results_credit_card.json
```

Tylko zmienione algorytmy (SubTab + RandomInit-SubTabArch + SCARF):

```bash
CUDA_VISIBLE_DEVICES=0 python run_experiments.py \
    --dataset credit_card --subsample 30000 \
    --algorithms SubTab RandomInit-SubTabArch SCARF \
    --seeds 0 2 4 6 8 \
    --results-file results_cc_gpu0.json \
    --resume > log_cc_gpu0.txt 2>&1 &
```

---

## Komendy Kaggle — PathMNIST (2 sesje)

PathMNIST z subsample 20k i 10 seedami szacunkowo zajmuje 8–10 h na 2×T4.
Bezpieczny plan: 5 seedów w sesji 1, pozostałe 5 w sesji 2.

### Sesja 1 (seedów 0,2,4,6,8)

```bash
%%bash
# Przygotuj splity (tylko raz — też przy wznowieniu sesji)
python run_experiments.py \
    --dataset pathmnist --subsample 20000 \
    --prepare-only --seeds 0 1 2 3 4 5 6 7 8 9

# Uruchom seedy 0,2,4,6,8
CUDA_VISIBLE_DEVICES=0 python run_experiments.py \
    --dataset pathmnist --subsample 20000 \
    --algorithms MoCo BYOL RandomInit \
    --seeds 0 4 8 \
    --results-file results_pathmnist_gpu0.json \
    --resume > log_pm_gpu0.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python run_experiments.py \
    --dataset pathmnist --subsample 20000 \
    --algorithms MoCo BYOL RandomInit \
    --seeds 2 6 \
    --results-file results_pathmnist_gpu1.json \
    --resume > log_pm_gpu1.txt 2>&1 &

wait
echo "Sesja 1 gotowa — zapisz results_pathmnist_gpu0.json i gpu1.json"
```

### Sesja 2 (seedów 1,3,5,7,9) — wznowienie

```bash
%%bash
# Pliki gpu0/gpu1 z sesji 1 muszą być obecne (--resume je uzupełni, nie nadpisze)
CUDA_VISIBLE_DEVICES=0 python run_experiments.py \
    --dataset pathmnist --subsample 20000 \
    --algorithms MoCo BYOL RandomInit \
    --seeds 1 5 9 \
    --results-file results_pathmnist_gpu0.json \
    --resume > log_pm_gpu0_s2.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python run_experiments.py \
    --dataset pathmnist --subsample 20000 \
    --algorithms MoCo BYOL RandomInit \
    --seeds 3 7 \
    --results-file results_pathmnist_gpu1.json \
    --resume > log_pm_gpu1_s2.txt 2>&1 &

wait

# Scal po obu sesjach
python scripts/merge_results.py \
    results_pathmnist_gpu0.json results_pathmnist_gpu1.json \
    --out results_pathmnist.json
```

> **Uwaga:** `--resume` pomija seedy już zapisane w pliku, więc można go bezpiecznie
> uruchomić ponownie bez ryzyka nadpisania. Plik wyników jest zapisywany po każdym seedzie.

---

## Podgląd postępu i diagnostyka

```bash
# W osobnej komórce Kaggle:
tail -f log_pm_gpu0.txt
# lub
cat log_pm_gpu0.txt | grep -E "\[Pretrain\]|\[Finetune\]|SEED|loss"

# Zużycie GPU:
nvidia-smi -l 10

# Ile seedów ukończonych w pliku wyników:
python -c "
import json
with open('results_pathmnist_gpu0.json') as f:
    r = json.load(f)
seeds = list(r.get('results', {}).get('MoCo', {}).keys())
print(f'MoCo: {len(seeds)} seedów: {seeds}')
"
```

---

## Kolejność sesji (zalecana)

1. **HAR** — najkrótszy (~2 h), dobry do weryfikacji poprawności pipeline po zmianach
2. **Credit Card** — ~3–5 h, jeden run
3. **PathMNIST sesja 1** — seedy parzyste (~5–6 h)
4. **PathMNIST sesja 2** — seedy nieparzyste + merge (~4–5 h)
