# Freight Rate Prediction

Predicts `posted_rate` for freight loads. Final predictions are for Nov–Dec 2025 loads
(`data/validation.csv`) using labelled data from Jan–Oct 2025 (`data/train_test.csv`).

## Run

```bash
python -m pip install -r requirements.txt
python train_predict.py --data-dir data --output-dir outputs       # validation + final fit (~2-3 min)
python score.py --predictions outputs/validation_predictions.csv \
                --december-predictions outputs/december_chart_inputs.csv \
                --output-dir outputs/scorer_results
```

Outputs: `outputs/validation_predictions.csv` (load_id,predicted_rate), the completed
`outputs/december_chart_inputs.csv`, `outputs/validation_metrics.json`, `outputs/scorer_results/candidate_december.png`.
Use `--skip-cv` to skip the validation folds and only fit + predict.

## Approach (short)

* **Split:** `validation.csv` is strictly later in time (Nov–Dec) and contains 8 cities never seen in training,
  so a random split would be over-optimistic. I validate with forward-chaining 2-month folds
  (train strictly before, test strictly after) plus a fold that also hides 8 cities.
* **Cleaning:** ~1.4% of labels are corrupt (rate ×2–4.5 or ÷2–5 vs. similar loads) and are dropped from
  training only (robust 5σ rule on a first-pass fit). Negative weights (sign error) → abs, weights capped at 47,500,
  missing `market_index` → the day's median, missing weight left to LightGBM / median in the linear model.
* **Features:** log distance, distance, equipment, weight, market_index, quote_signal, day of week, day-level
  market signals (computed from features only, no labels), and coordinates instead of city names so the model
  generalises to unseen cities.
* **Model:** average of (A) LightGBM (Huber loss) with a time index, and (B) linear trend model + LightGBM on its
  residual; 3 seeds each. Time is the main risk: rates drift slowly and A/B extrapolate it differently.
* **December chart:** the input file has no market_index / quote_signal, so the day's aggregate values from
  `validation.csv` are used.
