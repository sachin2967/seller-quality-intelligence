# Model card: seller risk models

## Models

| Track | Model | Purpose |
| --- | --- | --- |
| Temporal | Class-weighted logistic regression, 8 lookback features | Point-in-time retrospective benchmark |
| Retrospective | Logistic regression baseline and LightGBM champion | Reconstruct a rule-derived label; served by the API |
| Retrospective | Cox proportional hazards | In-sample association exploration |

Temporal features: orders, items, GMV, average price, late rate, mean review, transit
days and tenure days (all as of the cutoff).

## Intended use

Studying point-in-time feature construction and honest retrospective evaluation on a
public marketplace dataset, and demonstrating a diagnostic scoring service.

## Out-of-scope use

Ranking or sanctioning real sellers, forecasting churn, estimating savings or ROI, or
choosing interventions. No intervention or counterfactual data exists.

## Data

Olist Brazilian E-Commerce Public Dataset (Kaggle): a small, finite historical
extract. Orders and sellers are not independent, and snapshots repeat sellers.

## Evaluation

Temporal holdout (cutoff 2018-03-27, 379 rows, prevalence 0.420):

| Slice | ROC-AUC | Average precision |
| --- | ---: | ---: |
| All holdout rows | 0.669 | 0.632 |
| 133 rows from sellers absent in training (prevalence 0.346) | 0.635 | 0.468 |

Holdout Brier score 0.227.

Retrospective 5-fold random-seller CV (1,104 sellers, 36.3% labeled at-risk):
LightGBM ROC-AUC 0.833 / PR-AUC 0.821, logistic regression 0.784 / 0.708. These measure
reconstruction of a rule, not prediction. Cox concordance 0.706 is in-sample.

## Limitations

- Outcomes exclude incomplete or undelivered future orders, which can bias risk.
- Only six cutoffs and one holdout cutoff; no confidence intervals.
- Scores are uncalibrated; `risk_probability` in the API is a compatibility field name.
- No threshold selection, proportional-hazards diagnostics or censoring treatment.
- Cancellation rate is not a temporal feature because the point-in-time history holds
  completed deliveries only.
- Recommended actions are heuristic and untested as interventions.
