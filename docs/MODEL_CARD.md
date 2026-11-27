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

Olist Brazilian E-Commerce Public Dataset (Kaggle). The rerun uses the
complete cached historical release and checks it against the source manifest. Orders and sellers are not independent, and snapshots repeat sellers.

## Evaluation

The fixed point-in-time benchmark uses the full Olist release. Delivered orders stay
in the history when reviews are unavailable. Training uses only mature outcomes.

| Holdout at 2018-03-27 | Rows | Prevalence | ROC-AUC | Average precision |
| --- | ---: | ---: | ---: | ---: |
| All holdout sellers | 961 | 0.6035 | 0.7370 | 0.8119 |
| Sellers absent from eligible training snapshots | 450 | 0.5711 | 0.7017 | 0.7866 |

Holdout Brier score is 0.2079.
The README's old-vs-new table includes [the matched original-code rerun](../reports/temporal_migration_comparison.json).

The rolling evaluation covers 11 eligible cutoffs. At 2018-04-01,
fixed logistic regression achieved AP 0.8065 versus
prevalence 0.5883. Its 95% seller-cluster interval was
0.7736–0.8377, conditional on that period.

Retrospective 5-fold random-seller CV on 2,462 sellers
(34.2% labeled at-risk) gives LightGBM ROC-AUC
0.826 and PR-AUC 0.769,
with logistic regression at 0.793 and
0.699, respectively. These results measure
reconstruction of a contemporaneous rule. Cox concordance 0.666
is apparent in-sample.

## Limitations

- Outcomes exclude incomplete or undelivered future orders, which can bias risk.
- The single holdout has no uncertainty interval. Rolling intervals describe
  within-window seller sampling and the windows overlap.
- Scores are uncalibrated; `risk_probability` in the API is a compatibility field name.
- No threshold selection, proportional-hazards diagnostics or censoring treatment.
- Cancellation rate is not a temporal feature because the point-in-time history holds
  completed deliveries only.
- Recommended actions are heuristic and untested as interventions.
