# Reproducao do congelamento e do teste (item 12)

Ambiente: pandas 3.0.6. Comparacao entre os valores commitados e os recomputados.
Resultado geral: TODOS IGUAIS (33/33 itens identicos).

| item | comprometido | reproduzido | igual |
|---|---|---|---|
| config XGBRes h30 | {"max_depth": 3, "n_estimators": 400, "learning_rate": 0.03, "reg_lambda": 20.0, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5} | {"max_depth": 3, "n_estimators": 400, "learning_rate": 0.03, "reg_lambda": 20.0, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5} | sim |
| thr B3 lim40 h30 | 40.0 | 40.0 | sim |
| thr XGBRes lim40 h30 | 40.0 | 40.0 | sim |
| thr B3 lim30 h30 | 30.0 | 30.0 | sim |
| thr XGBRes lim30 h30 | 29.0 | 29.0 | sim |
| teste MAE B3 h30 | 2.949 | 2.949 | sim |
| teste skill_B3 B3 h30 | 0.0 | 0.0 | sim |
| teste PR_AUC_entrada40 B3 h30 | 0.989 | 0.989 | sim |
| teste MAE XGBRes h30 | 2.242 | 2.242 | sim |
| teste skill_B3 XGBRes h30 | 0.2397 | 0.2397 | sim |
| teste PR_AUC_entrada40 XGBRes h30 | 0.9653 | 0.9653 | sim |
| config XGBRes h60 | {"max_depth": 3, "n_estimators": 400, "learning_rate": 0.03, "reg_lambda": 20.0, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5} | {"max_depth": 3, "n_estimators": 400, "learning_rate": 0.03, "reg_lambda": 20.0, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5} | sim |
| thr B3 lim40 h60 | 41.0 | 41.0 | sim |
| thr XGBRes lim40 h60 | 41.0 | 41.0 | sim |
| thr B3 lim30 h60 | 27.0 | 27.0 | sim |
| thr XGBRes lim30 h60 | 27.0 | 27.0 | sim |
| teste MAE B3 h60 | 4.866 | 4.866 | sim |
| teste skill_B3 B3 h60 | 0.0 | 0.0 | sim |
| teste PR_AUC_entrada40 B3 h60 | 0.9762 | 0.9762 | sim |
| teste MAE XGBRes h60 | 3.75 | 3.75 | sim |
| teste skill_B3 XGBRes h60 | 0.2292 | 0.2292 | sim |
| teste PR_AUC_entrada40 XGBRes h60 | 0.908 | 0.908 | sim |
| config XGBRes h90 | {"max_depth": 2, "n_estimators": 400, "learning_rate": 0.03, "reg_lambda": 10.0, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5} | {"max_depth": 2, "n_estimators": 400, "learning_rate": 0.03, "reg_lambda": 10.0, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5} | sim |
| thr B3 lim40 h90 | 43.0 | 43.0 | sim |
| thr XGBRes lim40 h90 | 41.0 | 41.0 | sim |
| thr B3 lim30 h90 | 27.0 | 27.0 | sim |
| thr XGBRes lim30 h90 | 27.0 | 27.0 | sim |
| teste MAE B3 h90 | 6.267 | 6.267 | sim |
| teste skill_B3 B3 h90 | 0.0 | 0.0 | sim |
| teste PR_AUC_entrada40 B3 h90 | 0.9753 | 0.9753 | sim |
| teste MAE XGBRes h90 | 5.221 | 5.221 | sim |
| teste skill_B3 XGBRes h90 | 0.1668 | 0.1668 | sim |
| teste PR_AUC_entrada40 XGBRes h90 | 0.945 | 0.945 | sim |
