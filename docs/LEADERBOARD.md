# Livrable soutenance — système intégré 4 modules, 0 ground-truth

> Éval déterministe 13-scénarios avec la **perception RÉELLE** (Karim YOLOPv2 + Franck), **0 GT CARLA**, `next_command` de Victor réparé. Barre visée : « roule droit + capable de tourner, collisions tolérées ».

| Modèle | signal de voie | Phase-1 | centrage (offset moy) | crashs /13 |
|---|---|---|---|---|
| **🥇 v23 88k** | lignes + **fallback drivable** | **7/8** | **0.40** | **0** |
| v23 model_final (110k) | idem | 1/8 | 0.39 | 0 (sur-entraîné : n'atteint plus) |
| v22 88k | lignes seules (offset périmé 94 %) | 2/8 | 0.74 | la plupart |
| v22 model_final | idem | 0/8 | 0.86 | 7 |
| v21 model_final | commande discrète, sans bearing | 1/8 | — | biais gauche, crash à 12 m |

**🏆 LIVRABLE FINAL = v23, checkpoint 88k** — `runs/2026-07-18_10-47_v23_drivable_real_110k/best_model.zip`.
- **7/8 Phase-1, 0 crash sur les 13 scénarios** : les 5 scénarios « rouler proprement » + 2 destinations GPS atteintes (turn_right, junction_straight) ; seul échec turn_left (66 m sans crash, cible non atteinte). Au-dessus du plafond historique *avec* GT (v18 : 3/8).
- **Le fix décisif (v23)** : quand le détecteur de lignes renvoie NONE (93,8 % des frames sur Town02), l'offset vient du masque **zone roulable** de YOLOPv2 (`src/ai/inference/lane_fusion.py`, code de Karim intact) — signal frais 98 % du temps, corr 0,57 vs GT (contre 0,18 pour les lignes). Centrage ÷2 (0,86 → 0,40).
- Démos : `demo_route_success.mp4` (**156 m sans collision**, HUD 14 inputs / 2 outputs + minimap) et `demo_best_model.mp4` (free-run).
- Limite mesurée : fiable jusqu'à ~150 m par trajet (curriculum d'entraînement ≤ 70 m) ; les routes > 400 m échouent. Piste : curriculum ~200 m.

> ⚠️ Le classement ci-dessous est en `--ground-truth-lane` (vérité terrain CARLA) — **hors-concours pour la soutenance** (« triché », convenu avec l'utilisateur). Gardé comme référence de plafond.

---

# Leaderboard — meilleur modèle (éval déterministe Phase 1, `--ground-truth-lane`)

Objectif : **proposer le meilleur modèle possible**. Juge = éval déterministe 13-scénarios (8 notés en Phase 1). **Éval vérifiée reproductible** (v18 `model_final` re-évalué = 3/8 identique, mêmes scénarios) → une éval par modèle suffit, leaderboard fiable.

## Classement complet (runs × checkpoints)

| Modèle | ckpt | P1 | Scénarios gagnés |
|---|---|---|---|
| 🥇 **v18** | **final (110k)** | **3/8** | turn_right, junction_straight, npc_crossing |
| v18 | 77k | 3/8 | straight, curve_right, turn_right |
| v18 | 88k | 3/8 | curve_right, turn_right, junction_straight |
| v17 | final (80k) | 2/8 | curve_left, npc_crossing |
| v19 | 90k | 2/8 | curve_right, npc_crossing |
| v16 | final (40k) | 1/8 | curve_right |
| v19 | 105k / 135k | 1/8 | curve_right |
| v15 | final | 0/8 | — (passive) |
| v19 | final (150k) / 120k | 0/8 | — (sur-entraîné, crashe) |

## 🏆 Champion actuel : `runs/2026-07-15_13-34_ppo_v18_long_110k/model_final.zip` — 3/8
2 scénarios reach-dest (navigation GPS, les plus durs : turn_right, junction_straight) + npc_crossing. 98 % en mouvement, **0 crash** sur les 13, 36 m de distance moyenne. Vidéo : `.../eval_model_final.mp4`. Config : route-aware + gate + bearing + curriculum + squash/stall + vérité-terrain, entraîné 110k.

## Enseignements de la sélection
- **Plafond de 3/8 par modèle**, mais **union = 6/8** (curve_left, curve_right, junction_straight, npc_crossing, straight, turn_right ; jamais gagnés : turn_left, npc_follow). La config *peut* tout faire, mais pas dans un seul modèle → **variance dominante**.
- **`model_final` peu fiable** : sur-entraînement réel (v19 150k final = 0/8 vs son checkpoint 90k = 2/8). Toujours miner les checkpoints. Sweet spot ~90-110k.
- **Plus d'entraînement ne bat pas 110k** (v19 150k < v18 110k).
- `spawn_idx=13` (scénario `straight`) **n'est pas cassé** (conduit normalement) — les échecs `straight` sont de la variance (v18 77k le gagne, v18 final non).

## v20 (dernier tirage) — n'a pas battu v18
v20 `model_final` = 2/8 (curve_right, npc_crossing), checkpoints 77k/88k = 1/8. **Plafond de 3/8 confirmé par un 5e run.**

## ✅ CONCLUSION — MODÈLE FINAL PROPOSÉ
**`runs/2026-07-15_13-34_ppo_v18_long_110k/model_final.zip` — 3/8 Phase 1.**
- Meilleur des ~15 modèles évalués (5 runs × checkpoints). Éval reproductible.
- Réussit 2 navigations GPS (turn_right, junction_straight) + npc_crossing, 98 % en mouvement, **0 crash** sur 13 scénarios.
- Config : route-aware + gate anti-passivité + bearing-to-goal + curriculum + squash/stall + vérité-terrain, 110k steps.
- Vidéo : `runs/2026-07-15_13-34_ppo_v18_long_110k/eval_model_final.mp4`.
- Point de départ il y a 24 h : 0/8 (voiture immobile). Résultat honnête : le plafond de variance de cette approche est 3/8.
