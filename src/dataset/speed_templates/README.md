# Templates des panneaux de limite de vitesse

`enrich_labels.py` charge automatiquement les fichiers de ce dossier comme
templates pour le matching des panneaux, **avant** de tomber en fallback
sur les templates synthétiques.

## Comment populer ce dossier

1. **Lance d'abord une collecte avec --debug-drops** :

   ```powershell
   uv run -m src.dataset.run_collection --duration 600 --npcs 30
   uv run -m src.dataset.enrich_labels --run data/runs/<run> --debug-drops
   ```

2. **Ouvre** `data/runs/<run>/debug_dropped_signs/` en mode vignettes.

3. **Sélectionne 1 crop par valeur** :
   - Pour chaque valeur dans {30, 40, 50, 60, 70, 80, 90}, trouve un PNG où
     tu vois clairement un panneau de cette valeur.
   - Prends un crop pas trop petit (≥ 15×15 px), bien centré, lentille
     claire, peu d'occlusion. C'est le template — il doit être représentatif.

4. **Renomme et copie** chaque crop sélectionné dans ce dossier :

   ```powershell
   Copy-Item data\runs\<run>\debug_dropped_signs\000012_0_best70@-0.02.png src\dataset\speed_templates\30.png
   Copy-Item data\runs\<run>\debug_dropped_signs\000020_0_best40@0.07.png  src\dataset\speed_templates\60.png
   # ... idem pour 40, 50, 70, 80, 90
   ```

   Format final accepté :
   - `<value>.png` — un template unique (ex : `30.png`)
   - `<value>_*.png` — N templates différents par valeur, tous chargés et
     scorés en parallèle (ex : `30_a.png`, `30_proche.png`, `30_de_biais.png`).
     Le score final pour la valeur = max sur tous les templates.

   **Plusieurs templates par valeur** = mieux. Couvre différents angles,
   distances, conditions de lumière. 2-4 templates par valeur est un bon
   compromis. Au-delà ça ralentit le post-process sans gain notable.

5. **Relance enrich_labels** :

   ```powershell
   uv run -m src.dataset.enrich_labels --run data/runs/<run> --debug-drops
   ```

   Vérifie que les compteurs `tl_classified` / `sign_classified` augmentent
   et que les drops baissent. Les scores des matches devraient passer
   de ~0.0 (random synthétique) à ~0.4-0.7 (real CARLA).

## Valeurs disponibles dans CARLA

CARLA expose `traffic.speed_limit.{30,40,50,60,90}` selon les maps. Les
maps Town01-Town05 contiennent principalement 30, 60, 90. Les valeurs 40,
50, 70, 80 sont plus rares — si tu ne trouves pas de bon crop, laisse le
template synthétique en fallback (`enrich_labels` continuera à utiliser le
synthétique pour ces valeurs).

## Limites

- Si tes crops template sont eux-mêmes mal cropés ou flous → mauvaise
  référence, faux positifs. Garde des templates **propres**.
- Le matching reste sensible à la perspective : un crop pris face au
  panneau matchera moins bien un panneau vu de biais. À terme, considérer
  plusieurs templates par valeur (différents angles) — mais on n'en est pas
  encore là.
