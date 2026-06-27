# `src/lane_detection/` — Détection de voie

Détecte la voie courante dans une image RGB et fournit la **direction** et
l'**angle** de braquage pour aligner la voiture sur le centre de sa voie.

Modèle : **YOLOPv2** pré-entraîné (RGB → zone roulable + lignes de voie, lignes
continues), suivi d'un post-traitement géométrique qui isole la voie ego et en
déduit le centre + l'angle. Vision pure, aucune fonction CARLA à l'inférence.

## Deux modes d'exécution

### 1. Fonction (pour l'IA centrale)
```python
from lane_perception import estimate

direction, angle = estimate(rgb)        # rgb : image RGB (H, W, 3) uint8
# direction : "GAUCHE" / "DROITE" / "ALIGNE" / "NONE"
# angle     : degrés, <0 = gauche, >0 = droite (0 si aligné / NONE)
```
Le modèle se charge une fois (singleton). Pour le dict complet :
`LaneDetector().detect(rgb)`.

### 2. Viewer live (highlight)
```powershell
python live_highlight.py
```
Lance CARLA, affiche l'image avec la détection (zone roulable, lignes,
trajectoire, consigne). `q`/Échap quitter, `s` sauver une frame.

## Fichiers (production)

| Fichier | Rôle |
|---|---|
| `lane_perception.py` | **Production** : modèle YOLOPv2 + `estimate()` + rendu (`draw_overlay`) |
| `lane_geometry.py` | Géométrie pure (masques → centre voie, trajectoire, direction, angle), sans modèle |
| `live_highlight.py` | Viewer live (mode highlight) |
| `carla_integration.py` | Spawn monde / véhicule / caméra |
| `utils.py` | Conversion image CARLA + constantes caméra |
| `weights/yolopv2.pt` | Poids YOLOPv2 (téléchargés au 1er lancement, gitignoré) |

## `test/` — code maison conservé (fine-tuning CARLA)

Pipeline de fine-tuning d'un modèle de segmentation sur CARLA (utile si on veut
un modèle spécifique CARLA plus tard), + anciennes approches :

- `generate_seg_dataset.py` — dataset CARLA labellisé (caméra sémantique)
- `train_lane_seg.py` — entraînement LRASPP MobileNetV3 → `test/weights/lane_seg.pt`
- `lane_seg_model.py` — inférence avec le modèle fine-tuné
- `yolop_lane.py`, `lane_highlight.py`, `debug_pipeline.py`, `main.py` — anciennes versions
