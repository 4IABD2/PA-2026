# Génération du dataset YOLO — problèmes rencontrés et solutions

Ce document récapitule les bugs rencontrés en mettant en place la génération
de labels YOLO depuis CARLA (`yolo_labels.py`), et la solution retenue pour
chacun. Il sert de mémoire technique pour comprendre **pourquoi** le code a
sa forme actuelle, et de checklist quand tu reprends ou étends le pipeline.

Pour la vue d'ensemble pédagogique du code lui-même, voir
[../../COMPUTE_LABELS_EXPLAINED.md](../../COMPUTE_LABELS_EXPLAINED.md). Pour les
décisions d'architecture côté Franck, voir
[../../GUIDE_FRANCK.md](../../GUIDE_FRANCK.md).

---

## 1. Au départ : `compute_labels` n'était pas implémenté

**Symptôme** : tous les `labels_yolo/*.txt` ressortaient vides après une
collecte ; `compute_labels` levait `NotImplementedError`.

**Cause** : le scaffold initial avait juste le squelette de classe avec un
docstring de spec, pas de logique.

**Fix** : implémenter la conversion **mask d'instance → bboxes YOLO** :
1. Lire `self.instance_capture._last_rgb` (buffer live du sensor)
2. Pack en `uint32` via `pack_instance_carla` (R=class_id, GB=instance_id)
3. `np.unique(instance_map)` pour itérer chaque objet
4. Pour chaque instance : récupérer son `class_id`, mapper vers la classe
   YOLO, calculer la bbox via `np.where(mask)`, normaliser dans `[0, 1]`

Voir [yolo_labels.py:87-180](yolo_labels.py).

---

## 2. État rouge/jaune/vert des feux : `world.get_actor()` échoue

**Symptôme** : on voulait sortir directement 5 classes
(`red_light/yellow_light/green_light`) en lisant l'état du feu via
`actor.get_state()`. Mais aucun feu n'apparaissait dans les labels.

**Investigation** :
- Étape 1 : on supposait que `world.get_actor(inst_id)` retournerait le
  contrôleur du feu. Échec → on a découvert que CARLA n'encode que les
  **16 bits du bas** de `actor.id` dans le mask, donc les feux avec `id > 65535`
  étaient introuvables par lookup direct.
- Étape 2 : on a indexé tous les contrôleurs `traffic.traffic_light*` par
  `actor.id & 0xFFFF` une fois par frame pour faire le match. Toujours rien.
- Étape 3 : diagnostic en logs pendant la collecte. Résultat : les
  `instance_id` du mask ne correspondent **à aucun actor Python-accessible**.
  Ils référencent des sous-meshes UE4 (poteau, boîtier, lentilles) que CARLA
  n'expose pas du tout en Python.

**Cause racine** : la pipeline `Python → world.get_actor → get_state` est
**structurellement impossible** pour les feux statiques de la map CARLA.
Le mask les tague avec des IDs de meshes UE4 internes, pas avec les IDs des
contrôleurs `TrafficLight`.

**Fix retenu** : pipeline en deux étages.

1. **Collecte** (ce module) écrit une classe générique `traffic_light=2`
   pour tous les feux détectés. Pas d'état.
2. **Post-process** ([colorize_traffic_lights.py](colorize_traffic_lights.py))
   relit chaque label `class==2`, crope la zone correspondante de l'image RGB,
   classifie par analyse HSV (rouge / jaune / vert), écrit le résultat dans
   `labels_yolo_color/` avec le mapping 5-classes utilisé à l'entraînement.

**Bonus** : la même heuristique HSV marche à l'inférence, donc pas de
dépendance à CARLA pour décider de l'état d'un feu en production.

---

## 3. Feux non détectés malgré leur visibilité dans l'image

**Symptôme** : sur certaines frames, un feu est clairement visible dans
l'image RGB et apparaît bien dans le `semantic_viz/*.png` (orange = class 7),
mais aucun label `traffic_light` n'est généré.

**Investigation** : diagnostic des `instance_id` des pixels class 7 pour la
frame en question. Découverte que beaucoup de feux ont **`instance_id == 0`**
dans le mask (CARLA ne leur assigne pas d'instance trackable). Notre code
les skippait via `if inst_id == 0: continue`.

**Fix** : pour les feux uniquement, on **abandonne le mask d'instance** et on
travaille sur le mask **sémantique** via `cv2.connectedComponents`. Chaque blob
de pixels class 7 contigus = un feu. Voir [yolo_labels.py:109-136](yolo_labels.py).

```python
tl_class_mask = (class_map == 7).astype(np.uint8)
n_comp, comp_labels = cv2.connectedComponents(tl_class_mask, connectivity=8)
for comp_id in range(1, n_comp):
    blob = comp_labels == comp_id
    # filtrer + créer le label
```

**Avantages** :
- Détecte tous les feux visibles, indépendamment de l'instance ID
- Sépare correctement plusieurs feux dans le même frame
- Plus simple à raisonner

---

## 4. Feux trop petits filtrés (la voiture devait être collée)

**Symptôme** : seuls les feux à très courte distance (5-10m) apparaissaient
dans les labels. Au-delà → rien.

**Cause** : filtre `_MIN_BBOX_SIDE_PX = 6` appliqué uniformément. La lentille
d'un feu ne mesure que ~10 cm, donc à 50-80m sa bbox tombe sous 6 px. Le
filtre était calibré pour les véhicules (où l'instance sensor produit
régulièrement des artefacts de 2-3 pixels).

**Fix** : seuil **spécifique** pour les feux :
- `_MIN_TL_BBOX_SIDE_PX = 4` (vs 6 pour véhicules/piétons)
- `_MIN_TL_PIXELS = 25` (filtre la fragmentation derrière la végétation)

Voir [yolo_labels.py:45-51](yolo_labels.py).

---

## 5. L'ego (capot Tesla) labellisé comme un "vehicle"

**Symptôme** : la caméra est sur le toit, le capot occupe le bas de l'image.
Notre code le voyait comme un véhicule normal et générait un bbox énorme en
bas de chaque frame.

**Fix 1 (ID-based)** : skip l'instance qui correspond à `ego.id & 0xFFFF`.

**Fix 2 (filet de sécurité)** : skip toute bbox dont le bord bas atteint le
bord bas de l'image (`y2 >= image_h - 5`). C'est utile si le capot a une
sous-géométrie avec un `actor.id` différent de l'ego.

Voir [yolo_labels.py:104-107, 162-164](yolo_labels.py).

---

## 6. Motards labellisés "walker"

**Symptôme** : les personnes à moto/vélo étaient classées `walker` au lieu
de `vehicle`.

**Cause** : le mapping CityScape officiel CARLA distingue `Pedestrian` (12)
et `Rider` (13, un humain sur un véhicule). Initialement on mappait `13: 1`
(Rider → walker), ce qui est sémantiquement correct mais inutile pour la
conduite.

**Fix** : `13: 0` (Rider → vehicle). Pour la voiture autonome, un motard se
comporte comme un véhicule (trajectoire, vitesse, respect des feux), pas
comme un piéton.

Voir [yolo_labels.py:34-43](yolo_labels.py).

---

## 7. Véhicules fantômes à travers grilles / vitres

**Symptôme** : des bboxes `vehicle` apparaissaient parfois sur des éléments
de décor (garde-corps, grillages), avec un véhicule réel situé derrière.
Confirmé via le `semantic_viz/*.png` : on voit clairement un mini-blob bleu
(class Car) qui "fuit" à travers la barrière.

**Cause** : bug connu de l'instance sensor CARLA — certains matériaux
(grilles, vitres semi-transparentes) ne masquent pas le tag d'instance des
objets derrière. Le mask reçoit donc des pixels véhicule à des endroits
visuellement occultés.

**Fix** : filtre sur le **fill ratio** (pixels du mask / aire du bbox).
- Vrai véhicule : pixels remplissent >50% du bbox
- Fuite à travers grille : pixels dispersés en grille, fill ratio <20%

Seuil retenu : `_MIN_FILL_RATIO_NON_TL = 0.25`. Voir
[yolo_labels.py:53-56, 158-161](yolo_labels.py).

---

## 8. Feux très loin (derrière des arbres) faussement détectés

**Symptôme** : un feu invisible à l'œil nu (perdu dans la végétation à 100m+)
était parfois labellisé `traffic_light`.

**Cause** : avec les composants connexes sur le mask sémantique, même
quelques pixels visibles à travers les feuilles produisent un blob. Sans
seuil de taille suffisant, ces blobs passaient.

**Fix** : `_MIN_TL_PIXELS = 25` minimum par blob. Un feu lisible (même à
80m) atteint facilement cette taille. Un feu fragmenté derrière des arbres
produit plusieurs blobs de <10 pixels chacun → tous droppés.

---

## 9. Désynchronisation entre RGB et masks (bboxes décalées)

**Symptôme** : les bboxes (depuis le mask semantic/instance) sont décalées
de quelques pixels par rapport au contenu du RGB. Visible sur les crops
panneaux : le premier chiffre est coupé en deux, parce que le mask
référence la frame N et l'image RGB la frame N-1 (ou inversement).

**Cause** : CARLA en mode synchrone garantit qu'à chaque `world.tick()`
tous les sensors **produisent** une frame, mais les **callbacks Python**
de `sensor.listen()` tournent sur des **threads séparés**. Quand le
collector appelle `sensor.save()` immédiatement après `world.tick()`, le
buffer `_last_rgb` d'un sensor peut encore contenir la frame du tick
précédent si son callback n'a pas encore été exécuté.

**Fix** : chaque `CameraSensor` track maintenant le `image.frame` reçu
dans `_last_frame_num`. Avant de sauver, le collector appelle
`_wait_sensors_sync(expected_frame)` qui poll jusqu'à ce que **tous** les
sensors aient leur buffer à la frame attendue (timeout 2 s, warn si
dépassé).

Voir [sensors.py:53,81-84](sensors.py) et
[collector.py:_wait_sensors_sync](collector.py).

---

## 10. Panneaux de vitesse — détection via composants connexes + template matching

**Symptôme** : pas de symptôme initial, mais besoin produit : la voiture
doit pouvoir reconnaître les limites de vitesse pour les respecter.

**Cause** : même limitation que les feux (instance mask CARLA non fiable
pour les statiques de map). Lookup `world.get_actor(inst_id)` impossible.

**Fix** : copier l'approche TL.
- Côté **collecte** : composants connexes sur `class_id == 8` (CityScape
  TrafficSign), output en classe générique `traffic_sign=3` dans les raw
  labels. Filtres `_MIN_SIGN_PIXELS = 30`, `_MIN_SIGN_BBOX_SIDE_PX = 5`.
- Côté **post-process** ([enrich_labels.py](enrich_labels.py)) : template
  matching contre 7 templates synthétiques (30/40/50/60/70/80/90), générés
  au runtime via PIL (cercle rouge + chiffre noir centré, 64×64). Score
  `cv2.TM_CCOEFF_NORMED ≥ 0.45` requis, sinon drop.

---

## Récapitulatif des filtres actuellement actifs

Dans l'ordre d'application (cf. [yolo_labels.py](yolo_labels.py)) :

| Filtre | Cible | Justification |
|---|---|---|
| `inst_id == 0` | dynamiques | pixels fond / non trackable |
| `inst_id == ego.id & 0xFFFF` | dynamiques | exclure le capot ego (ID-based) |
| `class_id` hors `_CARLA_TO_YOLO` | dynamiques | classes non pertinentes (poteaux, ciel...) |
| `class_id ∈ {7, 8}` | dynamiques | traités séparément via composants connexes |
| feux via composants connexes (class_id=7) | TL | mask d'instance non fiable |
| `pixel_count < 25` | TL | feux fragmentés derrière végétation |
| `w_px < 4 ou h_px < 4` | TL | feux trop petits |
| panneaux via composants connexes (class_id=8) | Sign | mask d'instance non fiable |
| `pixel_count < 30` | Sign | panneaux fragmentés |
| `w_px < 5 ou h_px < 5` | Sign | panneaux trop petits |
| `w_px < 6 ou h_px < 6` | dynamiques | bruit de l'instance sensor |
| `fill_ratio < 0.25` | dynamiques | fuites à travers grilles / vitres |
| `y2 >= image_h - 5` | dynamiques | sous-mesh du capot ego (filet) |
| `distance > 80m` (si actor) | dynamiques | trop loin pour être utile |

Côté `enrich_labels` (post-process, drop le label si échec) :

| Filtre | Cible | Justification |
|---|---|---|
| HSV `max(R,Y,G) < 8 pixels` | feux raw 2 | aucune couleur dominante claire |
| Template match `score < 0.45` | panneaux raw 3 | aucune valeur 30..90 ne matche |

---

## Workflow validé

```powershell
# 1. Collecte CARLA
uv run -m src.dataset.run_collection --duration 60 --npcs 15

# 2. Enrichissement labels (raw 4-classes → final 12-classes)
uv run -m src.dataset.enrich_labels --run data/runs/<NOUVELLE>

# 3. Inspection visuelle (FiftyOne) ou en console
uv run -m src.tools.visualize_fiftyone --run data/runs/<NOUVELLE>
uv run -m src.dataset.inspect_run --run data/runs/<NOUVELLE>
```

À chaque modification de `yolo_labels.py` : **toujours refaire une nouvelle
collecte** avant de juger du résultat. Les `labels_yolo/*.txt` ne sont pas
régénérés sur les anciennes runs.

Pour les multi-runs (plusieurs maps × météos) :

```powershell
uv run -m src.dataset.collect_multi --maps Town01,Town03 --weathers ClearNoon,CloudyNoon --frames-per-run 500 --enrich
```

---

## Pour aller plus loin (problèmes ouverts / améliorations possibles)

- **Feux orange sous-représentés** : `Yellow` ne dure que ~3s sur un cycle de
  30s, donc rare dans les collectes. Peut nécessiter de l'augmentation
  visuelle ou de la pondération de classe à l'entraînement.
- **Seuils HSV à tweaker** : les seuils dans
  [enrich_labels.py](enrich_labels.py) lignes 41-50 sont pensés pour
  `ClearNoon`. À valider sur d'autres météos (`CloudyNoon`, `WetSunset`) —
  la teinte des feux change avec la lumière ambiante.
- **Seuil template matching** : `_MIN_TEMPLATE_SCORE = 0.45` dans
  [enrich_labels.py](enrich_labels.py). Si trop de panneaux droppés sur des
  vrais panneaux clairement visibles, baisser à 0.35-0.40. Si trop de faux
  positifs (un panneau "stop" classé "speed_30"), monter à 0.55.
- **Templates synthétiques imparfaits** : on génère les templates en PIL
  avec un cercle rouge + chiffre noir, mais la précision en pratique est
  proche du **hasard** (scores ~0.0 sur des crops réels). Solution : poser
  des crops CARLA réels dans
  [src/dataset/speed_templates/](speed_templates/) (un par valeur,
  `30.png`/`40.png`/...). `enrich_labels` les charge automatiquement en
  priorité sur les synthétiques. Voir
  [speed_templates/README.md](speed_templates/README.md) pour la procédure.
- **Distance des feux** : on n'a pas de filtre distance pour les feux
  (impossible à calculer via actor lookup). Pour les filtrer plus finement,
  il faudrait projeter les contrôleurs `TrafficLight` actuellement actifs en
  pixels via la matrice caméra et matcher au blob le plus proche.
- **Walkers absents du dataset** : `collector.py:_spawn_npcs` ne spawne pas
  encore de piétons (TODO explicite). La classe `walker` reste vide tant que
  c'est pas implémenté.

---

## Variantes / alternatives possibles

Section design — **rien de tout ça n'est implémenté aujourd'hui**, c'est pour
documenter les choix qu'on a écartés ou repoussés. Si tu veux changer
d'approche plus tard, repars d'ici.

### A. Détection couleur des feux en live (au lieu d'un post-process)

**Approche actuelle** : `enrich_labels.py` enrichit les labels **à l'avance**
(4 classes raw → 12 classes finales). YOLO s'entraîne directement sur
`red_light / yellow_light / green_light` comme classes distinctes.

**Alternative live** : YOLO ne détecte qu'**une seule classe** `traffic_light`.
À l'inférence, pour chaque bbox détectée, on crope la zone en RGB et on
applique l'analyse HSV (la même que dans `enrich_labels.py`) pour
déterminer la couleur **au moment de la conduite**, pas à l'entraînement.

**Pros** :
- Modèle YOLO plus simple, jeu de données plus équilibré (le `yellow_light`
  sous-représenté disparaît du problème d'apprentissage)
- Pas besoin de régénérer les labels quand on tweake les seuils HSV : juste
  modifier le code d'inférence
- Si on ajoute des cas (feu clignotant, feu cassé...), c'est un patch de la
  logique HSV, pas un retraining
- La logique HSV est identique training/inférence → moins de risque de drift

**Cons** :
- Coût CPU supplémentaire à chaque frame d'inférence (négligeable : ~1 ms
  pour cropper + HSV sur une dizaine de bboxes)
- YOLO ne profite pas du signal couleur pour mieux discriminer les feux des
  autres objets ronds éclairés
- Sensibilité au glare / contre-jour : si la lentille est éblouie, HSV peut
  hésiter — le modèle YOLO multi-classes l'aurait peut-être appris

**Comment switcher** :
1. Pointer le `data.yaml` YOLO sur `labels_yolo/` (raw 4 classes) au lieu de
   `labels_yolo_color/`
2. Déplacer la fonction `classify_tl_color` de [enrich_labels.py](enrich_labels.py)
   vers le module d'inférence (par exemple `src/perception/yolo/detector.py`)
3. À chaque `DetectedObject` avec `class_name=TRAFFIC_LIGHT`, appeler
   `classify_tl_color` sur le crop de la bbox dans la frame courante,
   ajouter l'info (`tl_state`) dans le `DetectedObject` pour l'IA centrale

Le même raisonnement s'applique à la détection de la **valeur des panneaux**
de vitesse (`classify_speed_sign` est dans le même fichier et porte la même
logique runtime-friendly).

### B. Détection des panneaux de limite de vitesse

**État actuel** : **implémenté** via la même approche que les feux —
détection bbox dans la collecte (composants connexes sur class_id=8) +
post-process pour classifier la valeur (template matching contre 7 templates
synthétiques 30..90). Voir section 9 du chapitre "Problèmes rencontrés".

**Pistes complémentaires non implémentées** :

**B.1 — Colonne `speed_limit_kmh` dans le manifest**

Optionnel mais utile : ajouter une colonne au manifest qui capture la limite
courante via `ego.get_speed_limit()` (CARLA la calcule automatiquement à
partir du panneau le plus proche). L'IA centrale (Frédéric) la consomme
directement comme feature, en complément du label YOLO.

Avantage par rapport au label YOLO seul : marche même quand le panneau
n'est plus visible (la limite reste valable jusqu'au prochain panneau). 5
lignes à ajouter dans [manifest_writer.py](manifest_writer.py) +
[collector.py](collector.py).

**B.2 — Détection live sans templates (à l'inférence)**

Pareil que pour les feux : YOLO ne détecte qu'une classe `traffic_sign`, et
à l'inférence on lit la valeur par template matching live (la fonction
`classify_speed_sign` de [enrich_labels.py](enrich_labels.py) est déjà
prête, il suffit de la déplacer dans le module d'inférence).

Avantage : pas besoin que les 7 classes `speed_*` soient présentes dans le
dataset → modèle plus simple à équilibrer.
