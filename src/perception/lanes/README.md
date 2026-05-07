# `src/perception/lanes/` — Détection de lignes de voie

> **Owner** : Karim Arfaoui
> **Contrat** : [`LaneDetector`](../../interfaces/perception_types.py)

---

## Responsabilité

Détecter dans une image RGB les marquages au sol de la voie courante (ligne de gauche, ligne de droite) et calculer l'**offset** du véhicule par rapport au centre de la voie.

Cette information sert ensuite à l'IA centrale pour rester centrée sur sa voie.

## API publique

```python
class LaneDetector:
    def __init__(self) -> None: ...

    def detect(self, image: np.ndarray) -> LanesInfo:
        """
        Args:
            image: RGB (H, W, 3), dtype uint8.
        Returns:
            LanesInfo avec ligne gauche, ligne droite, offset normalisé.
                Champs None si non détectés.
        """
```

## Approche recommandée : OpenCV pur

Pas besoin de deep learning ici, un pipeline OpenCV classique suffit et tourne en ~5 ms :

1. **Conversion en niveaux de gris** : `cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)`
2. **Réduction du bruit** : `cv2.GaussianBlur(gray, (5, 5), 0)`
3. **Détection de contours** : `cv2.Canny(blurred, 50, 150)`
4. **Région d'intérêt** : masque triangulaire sur la moitié basse de l'image (où sont les lignes)
5. **Détection de segments** : `cv2.HoughLinesP(masked, ...)`
6. **Tri en gauche / droite** : selon la pente des segments
7. **Calcul de l'offset** : distance entre le centre de l'image et le milieu des deux lignes, normalisée

## Squelette suggéré

```
src/perception/lanes/
├── __init__.py
├── detector.py        ← classe LaneDetector (pipeline OpenCV)
├── roi.py             ← définition de la région d'intérêt
└── README.md
```

## Cas particuliers à gérer

- Une seule ligne visible (la voiture est près du bord) → renseigner uniquement le côté détecté, autre = None
- Aucune ligne (intersection, parking) → tous les champs None, `center_offset=None`
- Lignes pointillées → les fragments seront renvoyés par HoughLinesP, prendre la moyenne ou la régression linéaire des segments du même côté

## Performance attendue

- Inference < 10 ms par image sur CPU
- Robustesse aux conditions de lumière variables (pénombre, contre-jour) à valider sur le dataset CARLA

## Validation et benchmarks

Dans [benchmarks/perception/lanes/](../../../benchmarks/perception/lanes/) :

- **`smoke.py`** : vérifier le protocole `LaneDetector`, image vide / uniforme → tous les champs None, image avec deux lignes nettes (synthétique) → offset proche de 0 si centré
- **`benchmark.py`** : précision détection ligne sur dataset annoté, latence

Voir [benchmarks/README.md](../../../benchmarks/README.md) pour la convention.

## Liens

- [Tutoriel OpenCV lane detection (medium classique)](https://medium.com/@galen.ballew/opencv-lanedetection-419361364fc0)
- [`cv2.HoughLinesP` documentation](https://docs.opencv.org/4.x/d6/d10/tutorial_py_houghlines.html)
