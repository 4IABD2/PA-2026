# Journal — `src/ai/`

> **Owner** : Frédéric Huang
> **Format** : entrée datée à chaque session significative (avancée, difficulté résolue, décision, benchmark).
> **À la fin du projet** : un agent synthétisera tous les `JOURNAL.md` pour produire le bilan global.

---

## Template d'entrée

```
## YYYY-MM-DD
**Avancement** :
**Difficultés** :
**Décisions** :
**Benchmarks** :
**Prochaine étape** :
```

---

## 2026-05-07

**Avancement** :
- Architecture du repo entièrement scaffolded (READMEs par module, contrats `src/interfaces/`, structure `src/ai/{models,training,inference}/`, `JOURNAL.md` initiés, `.gitignore` configuré pour les docs perso).
- Spec design du pipeline d'apprentissage rédigé dans `docs/superpowers/specs/2026-05-07-pa2026-cil-pipeline-design.md` (gitignored).
- Setup CARLA sur le serveur Linux multi-GPU démarré : Docker 28.1.1 OK, 2x RTX A6000 disponibles (GPU 1 libre), ports 2000-2002 libres. NVIDIA Container Toolkit pas encore installé sur le serveur (à installer via sudo).

**Difficultés** :
- NVIDIA Container Toolkit absent du serveur → blocage pour lancer le conteneur CARLA en GPU. Installation à venir, nécessite sudo et coordination avec les autres users (restart Docker).

**Décisions** :
- **Approche d'apprentissage révisée : RL hybride** (CIL pre-training puis RL fine-tuning) au lieu de CIL pur. Phase 1 = PilotNet entraîné par imitation sur dataset autopilot CARLA (échéance courte). Phase 2 = fine-tuning du modèle pré-entraîné via PPO ou SAC en boucle CARLA (échéance fin de projet). Justification : on veut une vraie composante RL pour la défense (sujet originel parlait de "deep reinforcement learning") sans le risque d'un RL pur from scratch qui peut ne jamais converger en 3 mois sur la conduite multi-objectifs.
- **Stack** : TensorFlow / Keras (cohérence avec le code existant). Stable-Baselines3 envisagé pour la Phase 2, à confirmer.
- **Caméra POV partagée équipe** : `Location(x=0.5, y=-0.3, z=1.2), Rotation(pitch=-5)`, mode synchrone CARLA 20 FPS, capture 1 frame toutes les 2s.
- **Hardware** : tout tourne sur le serveur Linux multi-GPU (CARLA + collecte + training + démo). Le laptop sert uniquement à coder.

**Benchmarks** : néant (aucun modèle entraîné).

**Prochaine étape** :
1. Installer NVIDIA Container Toolkit sur le serveur, lancer le conteneur `carla-fhuang` en headless sur GPU 1.
2. Tester la connexion Python (`carla.Client('localhost', 2000)`).
3. Implémenter `src/dataset/` pour la collecte (en collaboration avec Franck).
4. Implémenter `src/ai/models/pilotnet.py` + `src/ai/training/train.py`.
5. Première collecte dataset (~10-15k images sur Town01) puis premier entraînement.
