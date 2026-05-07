Bonjour,

Nous faisont le projet annuel à 4 : 
FRANCK ZHUANG
FREDERIK HUANG
KARIM ARFAOUI
VICTOR DALET


Notre sujet consiste à faire de la conduite autonome sur un simulateur de conduite (CARLA). L'objectif est qu'une voiture sur Carla puisse se déplacer toute seule en prenant compte des distances avec les autres véhicules, des trajectoires (respecter la route), et également de son déplacement selon sa vitesse et son accélération.

Pour les techno, nous allons utiliser : 
YOLO, détection d'objet sur la route
MIDAS, depth estimation pour prédire une distance associée à un objet détecter
Un algorithme encore non déterminé pour la détection de ligne
Un depth reforcement neural network, nous permettant de faire la décision via un système de score.

Tout se fera en Python via l'API de CARLA (notre simulateur de conduite), avec quelques potentiels bouts de code en C++ ou en rust avec un binding si nous avons des performances trop faibles sur le temps réel final.

Merci.

Victor DALET