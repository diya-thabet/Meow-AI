# Rapport Complet : Vision-LLM pour la Reconnaissance Faciale des Émotions Composées (FER-CE)

## 1. Contexte et Motivation

### Le Problème : Au-delà des Émotions Simples
La reconnaissance faciale des émotions (FER) est un domaine clé de l'intelligence artificielle, utilisé en psychologie, robotique et interaction homme-machine.

Historiquement, les systèmes classiques (utilisant des réseaux CNN comme ResNet) se concentraient sur **7 émotions basiques** : joie, colère, tristesse, peur, dégoût, surprise et neutre.

Cependant, dans la vraie vie, les humains sont bien plus complexes. Nous ressentons souvent des **émotions composées** (Compound Expressions), c'est-à-dire un mélange de deux émotions simultanées.
Par exemple :
*   **Happily Surprised** (Heureusement surpris) : Yeux grands ouverts (surprise) + Sourire (joie).
*   **Sadly Angry** (Tristement en colère) : Un mélange amer de frustration et de peine.
*   **Fearfully Disgusted** (Dégoûté et effrayé).

Ces émotions mixtes sont très difficiles à détecter pour les IA classiques car les signaux sur le visage (les micro-mouvements musculaires appelés AUs - Action Units) sont subtils et parfois contradictoires.

### La Solution : Vision-LLM
C'est là qu'interviennent les **Vision-LLMs** (Large Vision-Language Models). Ces modèles révolutionnaires ne se contentent pas de "voir" une image, ils peuvent la "comprendre" et en parler comme un humain.

L'objectif de ce projet est d'utiliser un Vision-LLM non seulement pour **classifier** ces émotions complexes (dire "c'est de la tristesse mêlée à de la colère"), mais aussi pour **expliquer pourquoi** (dire "Je vois des sourcils froncés typiques de la colère, mais des yeux tombants qui marquent la tristesse").

---

## 2. Données Utilisées : Le Dataset RAF-CE

Pour ce projet, nous utilisons le jeu de données **RAF-CE** (Real-world Affective Faces - Compound Expressions).

*   **Contenu** : Des images de visages en conditions réelles (pas d'acteurs en studio, mais des vraies photos du web).
*   **Classes** : Il contient **14 catégories** d'émotions composées.
*   **Richesse** : Chaque image possède aussi des annotations sur les mouvements musculaires (Action Units), ce qui nous aide à comprendre la mécanique du visage.

---

## 3. Méthodologie : Notre Pipeline en 3 Couches

Nous avons conçu une approche structurée en trois étapes pour résoudre ce problème.

### Couche 1 : Préparation des Données
Avant de nourrir l'IA, nous devons préparer les images :
1.  **Détection et Recadrage** : On s'assure que le visage est bien au centre.
2.  **Normalisation** : On ajuste les couleurs et la lumière pour que tout soit cohérent.
3.  **Data Augmentation** : On crée des variantes des images (rotations légères, changement de luminosité) pour rendre le modèle plus robuste et éviter qu'il n'apprenne par cœur.

### Couche 2 : Le Cœur Vision-LLM
Ici, nous combinons la vision et le langage.
*   **L'œil (Encodeur Visuel)** : On utilise des modèles puissants comme CLIP ou ViT pour analyser les pixels.
*   **Le Cerveau (LLM)** : On utilise un modèle de langage (comme Vicuna ou LLaMA) pour raisonner.
*   **Le Lien (Q-Former)** : C'est le pont qui traduit ce que l'œil et voit en concepts que le cerveau peut comprendre.

**Objectifs d'apprentissage :**
1.  **Classification** : Prédire correctement l'une des 14 émotions composées.
2.  **Explication** : Générer une phrase qui décrit l'émotion (ex: "La personne semble agréablement surprise, ses yeux sont écarquillés et elle sourit.").

**Technique Avancée : Prompt Engineering Visuel**
Nous guidons le modèle avec des instructions précises, par exemple :
> *"Décris l'état émotionnel et explique quels indices faciaux y contribuent (sourcils, bouche, yeux)."*
Cela force le modèle à être attentif aux détails physiques.

### Couche 3 : Interprétation Multimodale (Comprendre la décision)
Il ne suffit pas que l'IA ait raison, il faut savoir pourquoi.
*   **Visuellement (Grad-CAM)** : Nous générons des cartes de chaleur (heatmaps) pour voir où l'IA regarde. Regarde-t-elle bien la bouche pour un sourire ? Ou se perd-t-elle sur le fond de l'image ?
*   **Linguistiquement** : Nous analysons les phrases générées pour vérifier si elles sont cohérentes avec l'image.

---

## 4. Benchmarks et Résultats Expérimentaux

Nous avons comparé plusieurs approches pour évaluer la performance de notre solution.

### 4.1. Approches Vision-Only (Baselines)
Nous avons d'abord testé des modèles classiques de vision par ordinateur pour établir un score de référence.

1.  **ResNet-50** (Testé dans `Ala's Try` et `Sat Try`)
    *   Architecture robuste et éprouvée.
    *   **Résultat obtenu** : ~51% d'Accuracy.
    *   *Observation* : Le modèle peine à distinguer les nuances subtiles entre deux émotions proches.
2.  **ViT (Vision Transformer)** (Exploré dans `Dhia Try`)
    *   Découpe l'image en "patches" et analyse les relations globales.
    *   Potentiellement plus puissant que ResNet sur des grands datasets, mais demande beaucoup de données pour converger.

### 4.2. Approches Vision-LLM (Notre Innovation)
Nous proposons l'utilisation de modèles multimodaux :
*   **BLIP-2 / LLaVA / Qwen-VL**
*   **Avantages attendus** :
    *   Meilleure compréhension du contexte global.
    *   Capacité à utiliser la connaissance du langage pour désambiguïser des expressions visuelles complexes.
    *   **Score visé** : Supérieur aux 51% du ResNet, avec en prime la capacité d'explication.

### Tableau Comparatif des Performances
| Modèle | Type | Accuracy (Est.) | Avantages | Inconvénients |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet-50** | Vision Pure (CNN) | ~51% | Rapide, Léger | "Boite noire", pas d'explication, confusion sur les classes mixtes |
| **ViT** | Vision Pure (Transformer) | ~53-55% | Vue globale | Lourd à entraîner |
| **Vision-LLM** | Multimodal | **> 60% (Cible)** | **Explicabilité**, Raisonnement, Précision sur les cas ambigus | Très lourd, lent à l'inférence |

---

## 5. Contributions et Livrables

Ce projet apporte trois contributions majeures :
1.  **Un Pipeline Unifié** : Une méthode complète qui aligne l'image et le texte pour l'analyse d'émotions.
2.  **Un Benchmark Comparatif** : Une évaluation claire montrant les limites des modèles classiques (ResNet) face à la complexité des émotions composées.
3.  **L'Explicabilité (XAI)** : Contrairement aux anciens modèles qui donnaient juste un chiffre, notre système explique son raisonnement, ce qui est crucial pour la confiance utilisateur (santé, recrutement, etc.).

### Livrables du Projet
*   📂 **Code Source** : Notebooks propres et organisés.
*   📄 **Rapport Scientifique** : Ce document détaillant toute notre démarche.
*   📊 **Visualisations** : Cartes de chaleur montrant les zones du visage analysées.
*   🤖 **Interface de Démo** (Optionnel) : Pour tester le modèle en direct.

---
*Ce rapport a été généré pour servir de référence centrale au projet FER-CE. Il synthétise les travaux réalisés dans les différents environnements de test (`Ala's Try`, `Dhia Try`, `Sat Try`) et formalise la direction scientifique du projet.*
