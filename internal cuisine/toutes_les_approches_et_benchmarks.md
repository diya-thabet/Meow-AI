# Rapport Technique & Benchmark : Projet Meow-AI (FER-CE)

## 1. Introduction et Vision du Projet

### La Problématique
La reconnaissance faciale des émotions (FER) est traditionnellement limitée à 7 classes basiques (joie, colère, surprise, etc.). Or, l'humain est complexe et exprime souvent des **émotions composées** (*Compound Expressions*).
Exemples :
*   *Happily Surprised* (Heureusement surpris).
*   *Fearfully Disgusted* (Dégoûté et effrayé).

Les modèles classiques (CNN) échouent souvent sur ces cas ambigus car ils cherchent une classe unique sans comprendre le contexte.

### Notre Solution : Vision-LLM & IA Multimodale
Le projet **Meow-AI** propose une rupture technologique en utilisant des **Large Vision-Language Models (Vision-LLMs)**.
Au lieu de simplement classer une image, notre système :
1.  **Observe** les micro-expressions (Action Units).
2.  **Risonne** grâce à un LLM.
3.  **Explique** sa décision en langage naturel.

---

## 2. Infrastructure Technique et Données

### 2.1. Le Dataset RAF-CE
Nous travaillons sur le dataset de référence **RAF-CE (Real-world Affective Faces - Compound Emotions)**.
*   **Volume** : ~4,500 images en conditions réelles ("in the wild").
*   **Complexité** : 14 classes d'émotions mixtes très difficiles à distinguer.
*   **Déséquilibre** : Certaines classes sont très fréquentes (*Happily Surprised*), d'autres très rares (*Fearfully Disgusted*), ce qui pose un défi majeur pour l'entraînement.

### 2.2. Architecture Backend (Dockerisée)
Le projet ne se limite pas à des notebooks. Une architecture backend robuste a été conçue pour le déploiement.
*   **Conteneurisation** : L'application est entièrement Dockerisée (`docker-compose`), garantissant la reproductibilité.
*   **Service** : Le backend expose le modèle via une API, permettant d'envoyer une image et de recevoir la prédiction JSON + l'explication textuelle.
*   **État Actuel** : Le service est fonctionnel en mode développement, avec une intégration prévue des modèles Vision-LLM optimisés (quantization 4-bit/8-bit pour tourner sur des GPU standards).

---

## 3. Analyse des Benchmarks et Expérimentations

Nous avons mené des tests rigoureux sur plusieurs architectures pour prouver la supériorité de notre approche.

### 3.1. Approche 1 : ResNet-50 (Baseline Classique)
*   **Technologie** : CNN traditionnel.
*   **Résultat** : **~51% d'Accuracy**.
*   **Analyse** : Le modèle sature rapidement. Il apprend bien les classes dominantes mais échoue totalement sur les subtilités. C'est une "boite noire" : on ne sait pas pourquoi il se trompe.

### 3.2. Approche 2 : Vision Transformer (ViT) - Analyse Critique
*   **Modèle** : `google/vit-base-patch16-224`
*   **Résultat Final** : **47.91% d'Accuracy** (F1-Score Macro : 0.34).
*   **Analyse de l'Échec** :
    *   **Overfitting Massif** : Dès la 3ème époque, la *Training Loss* descend (le modèle apprend par cœur) mais la *Validation Loss* stagne voire remonte (~1.8).
    *   **Manque de Données** : Les ViT ont besoin de millions d'images pour généraliser correctement ("Inductive Bias" faible par rapport aux CNN). Avec seulement ~4000 images, le ViT n'arrive pas à apprendre des structures robustes.
    *   **Conclusion** : Le ViT *seul* n'est pas adapté à ce dataset sans un pré-entraînement massif ou une augmentation de données extrême.

### 3.3. Approche 3 : Vision-LLM (La Solution Retenue)
C'est ici que Meow-AI innove. En utilisant un modèle pré-entraîné sur des milliards d'images et de textes (comme BLIP-2 ou Qwen-VL), nous contournons le problème du manque de données.
*   **Avantage 1 : Transfer Learning Massif**. Le modèle "sait" déjà à quoi ressemble un visage surpris ou fâché.
*   **Avantage 2 : Raisonnement**. Si l'image est floue, le LLM peut déduire l'émotion par le contexte global, là où ResNet et ViT échouent.
*   **Performance Attendue (Cible)** : Nous visons un score **> 60%** (basé sur la littérature SOTA).
    *   *Note Importante* : Ce score est une **projection théorique**. Les expérimentations sur Vision-LLM sont en cours d'intégration dans le backend Dockerisé. Contrairement au ResNet et au ViT, nous n'avons pas encore de benchmark finalisé sur ce modèle spécifique dans l'environnement actuel.

### Tableau Comparatif des Performances
| Modèle | Type | Accuracy | Statut |
| :--- | :--- | :--- | :--- |
| **ResNet-50** | Vision Pure (CNN) | ~51% | **Validé** (Ala's Try) |
| **ViT** | Vision Pure (Transformer) | 47.91% | **Validé** (Dhia Try) - Échec (Overfitting) |
| **Vision-LLM** | Multimodal | **> 60% (Cible)** | **En cours d'intégration** |

---

## 4. Méthodologie Complète du Pipeline

Pour atteindre nos objectifs, nous avons standardisé le pipeline :

1.  **Pré-traitement Avancé** :
    *   Alignement des visages (MTCNN/RetinaFace).
    *   Oversampling intelligent pour compenser les classes rares (ex: *Fearfully Disgusted*).
2.  **Fine-Tuning LoRA** :
    *   Nous n'entraînons pas tout le modèle (trop lourd). Nous utilisons **LoRA (Low-Rank Adaptation)** pour adapter uniquement une petite partie des paramètres du Vision-LLM. Cela permet d'entraîner le modèle sur un GPU grand public.
3.  **Prompt Engineering Visuel** :
    *   Nous ne demandons pas juste "Quelle est l'émotion ?".
    *   Prompt optimisé : *"Analyse les sourcils, les yeux et la bouche pour déterminer l'émotion composée exacte parmi les 14 classes possibles."*

---

## 5. Conclusion et Perspectives

Le projet Meow-AI démontre que pour des tâches complexes comme les émotions composées, **la force brute (ViT/CNN) ne suffit plus**.

*   **ResNet** atteint un plafond de verre (~51%).
*   **ViT** s'effondre par manque de données (~48%).
*   **Vision-LLM** est la seule voie viable pour dépasser ces limites, en apportant en plus une couche d'explication cruciale pour la confiance utilisateur (XAI).

L'infrastructure Dockerisée est prête pour accueillir ce modèle, transformant une expérience académique en un véritable produit déployable.
