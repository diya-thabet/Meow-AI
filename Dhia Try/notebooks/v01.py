# %%
from google.colab import drive
import pandas as pd
import os
import numpy as np
import re # Nécessaire pour nettoyer les AUs

# 1. Monter le Drive
drive.mount('/content/drive')

print("Google Drive monté avec succès.")

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
# --- CONFIGURATION DU CHEMIN ---

# COLLE LE CHEMIN ICI (garde les guillemets)
# Exemple : '/content/drive/MyDrive/MonProjet/RAF-CE_Annotation'
DATASET_PATH = '/content/drive/MyDrive/Colab Datasets'

# Vérification simple pour voir si le dossier existe
if os.path.exists(DATASET_PATH):
    print(f"Chemin trouvé : {DATASET_PATH}")
    print("Fichiers présents :", os.listdir(DATASET_PATH))
else:
    print(f"ATTENTION : Le chemin '{DATASET_PATH}' n'existe pas. Vérifie le chemin copier-coller.")

# %%
# 1. Mapping des Émotions Composées (RAF-CE)
emotion_map = {
    0: 'Happily surprised', 1: 'Happily disgusted', 2: 'Sadly fearful',
    3: 'Sadly angry', 4: 'Sadly surprised', 5: 'Sadly disgusted',
    6: 'Fearfully angry', 7: 'Fearfully surprised', 8: 'Fearfully disgusted',
    9: 'Angrily surprised', 10: 'Angrily disgusted', 11: 'Disgustedly surprised',
    12: 'Happily fearful', 13: 'Happily sad'
}

# 2. Mapping des Partitions
partition_map = { 0: 'Train', 1: 'Test', 2: 'Validation' }

# 3. Mapping des Action Units (AUs) - Basé sur Wikipédia
au_map_wikipedia = {
    0: 'Neutral face', 1: 'Inner brow raiser', 2: 'Outer brow raiser',
    4: 'Brow lowerer', 5: 'Upper lid raiser', 6: 'Cheek raiser',
    7: 'Lid tightener', 8: 'Lips toward each other', 9: 'Nose wrinkler',
    10: 'Upper lip raiser', 11: 'Nasolabial deepener', 12: 'Lip corner puller',
    13: 'Sharp lip puller', 14: 'Dimpler', 15: 'Lip corner depressor',
    16: 'Lower lip depressor', 17: 'Chin raiser', 18: 'Lip pucker',
    19: 'Tongue show', 20: 'Lip stretcher', 21: 'Neck tightener',
    22: 'Lip funneler', 23: 'Lip tightener', 24: 'Lip pressor',
    25: 'Lips part', 26: 'Jaw drop', 27: 'Mouth stretch', 28: 'Lip suck'
}

# %%
def decode_aus(au_string):
    """Convertit '1 2 4' en liste ['Inner brow raiser', ...]"""
    if not isinstance(au_string, str): return []
    decoded_names = []
    au_ids = re.findall(r'\d+', au_string) # Trouve les chiffres
    for au_id in au_ids:
        name = au_map_wikipedia.get(int(au_id), f"Unknown AU {au_id}")
        decoded_names.append(name)
    return decoded_names

def load_rafce_dataset(base_path):
    try:
        # Chargement Emotion
        path_emo = os.path.join(base_path, 'RAFCE_emolabel.txt')
        df_emo = pd.read_csv(path_emo, sep=r'\s+', header=None, names=['filename', 'emotion_id'])

        # Chargement Partition
        path_part = os.path.join(base_path, 'RAFCE_partition.txt')
        df_part = pd.read_csv(path_part, sep=r'\s+', header=None, names=['filename', 'partition_id'])

        # Chargement AU (Lecture ligne par ligne pour sécurité)
        path_au = os.path.join(base_path, 'RAFCE_AUlabel.txt')
        with open(path_au, 'r') as f:
            au_lines = f.readlines()

        au_data = []
        for line in au_lines:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                au_data.append({'filename': parts[0], 'au_raw': parts[1]})
            else:
                au_data.append({'filename': parts[0], 'au_raw': ''})
        df_au = pd.DataFrame(au_data)

    except FileNotFoundError as e:
        print(f"ERREUR : Fichier introuvable. Vérifie le chemin dans la Cellule 2.\n{e}")
        return None

    # Fusion
    master_df = pd.merge(df_emo, df_part, on='filename')
    master_df = pd.merge(master_df, df_au, on='filename')

    # Application des Mappings
    master_df['emotion_label'] = master_df['emotion_id'].map(emotion_map)
    master_df['dataset_partition'] = master_df['partition_id'].map(partition_map)

    # Décodage des AUs (Nouveau !)
    master_df['au_names'] = master_df['au_raw'].apply(decode_aus)

    return master_df

# %%
# Lancer le chargement
df = load_rafce_dataset(DATASET_PATH)

if df is not None:
    # Nettoyage de l'affichage
    cols = ['filename', 'dataset_partition', 'emotion_label', 'au_names', 'au_raw']
    df = df[cols]

    print("✅ Données chargées avec succès !")
    print(f"Nombre total d'images : {len(df)}")

    print("\n--- Aperçu des 5 premières lignes ---")
    display(df.head())

    print("\n--- Répartition Train/Test ---")
    print(df['dataset_partition'].value_counts())

# %%
import matplotlib.pyplot as plt
import seaborn as sns

def check_dataset_balance(dataframe):
    if dataframe is None:
        print("Erreur : Le DataFrame n'est pas chargé.")
        return

    # 1. Calculer les comptes et pourcentages
    counts = dataframe['emotion_label'].value_counts()
    percentages = dataframe['emotion_label'].value_counts(normalize=True) * 100

    # Créer un petit tableau récapitulatif
    balance_df = pd.DataFrame({'Nombre': counts, 'Pourcentage (%)': percentages})

    print("--- Répartition des classes (Émotions) ---")
    display(balance_df)

    # 2. Vérification mathématique du déséquilibre
    max_count = counts.max()
    min_count = counts.min()
    ratio = max_count / min_count

    print(f"\nClasse majoritaire : {max_count} images")
    print(f"Classe minoritaire : {min_count} images")
    print(f"Ratio de déséquilibre : 1 pour {ratio:.2f}")

    if ratio > 2.0:
        print("\n⚠️ CONCLUSION : Le dataset est DÉSÉQUILIBRÉ (Unbalanced).")
        print("Le modèle risque de favoriser les classes majoritaires.")
    else:
        print("\n✅ CONCLUSION : Le dataset est plutôt ÉQUILIBRÉ (Balanced).")

    # 3. Visualisation Graphique
    plt.figure(figsize=(12, 6))
    sns.barplot(x=counts.index, y=counts.values, palette='viridis')
    plt.title('Distribution des Émotions dans le Dataset RAF-CE')
    plt.xlabel('Émotion Composée')
    plt.ylabel("Nombre d'images")
    plt.xticks(rotation=45, ha='right') # Rotation des étiquettes pour lisibilité
    plt.tight_layout()
    plt.show()

# Exécuter la vérification
check_dataset_balance(df)

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
import numpy as np

# --- ÉTAPE 1 : RÉPARATION (Fix du KeyError) ---
# On recrée la colonne 'emotion_id' à partir des noms, au cas où elle manque
name_to_id = {v: k for k, v in emotion_map.items()} # Inversion du dictionnaire
if 'emotion_id' not in df.columns:
    df['emotion_id'] = df['emotion_label'].map(name_to_id)

# --- ÉTAPE 2 : TABLEAU DE RÉPARTITION COMPLET ---
# On utilise un 'crosstab' pour croiser les Émotions avec les Partitions
distribution_table = pd.crosstab(df['emotion_label'], df['dataset_partition'])

# On ajoute une colonne 'Total' pour trier
distribution_table['Total'] = distribution_table.sum(axis=1)

# On trie du plus fréquent au moins fréquent
distribution_table = distribution_table.sort_values('Total', ascending=False)

print("--- Répartition DÉTAILLÉE (Train / Test / Validation) ---")
display(distribution_table)

# --- ÉTAPE 3 : VISUALISATION GRAPHIQUE ---
# On retire la colonne total pour le graphique pour ne pas fausser l'échelle
viz_data = distribution_table.drop(columns=['Total'])
viz_data.plot(kind='bar', stacked=True, figsize=(14, 7), color=['green', 'orange', 'blue'])
plt.title("Répartition Train/Test/Val par Émotion")
plt.ylabel("Nombre d'images")
plt.xlabel("Émotion Composée")
plt.xticks(rotation=45, ha='right')
plt.legend(title='Partition')
plt.tight_layout()
plt.show()

# --- ÉTAPE 4 : CALCUL DES POIDS (Class Weights) CORRIGÉ ---
print("\n--- Recalcul des Poids pour l'entraînement ---")
df_train = df[df['dataset_partition'] == 'Train']
y_train = df_train['emotion_id'].values

# Vérification qu'on a bien des données
if len(y_train) > 0:
    classes_present = np.unique(y_train)
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=classes_present,
        y=y_train
    )

    class_weights_dict = dict(zip(classes_present, weights))

    # Affichage propre
    print(f"{'ID':<5} | {'Émotion':<25} | {'Poids (Weight)':<10}")
    print("-" * 45)
    for emo_id in sorted(class_weights_dict.keys()):
        label = emotion_map[emo_id]
        weight = class_weights_dict[emo_id]
        print(f"{emo_id:<5} | {label:<25} | {weight:.4f}")
else:
    print("Erreur : Aucune donnée d'entraînement trouvée.")

# %%
import os
import cv2
import matplotlib.pyplot as plt

def verify_images_link(dataframe, folder_path):
    print(f"🔍 Vérification dans le dossier : {folder_path}")

    found = 0
    missing = 0

    # On prend un échantillon de 5 images pour tester
    sample_files = dataframe['filename'].head(5).values

    plt.figure(figsize=(15, 5))

    for i, filename in enumerate(sample_files):
        # --- MODIFICATION ICI ---
        # 1. On nettoie le nom venant du fichier texte (on enlève .jpg s'il est là)
        base_name = filename.replace('.jpg', '').replace('.jpeg', '').strip()

        # 2. On reconstruit le nom tel qu'il est sur le disque
        # Transformation : "train_0001" -> "train_0001_aligned.jpg"
        aligned_filename = f"{base_name}_aligned.jpg"

        # 3. Chemin complet
        full_path = os.path.join(folder_path, aligned_filename)
        # ------------------------

        if os.path.exists(full_path):
            found += 1
            # Lecture et affichage
            try:
                img = cv2.imread(full_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    plt.subplot(1, 5, i+1)
                    plt.imshow(img)
                    plt.title(base_name) # On affiche le nom court
                    plt.axis('off')
                else:
                    print(f"⚠️ Image trouvée mais illisible : {aligned_filename}")
            except Exception as e:
                print(f"Erreur de lecture : {e}")
        else:
            missing += 1
            print(f"❌ Manquant : {aligned_filename} (Cherché à : {full_path})")

    plt.show()

    if missing == 0:
        print(f"\n✅ SUCCÈS TOTAL : 5/5 images trouvées !")
        print("La logique de nommage (ajout de '_aligned') fonctionne.")
    else:
        print(f"\n⚠️ PROBLÈME : {missing} images manquantes.")
        print("Vérifie si le dossier contient bien des fichiers finissant par '_aligned.jpg'")

# Lancer la vérification avec le dossier décompressé
# Assure-toi que FINAL_IMAGE_PATH est bien défini (suite à la décompression précédente)
if 'FINAL_IMAGE_PATH' in locals():
    verify_images_link(df, FINAL_IMAGE_PATH)
else:
    print("La variable FINAL_IMAGE_PATH n'existe pas. Relance la cellule de décompression (Unzip).")

# %%
import zipfile
import os

# 1. Définition des chemins
zip_path = '/content/drive/MyDrive/Colab Datasets/aligned.zip'
local_extract_path = '/content/rafce_images_unzipped' # Dossier local rapide

# 2. Décompression (Unzip)
if not os.path.exists(local_extract_path):
    print(f"📂 Décompression de '{zip_path}' en cours...")
    print("Cela peut prendre quelques secondes/minutes selon la taille...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(local_extract_path)
        print("✅ Décompression terminée avec succès !")
    except FileNotFoundError:
        print("❌ ERREUR : Le fichier zip est introuvable au chemin indiqué.")
else:
    print("⚡ Le dossier existe déjà, on passe la décompression.")

# 3. Trouver le vrai dossier des images
# Parfois le zip contient un sous-dossier (ex: 'aligned/image.jpg')
# On regarde ce qu'il y a dedans
contents = os.listdir(local_extract_path)
print(f"\nContenu du dossier extrait : {contents}")

# Déterminer le bon chemin final
# Si le zip contient un dossier 'aligned', on rentre dedans. Sinon on reste à la racine.
if 'aligned' in contents and os.path.isdir(os.path.join(local_extract_path, 'aligned')):
    FINAL_IMAGE_PATH = os.path.join(local_extract_path, 'aligned')
else:
    # Si les images sont en vrac ou dans un autre dossier, on prend la racine
    # (Tu pourras ajuster ici selon ce que le print affiche)
    FINAL_IMAGE_PATH = local_extract_path

print(f"👉 Chemin final des images configuré sur : {FINAL_IMAGE_PATH}")

# 4. Relancer la vérification visuelle avec le nouveau chemin
# On réutilise la fonction verify_images_link définie juste avant
try:
    verify_images_link(df, FINAL_IMAGE_PATH)
except NameError:
    print("La fonction verify_images_link n'est pas définie. Relance la cellule précédente.")

# %%
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# 1. Définition des Transformations (Prétraitement)
# Les modèles comme ResNet attendent généralement du 224x224
data_transforms = {
    'Train': transforms.Compose([
        transforms.Resize((224, 224)),      # Redimensionner
        transforms.RandomHorizontalFlip(),  # Augmentation : retourner l'image (miroir)
        transforms.RandomRotation(10),      # Augmentation : légère rotation
        transforms.ToTensor(),              # Convertir en Tensor (0-1)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalisation standard ImageNet
    ]),
    'Test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'Validation': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 2. La Classe Dataset Personnalisée
class RAFCEDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): Le tableau contenant les noms de fichiers et labels.
            root_dir (string): Le dossier contenant les images dézippées.
            transform (callable, optional): Les transformations à appliquer.
        """
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

        # On crée un mapping inverse pour convertir les labels (0, 1...) en Tensors
        # Assure-toi que la colonne 'emotion_id' existe (on l'a créée plus tôt)
        self.labels = self.dataframe['emotion_id'].values
        self.filenames = self.dataframe['filename'].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # 1. Récupérer le nom de base
        img_name = self.filenames[idx]

        # 2. Gérer le suffixe "_aligned" comme on l'a validé ensemble
        base_name = img_name.replace('.jpg', '').replace('.jpeg', '').strip()
        full_img_name = f"{base_name}_aligned.jpg"
        img_path = os.path.join(self.root_dir, full_img_name)

        # 3. Charger l'image
        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, FileNotFoundError):
            # Si une image est corrompue, on crée une image noire pour ne pas planter l'entraînement
            print(f"⚠️ Erreur chargement: {img_path}. Utilisation image noire.")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        # 4. Appliquer les transformations
        if self.transform:
            image = self.transform(image)

        # 5. Récupérer le label
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, label

print("Classe RAFCEDataset définie avec succès.")

# %%
# --- ÉTAPE 1 : RÉPARATION DU TABLEAU (Fix KeyError) ---
# On s'assure que la colonne 'emotion_id' existe bien avant de continuer
# On inverse le dictionnaire pour retrouver l'ID à partir du nom
name_to_id = {v: k for k, v in emotion_map.items()}

if 'emotion_id' not in df.columns:
    print("🔧 Colonne 'emotion_id' manquante. Réparation en cours...")
    df['emotion_id'] = df['emotion_label'].map(name_to_id)
else:
    print("✅ La colonne 'emotion_id' est bien présente.")

# --- ÉTAPE 2 : SÉPARATION DU DATAFRAME ---
# On utilise .copy() pour éviter des avertissements de Pandas
df_train = df[df['dataset_partition'] == 'Train'].copy()
df_test = df[df['dataset_partition'] == 'Test'].copy()
df_val = df[df['dataset_partition'] == 'Validation'].copy()

print("-" * 30)
print(f"Images Train: {len(df_train)}")
print(f"Images Test:  {len(df_test)}")
print(f"Images Val:   {len(df_val)}")
print("-" * 30)

# --- ÉTAPE 3 : CRÉATION DES CHARGEURS (DataLoaders) ---
BATCH_SIZE = 32

# Création des Datasets
train_dataset = RAFCEDataset(df_train, FINAL_IMAGE_PATH, transform=data_transforms['Train'])
test_dataset = RAFCEDataset(df_test, FINAL_IMAGE_PATH, transform=data_transforms['Test'])
val_dataset = RAFCEDataset(df_val, FINAL_IMAGE_PATH, transform=data_transforms['Validation'])

# Création des DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print("✅ DataLoaders prêts et chargés !")

# Test final pour vérifier que les Tensors sortent bien
try:
    images, labels = next(iter(train_loader))
    print(f"Succès ! Forme d'un batch d'images : {images.shape}") # Doit afficher [32, 3, 224, 224]
    print(f"Succès ! Forme d'un batch de labels : {labels.shape}")
except Exception as e:
    print(f"❌ Erreur lors du chargement du batch : {e}")

# %%
import torch.nn as nn
from torchvision import models

# 1. Configuration du Matériel (GPU)
# Si tu as activé le T4, 'cuda' sera utilisé. Sinon 'cpu'.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🧠 Le modèle sera entraîné sur : {device}")

# 2. Télécharger le modèle pré-entraîné (ResNet18)
print("Chargement de ResNet18...")
model = models.resnet18(weights='IMAGENET1K_V1')

# 3. Modifier la dernière couche (Fully Connected - fc)
# ResNet original a 1000 sorties. Nous en voulons 14.
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 14) # 14 classes

# Envoyer le modèle sur le GPU
model = model.to(device)

# 4. Configuration de la Loss Function avec les POIDS (Class Weights)
# Rappel : Nous avons un dictionnaire 'class_weights_dict'.
# Nous devons le transformer en Tensor trié (de 0 à 13) pour PyTorch.

# On s'assure que les poids sont dans le bon ordre (0, 1, 2... 13)
weights_list = []
# Note : Si la variable class_weights_dict n'existe plus, on met des poids par défaut (1.0)
if 'class_weights_dict' in locals():
    print("✅ Utilisation des poids calculés pour le déséquilibre.")
    for i in range(14):
        weights_list.append(class_weights_dict.get(i, 1.0)) # 1.0 par défaut si erreur
else:
    print("⚠️ Attention : Poids introuvables. Utilisation de poids neutres.")
    weights_list = [1.0] * 14

# Conversion en Tensor PyTorch et envoi sur le GPU
class_weights_tensor = torch.FloatTensor(weights_list).to(device)

# La fonction de perte (Loss) qui guidera l'apprentissage
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# 5. L'Optimiseur (Celui qui ajuste les neurones)
# Adam est un excellent choix standard. Learning rate bas (0.0001) pour ne pas tout casser.
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

print("\n🚀 Modèle prêt à l'emploi !")
print(f"Structure de la dernière couche : {model.fc}")

# %%
#####################################################################
#######################################################################
##############   RESNET 50 EPOCH 93ad LILA kemla #####################
import time
import copy
import sys
import torch

def train_night_mode(model, criterion, optimizer, num_epochs=50):
    since = time.time()

    # On garde une copie en mémoire RAM
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"🚀 Démarrage de l'entraînement de nuit pour {num_epochs} époques...")

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['Train', 'Validation']:
            if phase == 'Train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Boucle sur les batchs
            for i, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Affichage de progression fluide
                if i % 10 == 0:
                    sys.stdout.write(f'\rPhase: {phase} | Batch {i}/{len(dataloader)} | Loss: {loss.item():.4f}')
                    sys.stdout.flush()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'\n✅ Fin {phase} : Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Historique
            if phase == 'Train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            # --- SAUVEGARDE DU MEILLEUR MODÈLE (Sur le Disque) ---
            if phase == 'Validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # Sauvegarde physique immédiate
                torch.save(model.state_dict(), 'best_model_night.pth')
                print(f"🌟 Record battu ({best_acc:.4f}) ! Sauvegardé dans 'best_model_night.pth'")

        # --- SAUVEGARDE DE SÉCURITÉ (Checkpoint) ---
        # Toutes les 10 époques, on fait une sauvegarde au cas où
        if (epoch + 1) % 10 == 0:
            ckpt_name = f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), ckpt_name)
            print(f"💾 Checkpoint de sécurité créé : {ckpt_name}")

    time_elapsed = time.time() - since
    print(f'\n🌙 Entraînement terminé en {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Meilleure Accuracy Finale: {best_acc:.4f}')

    # On charge les meilleurs poids pour finir
    model.load_state_dict(best_model_wts)
    return model, history

# --- LANCEMENT ---
# Assure-toi que 'model', 'criterion' et 'optimizer' sont définis dans les cellules d'avant
trained_model, history = train_night_mode(model, criterion, optimizer, num_epochs=50)

# %%
import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# --- 1. CONFIGURATION BASÉE SUR TON MAPPING ---
# On convertit ton dictionnaire en une liste ordonnée pour l'affichage
# (L'ordre 0, 1, 2... est crucial pour que les labels correspondent aux prédictions)
class_names = [emotion_map[i] for i in range(len(emotion_map))]
num_classes = len(class_names)

print(f"✅ Configuration détectée : {num_classes} émotions composées (RAF-CE).")
print(f"📋 Classes : {class_names}")

# --- 2. RECONSTRUCTION DU MODÈLE ---
FILENAME = 'best_model_night.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n🔄 Reconstruction de l'architecture ResNet18 pour {num_classes} sorties...")
trained_model = models.resnet18(weights=None)
num_ftrs = trained_model.fc.in_features
# C'est ici que la magie opère : on force 14 sorties pour matcher tes 14 émotions
trained_model.fc = nn.Linear(num_ftrs, num_classes)

# --- 3. CHARGEMENT DES POIDS ---
try:
    print(f"📥 Chargement des poids depuis '{FILENAME}'...")
    trained_model.load_state_dict(torch.load(FILENAME, map_location=device))
    trained_model = trained_model.to(device)
    trained_model.eval()
    print("✅ Modèle chargé avec succès !")
except RuntimeError as e:
    print(f"❌ ERREUR D'ARCHITECTURE : {e}")
    print("Le fichier sauvegardé ne correspond pas à 14 classes. As-tu entraîné sur 7 ou 14 ?")
    raise
except FileNotFoundError:
    print(f"❌ ERREUR : Le fichier '{FILENAME}' est introuvable.")
    raise

# --- 4. PRÉDICTIONS ---
def get_all_predictions(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    print("🔮 Calcul des prédictions sur le Test Set... (Patientez)")

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)

if 'test_loader' in locals():
    y_pred, y_true = get_all_predictions(trained_model, test_loader)

    # --- 5. VISUALISATION (MATRICE DE CONFUSION) ---
    plt.figure(figsize=(14, 12)) # Plus grand car il y a 14 classes
    cm = confusion_matrix(y_true, y_pred)

    # Heatmap avec les vrais noms
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    plt.xlabel('Prédiction du Modèle')
    plt.ylabel('Vraie Émotion')
    plt.title('Matrice de Confusion : Émotions Composées (RAF-CE)')
    plt.xticks(rotation=45, ha='right') # Rotation pour lire les noms longs
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # --- 6. RAPPORT DÉTAILLÉ ---
    print("\n--- 📝 RAPPORT DE PERFORMANCE PAR CLASSE ---")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

else:
    print("❌ ERREUR : 'test_loader' n'est pas défini. Relance la cellule des DataLoaders.")

# %%
import random
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 1. Fonction pour faire une prédiction sur une seule image
def predict_random_image(dataset, model):
    # Choisir un index au hasard
    idx = random.randint(0, len(dataset) - 1)

    # Récupérer l'image et le vrai label
    image_tensor, label_idx = dataset[idx]

    # Préparer l'image pour l'affichage (dé-normalisation)
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    image_display = inv_normalize(image_tensor)
    image_display = image_display.permute(1, 2, 0).numpy()
    image_display = np.clip(image_display, 0, 1) # S'assurer que les pixels sont entre 0 et 1

    # --- PRÉDICTION ---
    model.eval()
    input_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, 1)

    # --- CORRECTION ICI : Utilisation de 'emotion_map' ---
    # On utilise ton dictionnaire emotion_map défini plus tôt
    true_emotion = emotion_map[label_idx.item()]
    pred_emotion = emotion_map[pred_idx.item()]

    conf_score = confidence.item() * 100

    # --- AFFICHAGE ---
    plt.figure(figsize=(6, 6))
    plt.imshow(image_display)
    plt.axis('off')

    # Couleur du titre : Vert si correct, Rouge si erreur
    title_color = 'green' if true_emotion == pred_emotion else 'red'

    plt.title(f"Vrai: {true_emotion}\nPrédiction: {pred_emotion}\nConfiance: {conf_score:.1f}%",
              color=title_color, fontsize=14, fontweight='bold')
    plt.show()

# 2. Lancer le test plusieurs fois
print("🎲 Test sur 3 images au hasard du jeu de Test :")
predict_random_image(test_dataset, trained_model)
predict_random_image(test_dataset, trained_model)
predict_random_image(test_dataset, trained_model)

# %%
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms

# --- 1. DÉFINITION EN DUR DE LA LISTE DES ÉMOTIONS ---
# On s'assure que l'ordre est le même que celui utilisé lors de l'entraînement
# (Basé sur votre rapport de classification précédent)
LABELS_LIST = [
    'Happily surprised', 'Happily disgusted', 'Sadly fearful', 'Sadly angry',
    'Sadly surprised', 'Sadly disgusted', 'Fearfully angry', 'Fearfully surprised',
    'Fearfully disgusted', 'Angrily surprised', 'Angrily disgusted',
    'Disgustedly surprised', 'Happily fearful', 'Happily sad'
]

def predict_random_image_final(dataset, model):
    # Choisir un index au hasard
    idx = random.randint(0, len(dataset) - 1)

    # Récupérer l'image, le label ID et le nom du fichier
    image_tensor, label_idx = dataset[idx]
    filename = dataset.filenames[idx]

    # Préparation pour affichage
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    image_display = inv_normalize(image_tensor)
    image_display = image_display.permute(1, 2, 0).numpy()
    image_display = np.clip(image_display, 0, 1)

    # --- PRÉDICTION ---
    model.eval()
    input_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, 1)

    # --- TRADUCTION ID -> TEXTE ---
    # On utilise la liste définie au début de la cellule
    try:
        true_emotion = LABELS_LIST[label_idx.item()]
        pred_emotion = LABELS_LIST[pred_idx.item()]
    except IndexError:
        true_emotion = f"ID inconnu ({label_idx.item()})"
        pred_emotion = f"ID inconnu ({pred_idx.item()})"

    conf_score = confidence.item() * 100

    # --- AFFICHAGE ---
    plt.figure(figsize=(7, 7))
    plt.imshow(image_display)
    plt.axis('off')

    color = 'green' if true_emotion == pred_emotion else 'red'

    # Affichage du nom de fichier pour vérification manuelle
    plt.text(0, -25, f"Fichier : {filename}", fontsize=11,
             backgroundcolor='#f0f0f0', color='black')

    plt.title(f"Vrai : {true_emotion}\nPrédiction : {pred_emotion} ({conf_score:.1f}%)",
              color=color, fontsize=14, fontweight='bold')
    plt.show()

# --- LANCER LE TEST ---
print("🎲 Test Final (Version corrigée) :")
predict_random_image_final(test_dataset, trained_model)
predict_random_image_final(test_dataset, trained_model)
predict_random_image_final(test_dataset, trained_model)

# %%
import torch
from google.colab import files

# 1. Sauvegarder les poids du modèle dans Colab
model_save_name = 'emotion_resnet18_cpu.pth'
path = F"/content/{model_save_name}"
torch.save(trained_model.state_dict(), path)
print(f"✅ Modèle sauvegardé temporairement sous : {path}")

# 2. Télécharger le fichier sur ton ordinateur
print("📥 Téléchargement en cours...")
files.download(path)

# %%
from google.colab import files
from PIL import Image
import io

# 1. Fonction de prédiction sur une image externe
def predict_external_image(model):
    print("📸 Envoie une image (JPG/PNG) pour tester l'IA :")
    uploaded = files.upload()

    for fn in uploaded.keys():
        # Charger l'image
        image_data = uploaded[fn]
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # Transformer l'image (Même transformation que pour l'entraînement)
        # Redimensionner en 224x224 est CRUCIAL
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image_tensor = transform(image).unsqueeze(0).to(device)

        # Prédiction
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)

        pred_label = LABELS_LIST[pred_idx.item()]

        # Affichage
        plt.figure(figsize=(6,6))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Je vois : {pred_label}\nConfiance : {confidence.item()*100:.1f}%", fontsize=14)
        plt.show()

# 2. Lancer
predict_external_image(trained_model)

# %%
import os

def train_model_with_checkpoints(model, criterion, optimizer, num_epochs=50):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['Train', 'Validation']:
            if phase == 'Train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'✅ {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Enregistrement historique
            if phase == 'Train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            # Sauvegarde du meilleur modèle
            if phase == 'Validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # --- SÉCURITÉ : SAUVEGARDE INTERMÉDIAIRE TOUTES LES 5 ÉPOQUES ---
        if (epoch + 1) % 5 == 0:
            ckpt_name = f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), ckpt_name)
            print(f"💾 Sauvegarde de sécurité effectuée : {ckpt_name}")

    time_elapsed = time.time() - since
    print(f'\nEntraînement terminé en {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Meilleure Accuracy: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, history

# Lancer la nuit
# trained_model, history = train_model_with_checkpoints(model, criterion, optimizer, num_epochs=50)

# %%
!pip install grad-cam

# %%
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import numpy as np

# 1. Configuration de Grad-CAM pour ResNet
# ResNet est composé de 4 blocs principaux (layer1, layer2, layer3, layer4).
# On regarde généralement la dernière couche de convolution (layer4)
target_layers = [trained_model.layer4[-1]]

# Création de l'objet CAM
cam = GradCAM(model=trained_model, target_layers=target_layers)

def visualize_heatmap(dataset, index):
    # Récupérer l'image brute et le tensor
    img_tensor, label_idx = dataset[index]
    filename = dataset.filenames[index]

    # Préparer l'image pour Grad-CAM (Tensor 4D : Batch, Channel, Height, Width)
    input_tensor = img_tensor.unsqueeze(0).to(device)

    # --- GÉNÉRATION DE LA HEATMAP ---
    # On demande : "Montre-moi les pixels importants pour la classe prédite"
    grayscale_cam = cam(input_tensor=input_tensor)
    grayscale_cam = grayscale_cam[0, :] # On prend la première image du batch

    # Préparation de l'image de fond pour l'affichage (RGB normalisé entre 0 et 1)
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    rgb_img = inv_normalize(img_tensor).permute(1, 2, 0).cpu().numpy()
    rgb_img = np.clip(rgb_img, 0, 1) # Sécurité

    # Superposition (Image originale + Heatmap)
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # --- AFFICHAGE ---
    true_label = LABELS_LIST[label_idx]

    plt.figure(figsize=(10, 5))

    # Image originale
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.title(f"Originale : {filename}\n({true_label})")
    plt.axis('off')

    # Heatmap
    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title("Ce que l'IA regarde (Zones Rouges)")
    plt.axis('off')

    plt.show()

# Test sur quelques images
print("🕵️ Analyse XAI en cours...")
visualize_heatmap(test_dataset, 10) # Change le chiffre pour voir d'autres images
visualize_heatmap(test_dataset, 37)

# %%
!pip install transformers

# %%
from transformers import BlipProcessor, BlipForConditionalGeneration

print("🤖 Chargement du Vision-LLM (BLIP)...")
# On utilise la version "Base" qui est plus légère
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model_llm = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Envoyer sur CPU (ou GPU si dispo, mais attention à la mémoire avec 2 modèles chargés !)
device_llm = torch.device("cpu")
model_llm.to(device_llm)
print("✅ Vision-LLM chargé !")

# %%
def generate_caption(dataset, index):
    # Récupérer l'image brute
    img_tensor, label_idx = dataset[index]

    # Pour BLIP, on a besoin de l'image "Pil" (non tensorisée/normalisée)
    # Donc on la recharge proprement pour être sûr
    img_path = dataset.filenames[index]
    # On doit retrouver le chemin complet...
    # Astuce : on suppose que 'test_dataset.root_dir' est bon, sinon on bricole
    # Ici, on va faire l'inverse de la normalisation pour récupérer l'image visuelle
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    pil_image = transforms.ToPILImage()(inv_normalize(img_tensor))

    true_label = LABELS_LIST[label_idx]

    # --- PROMPT ENGINEERING VISUEL ---
    # On demande au modèle de décrire le visage et l'expression
    text_prompt = "a photograph of a face expressing emotion, detailed description of the eyes and mouth: "

    inputs = processor(pil_image, text_prompt, return_tensors="pt").to(device_llm)

    # Génération
    out = model_llm.generate(**inputs, max_new_tokens=50)
    description = processor.decode(out[0], skip_special_tokens=True)

    # Affichage
    plt.figure(figsize=(4, 4))
    plt.imshow(pil_image)
    plt.axis('off')
    plt.title(f"Label Réel : {true_label}")
    plt.show()

    print(f"💬 Description du Vision-LLM :\n➡️ '{description}'")

# Test
print("Test du Vision-LLM...")
generate_caption(test_dataset, 10)
generate_caption(test_dataset, 233)

# %%
import re

# 1. Ton dictionnaire Wikipedia (Déjà parfait)
au_map_wikipedia = {
    '1': 'Inner brow raiser', '2': 'Outer brow raiser',
    '4': 'Brow lowerer', '5': 'Upper lid raiser', '6': 'Cheek raiser',
    '7': 'Lid tightener', '8': 'Lips toward each other', '9': 'Nose wrinkler',
    '10': 'Upper lip raiser', '11': 'Nasolabial deepener', '12': 'Lip corner puller',
    '13': 'Sharp lip puller', '14': 'Dimpler', '15': 'Lip corner depressor',
    '16': 'Lower lip depressor', '17': 'Chin raiser', '18': 'Lip pucker',
    '19': 'Tongue show', '20': 'Lip stretcher', '21': 'Neck tightener',
    '22': 'Lip funneler', '23': 'Lip tightener', '24': 'Lip pressor',
    '25': 'Lips part', '26': 'Jaw drop', '27': 'Mouth stretch', '28': 'Lip suck',
    '29': 'Jaw thrust', '43': 'Eyes closed'
}

# 2. Fonction pour nettoyer les codes (ex: "L12" -> "12")
def clean_au_code(code):
    # Enlève les lettres (L, R, T, B) et garde juste les chiffres
    return re.sub("[^0-9]", "", code)

# 3. Chargement du fichier RAFCE_AUlabel.txt
# ATTENTION : Assure-toi que le fichier est bien dans Colab
au_file_path = '/content/drive/MyDrive/Colab Datasets/RAFCE_AUlabel.txt' # Vérifie ce chemin !
# Si tu as unzippé ailleurs, adapte le chemin.

image_to_au_text = {}

try:
    with open(au_file_path, 'r') as f:
        lines = f.readlines()
        print(f"📂 Fichier AU chargé : {len(lines)} lignes trouvées.")

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 2: continue # Ligne vide ou sans label

            filename = parts[0]
            au_string = parts[1]

            # Gestion des "null"
            if "null" in au_string:
                desc = "Aucune activation musculaire détectée"
            else:
                # Séparer par '+' et nettoyer
                au_codes = au_string.split('+')
                descriptions = []
                for code in au_codes:
                    clean_code = clean_au_code(code)
                    if clean_code in au_map_wikipedia:
                        descriptions.append(au_map_wikipedia[clean_code])

                desc = ", ".join(descriptions)

            # On stocke : "0001.jpg" -> "Inner brow raiser, Lips part"
            image_to_au_text[filename] = desc

    print("✅ Mapping AU terminé ! Exemple pour 0001.jpg :")
    print(f"👉 {image_to_au_text.get('0001.jpg', 'Introuvable')}")

except FileNotFoundError:
    print(f"❌ ERREUR : Le fichier {au_file_path} est introuvable.")
    print("Vérifie où se trouve 'RAFCE_AUlabel.txt' dans tes fichiers à gauche.")

# %%
def analyze_image_full(dataset, index):
    # 1. Récupération des données
    img_tensor, label_idx = dataset[index]
    # Le nom de fichier dans le dataset est peut-être "test_0001.jpg"
    # Le fichier AU utilise "0001.jpg". On doit harmoniser.
    full_filename = dataset.filenames[index]
    simple_filename = full_filename.split('_')[-1] # Garde juste "0001.jpg"

    true_label = LABELS_LIST[label_idx]

    # Récupérer l'explication AU
    au_explanation = image_to_au_text.get(simple_filename, "Données AU non disponibles")

    # 2. Prédiction ResNet
    model.eval()
    input_tensor = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, 1)

    pred_label = LABELS_LIST[pred_idx.item()]

    # 3. Vision-LLM (BLIP) - VERSION CORRIGÉE
    # On reconstruit l'image PIL
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    pil_image = transforms.ToPILImage()(inv_normalize(img_tensor))

    # ASTUCE : On donne juste le début de phrase pour le guider, mais pas trop
    inputs = processor(pil_image, "a close-up photo of a face looking", return_tensors="pt").to(device_llm)
    out = model_llm.generate(**inputs, max_new_tokens=20)
    blip_desc = processor.decode(out[0], skip_special_tokens=True)

    # 4. Affichage
    plt.figure(figsize=(8, 8))
    plt.imshow(pil_image)
    plt.axis('off')

    # Titre coloré
    color = 'green' if true_label == pred_label else 'red'
    plt.title(f"Vrai: {true_label} | Prédit: {pred_label} ({conf.item()*100:.0f}%)",
              color=color, fontweight='bold', fontsize=14)

    plt.show()

    # Rapport Textuel
    print("-" * 50)
    print(f"📄 Fichier : {simple_filename}")
    print(f"🧬 Anatomie (AUs réels) : \n   ➡️ {au_explanation}")
    print(f"🤖 Vision-LLM (Ce que voit l'IA) : \n   ➡️ {blip_desc}")
    print("-" * 50)

# --- TEST ---
print("🔬 Analyse Complète :")
# Essaie quelques index différents pour trouver des images intéressantes
analyze_image_full(test_dataset, 12)
analyze_image_full(test_dataset, 45)

# %%
def analyze_image_focused(dataset, index):
    # 1. Récupération Image & Labels
    img_tensor, label_idx = dataset[index]

    # Gestion du nom de fichier pour les AUs
    full_filename = dataset.filenames[index]
    simple_filename = full_filename.split('_')[-1] # "0073.jpg"

    true_label = LABELS_LIST[label_idx]
    au_explanation = image_to_au_text.get(simple_filename, "Non disponible")

    # 2. Préparation Image PIL (Pour BLIP)
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    pil_image = transforms.ToPILImage()(inv_normalize(img_tensor))

    # 3. INTERROGATOIRE DU VISION-LLM (Prompt Engineering)
    # On force le modèle à regarder des zones spécifiques

    # Question A: L'émotion globale
    inputs_emo = processor(pil_image, "the facial expression is", return_tensors="pt").to(device_llm)
    out_emo = model_llm.generate(**inputs_emo, max_new_tokens=10)
    desc_emo = processor.decode(out_emo[0], skip_special_tokens=True)

    # Question B: La Bouche (Pour valider les AUs comme 'Jaw Drop' ou 'Smile')
    inputs_mouth = processor(pil_image, "the mouth is", return_tensors="pt").to(device_llm)
    out_mouth = model_llm.generate(**inputs_mouth, max_new_tokens=10)
    desc_mouth = processor.decode(out_mouth[0], skip_special_tokens=True)

    # Question C: Les Yeux (Pour valider 'Brow Raiser' ou 'Eyes Closed')
    inputs_eyes = processor(pil_image, "the eyes are", return_tensors="pt").to(device_llm)
    out_eyes = model_llm.generate(**inputs_eyes, max_new_tokens=10)
    desc_eyes = processor.decode(out_eyes[0], skip_special_tokens=True)

    # 4. Affichage
    plt.figure(figsize=(6, 6))
    plt.imshow(pil_image)
    plt.axis('off')
    plt.title(f"Vrai: {true_label}", color='black', fontsize=14)
    plt.show()

    # 5. Le Rapport Comparatif (Humain vs Machine)
    print("-" * 60)
    print(f"📄 FICHIER : {simple_filename}")
    print("-" * 60)
    print(f"🧬 VÉRITÉ ANATOMIQUE (Ground Truth AUs) :")
    print(f"   ➡️ {au_explanation}")
    print("-" * 60)
    print(f"🤖 ANALYSE VISION-LLM (Génération Conditionnelle) :")
    print(f"   1. Sentiment global : '... {desc_emo}'")
    print(f"   2. État de la bouche : '... {desc_mouth}'")
    print(f"   3. État des yeux     : '... {desc_eyes}'")
    print("-" * 60)

    # Petite vérification automatique de cohérence
    if "open" in desc_mouth and ("Lips part" in au_explanation or "Jaw drop" in au_explanation):
        print("✅ COHÉRENCE DÉTECTÉE : Le LLM a vu la bouche ouverte (validé par AU).")
    elif "smil" in desc_mouth and "Lip corner puller" in au_explanation:
        print("✅ COHÉRENCE DÉTECTÉE : Le LLM a vu le sourire (validé par AU).")

# --- TEST ---
print("🔬 Test du Prompt Engineering Visuel :")
# Test sur l'image 73 (celle qui échouait avant)
analyze_image_focused(test_dataset, 36)
analyze_image_focused(test_dataset, 73) # Celle avec la bouche ouverte

# %%
def generate_multimodal_report(dataset, index, model_resnet, model_llm, processor_llm):
    print(f"\n🔄 GÉNÉRATION DU RAPPORT POUR L'IMAGE INDEX {index}...\n")

    # --- 1. DONNÉES DE BASE ---
    img_tensor, label_idx = dataset[index]
    full_filename = dataset.filenames[index]
    simple_filename = full_filename.split('_')[-1]
    true_label_text = LABELS_LIST[label_idx]

    # Image PIL pour BLIP et Affichage
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    pil_image = transforms.ToPILImage()(inv_normalize(img_tensor))

    # --- 2. PRÉDICTION RESNET (Vision Only) ---
    model_resnet.eval()
    with torch.no_grad():
        out = model_resnet(img_tensor.unsqueeze(0).to(device))
        probs = F.softmax(out, dim=1)
        conf, pred_idx = torch.max(probs, 1)
    pred_label_text = LABELS_LIST[pred_idx.item()]

    # --- 3. EXPLICATION SCIENTIFIQUE (Ground Truth AUs) ---
    au_desc = image_to_au_text.get(simple_filename, "Non spécifié")

    # --- 4. INTERPRÉTATION VISION-LLM (Prompt Engineering) ---
    # On pose 3 questions ciblées au LLM
    prompts = {
        "Expression": "the facial expression is",
        "Yeux": "the eyes are",
        "Bouche": "the mouth is"
    }
    llm_responses = {}

    for key, prompt in prompts.items():
        inputs = processor_llm(pil_image, prompt, return_tensors="pt").to(device_llm)
        out = model_llm.generate(**inputs, max_new_tokens=15)
        llm_responses[key] = processor_llm.decode(out[0], skip_special_tokens=True)

    # --- 5. AFFICHAGE DU RAPPORT VISUEL ---
    plt.figure(figsize=(10, 6))

    # Image
    plt.subplot(1, 2, 1)
    plt.imshow(pil_image)
    plt.axis('off')
    col = 'green' if true_label_text == pred_label_text else 'red'
    plt.title(f"Image: {simple_filename}\nLabel: {true_label_text}", fontsize=12)

    # Texte du rapport (Affiché comme une image pour le style)
    plt.subplot(1, 2, 2)
    plt.axis('off')

    text_report = (
        f"📊 CLASSIFICATION (ResNet)\n"
        f"---------------------------\n"
        f"Prédiction : {pred_label_text}\n"
        f"Confiance  : {conf.item()*100:.1f}%\n\n"

        f"🧬 ANATOMIE (Vérité Terrain)\n"
        f"---------------------------\n"
        f"AUs activés : {au_desc}\n\n"

        f"🤖 ANALYSE VISION-LLM\n"
        f"---------------------------\n"
        f"• {llm_responses['Expression']}\n"
        f"• {llm_responses['Yeux']}\n"
        f"• {llm_responses['Bouche']}\n"
    )
    plt.text(0, 0.2, text_report, fontsize=11, family='monospace', va='center')

    plt.tight_layout()
    plt.show()

# TEST FINAL
generate_multimodal_report(test_dataset, 35, trained_model, model_llm, processor)


