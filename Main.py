import numpy as np
import cv2

# --- IMPORTS DES DEUX FICHIERS ---
from Reseaux_lettres import Neuronneclass
from Découpage import (
    binarise2,
    decoupage_horizontal,
    decoupage_vertical,
    remplissage2
)

# ---------------------------------------------------------
#  TABLE DE CORRESPONDANCE EMNIST BALANCED (47 classes)
# ---------------------------------------------------------
# Source officielle EMNIST : mapping Balanced
EMNIST_LABELS = [
    '0','1','2','3','4','5','6','7','8','9',      # 0–9
    'A','B','C','D','E','F','G','H','I','J',      # 10–19
    'K','L','M','N','O','P','Q','R','S','T',      # 20–29
    'U','V','W','X','Y','Z',                      # 30–35
    'a','b','c','d','e','f','g','h','i','j',      # 36–45
    'k'                                           # 46
]


def charger_reseau():  # entrainement etc

    reseau = Neuronneclass([784, 400, 150, 47], 0.12)
    return reseau

# ---------------------------------------------------------
#  PRÉPARATION D’UNE IMAGE 28×28 POUR LE RÉSEAU
# ---------------------------------------------------------
def prepare_image(img):
    img = img.astype(np.float32)
    img = img.reshape(1, -1)
    img = img / 255.0
    return img

# ---------------------------------------------------------
#  RECONNAISSANCE D’UNE LETTRE
# ---------------------------------------------------------
def reconnaitre_lettre(reseau, img28):
    entree = prepare_image(img28)
    resultats = reseau.feedforward(entree)
    index = np.argmax(resultats[-1])
    return EMNIST_LABELS[index]

# ---------------------------------------------------------
#  PIPELINE COMPLET : IMAGE → TEXTE
# ---------------------------------------------------------
def ocr_image(path_image):
    # 1. Charger l’image
    img = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Impossible de charger l'image : " + path_image)

    # 2. Binarisation
    img_bin = binarise2(img)

    # 3. Découpage horizontal (lignes)
    lignes = decoupage_horizontal(img_bin)

    # 4. Découpage vertical (lettres)
    lettres_par_ligne = decoupage_vertical(lignes)

    # 5. Remplissage 28×28
    lettres_finales = []
    for ligne in lettres_par_ligne:
        lettres = remplissage2(ligne, resolution=28)
        lettres_finales.append(lettres)

    # 6. Reconnaissance
    reseau = charger_reseau()
    texte = []

    for ligne in lettres_finales:
        ligne_txt = ""
        for lettre in ligne:
            lettre_txt = reconnaitre_lettre(reseau, lettre)
            ligne_txt += lettre_txt
        texte.append(ligne_txt)

    return texte

# ---------------------------------------------------------
#  MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    image_path = "mon_image.png"  # ← mets ici ton image

    texte = ocr_image(image_path)  # construction de la matrice de matrice

    print("Texte reconnu :")
    for ligne in texte:   # annonce du texte ligne par ligne
        print(ligne)
