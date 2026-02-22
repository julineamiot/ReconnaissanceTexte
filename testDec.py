import cv2
import numpy as np
from Découpage import (
    binarise2,
    decoupage_horizontal,
    decoupage_vertical,
    remplissage2
)

def tester_decoupage(path_image):
    """
    Charge une image, applique le pipeline de découpage,
    puis affiche chaque lettre découpée dans une fenêtre OpenCV.
    """

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

    # 6. Affichage des lettres
    compteur = 0
    for i, ligne in enumerate(lettres_finales):
        for j, lettre in enumerate(ligne):
            # Normalisation pour affichage (0→255)
            img_aff = (lettre * 255).astype(np.uint8)

            cv2.imshow(f"Ligne {i} - Lettre {j}", img_aff)
            compteur += 1

    print(f"{compteur} lettres affichées.")
    print("Appuie sur une touche pour fermer les fenêtres.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    image_path = "TestImg.jpg"  # mets ton image ici
    tester_decoupage(image_path)
