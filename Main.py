import cv2
import numpy as np
from Reseaux_lettres import Neuronneclass

from Découpage import (
    binarise2,
    decoupage_horizontal,
    decoupage_vertical,
    remplissage2
)
EMNIST_LABELS = [
    '0','1','2','3','4','5','6','7','8','9',      # 0–9
    'A','B','C','D','E','F','G','H','I','J',      # 10–19
    'K','L','M','N','O','P','Q','R','S','T',      # 20–29
    'U','V','W','X','Y','Z',                      # 30–35
    'a','b','c','d','e','f','g','h','i','j',      # 36–45
    'k'                                           # 46
]
def charger_reseau():
    reseau = Neuronneclass([784, 400, 150, 47], 0.12)
    reseau.load_reseau("reseau_complet.txt")
    return reseau
def tester_decoupage(path_image):
    img = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Impossible de charger l'image : " + path_image)

    img_bin = binarise2(img)

    lignes = decoupage_horizontal(img_bin)

    lettres_par_ligne = decoupage_vertical(lignes)

    lettres_finales = []
    for ligne in lettres_par_ligne:
        lettres = remplissage2(ligne, resolution=28)
        lettres_finales.append(lettres)

    compteur = 0
    for i, ligne in enumerate(lettres_finales):
        for j, lettre in enumerate(ligne):
            img_aff = (lettre * 255).astype(np.uint8)

            cv2.imshow(f"Ligne {i} - Lettre {j}", img_aff)
            compteur += 1

    print(f"{compteur} lettres affichées.")
    print("Appuie sur une touche pour fermer les fenêtres.")
    reseau = charger_reseau()
    ligne_txt = ""
    for ligne in lettres_finales:

        for lettre in ligne:
            lettre_transp = lettre.T

            lettre_plate = lettre_transp.reshape(1, -1)

            resultat = reseau.feedforward(lettre_plate)
            index = np.argmax(resultat[-1])
            letter = EMNIST_LABELS[index]
            ligne_txt += letter
            #print(resultat)
        ligne_txt+="\n"
    print(ligne_txt)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "img_1.png"
    tester_decoupage(image_path)

    image_path2 = "img_2.png"
    tester_decoupage(image_path2)

"""
print("exemple 1")

if __name__ == "__main__":
    image_path = "1.png" 
    tester_decoupage(image_path)

print("exemple 2")

if __name__ == "__main__":
    image_path = "2.png" 
    tester_decoupage(image_path)

print("exemple 3")

if __name__ == "__main__":
    image_path = "3.png" 
    tester_decoupage(image_path)

print("exemple 4")

if __name__ == "__main__":
    image_path = "4.png" 
    tester_decoupage(image_path)

print("exemple 5")

if __name__ == "__main__":
    image_path = "5.png" 
    tester_decoupage(image_path)

print("exemple 6")

if __name__ == "__main__":
    image_path = "6.png" 
    tester_decoupage(image_path)

print("exemple 7")

if __name__ == "__main__":
    image_path = "7.png" 
    tester_decoupage(image_path)"""



