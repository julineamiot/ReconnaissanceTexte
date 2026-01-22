import numpy as np
from PIL import Image

class NeuronalNetwork:
    def __init__(self):
        result = []
        return None

    # partie pré-gestion des données

    def OpenIMGtxt(self):
        #ouvre une image et la convertit en matrice numpy,  avec des niveaux de gris entre 0 et 255
        image = Image.open("nom image")
        imageGris = image.convert("L")
        imageMatrice = np.asarray(imageGris)
        return imageMatrice

    def DecImg(self):
        return None # pour centrer l'image

    def Analyse(self):
        for i in self.DecImg():
            for j in self.DecImg():
                # calc les probas pour tt les chiffres avec notre rés de neuronnes initiales
                # garder la plus grande proba
                value = value
                self.result.append(value)
        return None