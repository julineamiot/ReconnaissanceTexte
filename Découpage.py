import numpy as np
import cv2

def binarise(image, seuil=200):
    h, w = image.shape
    for i in range(h):
        for j in range(w):
            if image[i][j] <= seuil:
                image[i][j] = 0   # texte sombre → 0
            else:
                image[i][j] = 1   # fond clair → 1



def binarise2(image):
    _, img_bin = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img_bin




def decoupage_horizontal(matrice, seuil=1):
    somme = np.sum(matrice, axis=1)
    lignes_finales = []
    debut = None

    for y in range(len(somme)):
        if somme[y] > seuil and debut is None:
            debut = y
        elif somme[y] <= seuil and debut is not None:
            lignes_finales.append(matrice[debut:y])
            debut = None

    if debut is not None:
        lignes_finales.append(matrice[debut:])

    return lignes_finales



def decoupage_vertical(liste_matrice, seuil=1):
    images_finales = []

    for matrice in liste_matrice:
        somme = np.sum(matrice, axis=0)

        lettres = []
        debut = None

        for x in range(len(somme)):
            if somme[x] > seuil and debut is None:
                debut = x

            elif somme[x] <= seuil and debut is not None:
                lettres.append(matrice[:, debut:x])
                debut = None

        if debut is not None:
            lettres.append(matrice[:, debut:])

        images_finales.append(lettres)

    return images_finales



def remplissage1(liste_images,resolution=28):
    nouvelles_images=[]
    for image in liste_images:
        while len(image)<resolution and len(image[0])<resolution:
            if len(image)<resolution:
                somme = np.sum(image, axis=1)
                blanc_haut=0
                x=0
                while somme[x]==0 and x<resolution:
                    blanc_haut+=1
                    x+=1
                blanc_bas=0
                x=len(image)
                while somme[x]==0 and x>0:
                    blanc_bas+=1
                    x-=1
                if blanc_bas>blanc_haut:
                    image=np.pad(image, pad_width=((1,0),(0,0)), mode='constant', constant_values=0)
                else:
                    image=np.pad(image, pad_width=((0,1),(0,0)), mode='constant', constant_values=0)
            if len(image[0]) < resolution:
                somme = np.sum(image, axis=2)
                blanc_gauche = 0
                x = 0
                while somme[x] == 0 and x < resolution:
                    blanc_gauche += 1
                    x += 1
                blanc_droite = 0
                x = len(image[0])
                while somme[x] == 0 and x > 0:
                    blanc_droite += 1
                    x -= 1
                if blanc_droite > blanc_gauche:
                    image = np.pad(image, pad_width=((0, 0), (1, 0)), mode='constant', constant_values=0)
                else:
                    image = np.pad(image, pad_width=((0, 0), (0, 1)), mode='constant', constant_values=0)
            nouvelles_images.append(image)
    return nouvelles_images


def remplissage2(liste_images, resolution=28):
    nouvelle_images = []
    for image in liste_images:
        hauteur, largeur = image.shape
        manque_h = resolution - hauteur
        manque_w = resolution - largeur
        if manque_h<0 or manque_w<0:
            nouvelle_images.append(image)
            continue
        haut=manque_h//2
        bas=manque_h-haut
        gauche=manque_w//2
        droite=manque_w-gauche
        image_remplie=np.pad(image,pad_width=((haut, bas), (gauche, droite)),mode='constant',constant_values=0)
        nouvelle_images.append(image_remplie)
    return nouvelle_images

def moyenne(image,taille=3):
    imagebis=np.zeros((int(len(image)/taille), int(len(image)/taille)))
    for n in range(int(len(image)/taille)):
        for p in range(int(len(image)/taille)):
            somme = 0
            for i in range(taille):
                for j in range(taille):
                    somme += image[n*taille+i][p*taille+j]
            res = somme/(taille*taille)
            imagebis[n][p] = res
    return imagebis
