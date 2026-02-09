import numpy as np
def binarise(image):
    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i][j]>200:
                image[i][j] = 1
            else:
                image[i][j] = 0


def binarise2(image,seuil=200):
    return np.where(image <= seuil, 1, 0)


def decoupage_horizontal(matrice):
    somme = np.sum(matrice, axis=1)
    lignes_finales = []
    n = 0
    debut_trouve = False
    start_y = 0
    while n < len(somme):
        if somme[n] != 0:
            if debut_trouve == False:
                start_y = n
                debut_trouve = True
        else:
            if debut_trouve == True:
                end_y = n
                tranche = matrice[start_y: end_y]
                lignes_finales.append(tranche)
                debut_trouve = False
        n = n + 1
    if debut_trouve == True:
        lignes_finales.append(matrice[start_y: len(matrice)])
    return lignes_finales


def decoupage_vertical(liste_matrice):
    images_finales=[]
    for matrice in liste_matrice:
        somme = np.sum(matrice, axis=2)
        lignes_finales = []
        n = 0
        debut_trouve = False
        start_y = 0
        while n < len(somme):
            if somme[n] != 0:
                if debut_trouve == False:
                    start_y = n
                    debut_trouve = True
            else:
                if debut_trouve == True:
                    end_y = n
                    tranche = matrice[start_y: end_y]
                    lignes_finales.append(tranche)
                    debut_trouve = False
            n = n + 1
        if debut_trouve == True:
            lignes_finales.append(matrice[start_y: len(matrice)])
        images_finales.append(lignes_finales)
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