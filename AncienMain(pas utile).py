import numpy as np
from ReadingMnist import MnistDataloader
import pickle

class ReseauNeuronne:

    def __init__(self):
        self.W = {}
        self.C = {}
        self.Z = {}  # pour stock les valeurs avant activation
        self.b = {} #biais
        return None

    def simogoid(self, x):
        return 1 / (1 + np.exp(-x))

    def d_simogoid(self, x):
        s = self.simogoid(x)
        return s * (1 - s)

    def importerImage(self):   #importer l'image
        return I # I l'image

    def IntialisationW(self, Pi, NbNeuronnes):
        self.W[1] = np.random.randn(Pi.shape[1], NbNeuronnes[0]) * 0.01
        self.b[1] = np.zeros((1, NbNeuronnes[0]))

    def CalcWCoucheINI(self, NbNeuronnes, n):



        if n == 1:
            # Poids entre couche cachée 1 et sortie
            self.W[2] = np.random.randn(NbNeuronnes[0], 1) * 0.01
            self.b[2] = np.zeros((1, 1))
            return


        for i in range(2, n + 1):
            # Taille de la couche précédente
            in_size = NbNeuronnes[i - 3]
            # Taille de la couche actuelle
            out_size = NbNeuronnes[i - 2]

            self.W[i] = np.random.randn(in_size, out_size) * 0.01
            self.b[i] = np.zeros((1, out_size))


        self.W[n + 1] = np.random.randn(NbNeuronnes[n - 2], 1) * 0.01
        self.b[n + 1] = np.zeros((1, 1))


    def forward(self, Pi, n, NbNeuronnes):


        Pi = Pi.reshape(1, -1) / 255.0


        if not hasattr(self, "initialized"):
            self.IntialisationW(Pi, NbNeuronnes)  # W[1] + b[1]
            self.CalcWCoucheINI(NbNeuronnes, n)  # W[2..n+1] + b[2..n+1]
            self.initialized = True

        self.Z = {}
        self.C = {}

        # Couche d'entrée
        self.C[0] = Pi


        for i in range(1, n + 1):
            self.Z[i] = np.dot(self.C[i - 1], self.W[i]) + self.b[i]
            self.C[i] = self.simogoid(self.Z[i])

      #sortie
        self.Z[n + 1] = np.dot(self.C[n], self.W[n + 1]) + self.b[n + 1]
        self.C[n + 1] = self.simogoid(self.Z[n + 1])

        return self.C[n + 1]

    def backward(self, target, n, eta=0.01):


        y = self.C[n + 1]  # sortie du réseau

        # BCE + sigmoid : donne un gradiant pls variant
        delta = y - target   # on prend cet fonction de coût car elle permet de créer des écarts dans l'apprentissage

        # Mise à jour poids + biais de la couche de sortie
        self.W[n + 1] -= eta * np.dot(self.C[n].T, delta)  # le T donne la transposée de la matrice, on ajuste les poids avec cet formule
        self.b[n + 1] -= eta * delta


        for i in range(n, 0, -1):  # on parcours les couches dans le sens inverses, cad, on part de l'arrivés et on remonte, en considérant pour modifier la couche i l'erreur et la modif de la couche i + 1
            # dérivée de la sigmoïde
            d_act = self.d_simogoid(self.Z[i])

            # rétropropagation du delta ( formule mathématique, dérivés en chaines )
            delta = np.dot(delta, self.W[i + 1].T) * d_act

            # mise à jour poids + biais
            self.W[i] -= eta * np.dot(self.C[i - 1].T, delta)
            self.b[i] -= eta * delta

    def save_weights(self, filename="poids_reseau.npz"):   # pour enregistrer le résultat de l'entrainement et pouvoir le réutiliser sans avoir à systématiquement relancer l'entrainement

        W_str = {f"W{k}": v for k, v in self.W.items()}
        b_str = {f"b{k}": v for k, v in self.b.items()}


        all_params = {**W_str, **b_str}

        np.savez(filename, **all_params)

    def load_weights(self, filename="poids_reseau.npz"):  # pour utiliser les données obtenue avec l'entrainement précédant
        data = np.load(filename)

        self.W = {}
        self.b = {}

        for key in data.files:
            if key.startswith("W"):
                idx = int(key[1:])
                self.W[idx] = data[key]
            elif key.startswith("b"):
                idx = int(key[1:])
                self.b[idx] = data[key]

        # Empêche forward de réinitialiser les poids
        self.initialized = True

    def save_initial_weights(self, filename="poids_initiaux.npz"):
        # np.savez exige des clés string
        W_str = {f"W{k}": v for k, v in self.W.items()}
        b_str = {f"b{k}": v for k, v in self.b.items()}

        all_params = {**W_str, **b_str}

        np.savez(filename, **all_params)

    def load_initial_weights(self, filename="poids_initiaux.npz"):
        data = np.load(filename)

        self.W = {}
        self.b = {}

        for key in data.files:
            if key.startswith("W"):
                idx = int(key[1:])
                self.W[idx] = data[key]
            elif key.startswith("b"):
                idx = int(key[1:])
                self.b[idx] = data[key]

        # Empêche forward de réinitialiser les poids
        self.initialized = True


# lancer un entrainement :

N=ReseauNeuronne()
P=MnistDataloader()
(x_train, y_train), (x_test, y_test) = P.load_data()
T = np.array(x_train[0])
N.forward(T, 1, [164]) # initialise les poids N.save_initial_weights()
N.save_initial_weights()

"""
e = 0
for i in x_train :
    i = np.array(i)
    y = N.forward(i, 1, [164])
    if e ==1 :
     N.save_initial_weights()
    target = 1 if y_train[e] == 9 else 0
    N.backward(target, 1, eta=0.05)
    e = e+1
N.save_weights()


# lancer un affichage de résultat post entrainement sur les données d'entrainement
"""
"""
N = ReseauNeuronne()
P=MnistDataloader()
N.load_weights()
(x_train, y_train), (x_test, y_test) = P.load_data()
g = 0
Find = 0
NotFind = 0
FalseFind = 0
Goodrjt = 0
for i in x_test :
    i = np.array(i)
    value = N.forward(i,1, [164]) 
    if value > 0.5 :
        value = 1
    else :
        value = 0
    if y_test[g] == 9 and value == 1 :
        Find = Find + 1
    if y_test[g] == 9 and value == 0 :
        NotFind = NotFind + 1
    if not y_test[g] == 9 and value == 1 :
        FalseFind = FalseFind + 1
    if not y_test[g] == 9 and value == 0 :
        Goodrjt = Goodrjt + 1
    g = g + 1
print("9 trouvé à juste titre : " + str(Find) + " / 9 non trouvé : " + str(NotFind) + " / Erreur d'affirmation : " + str(FalseFind) + " / Bon rejet : " + str(Goodrjt))"""

def train(n, NbNeuronnes, eta):
    e = 0
    for i in x_train:
        i = np.array(i)
        y = N.forward(i,n, NbNeuronnes)
        target = 1 if y_train[e] == 9 else 0
        N.backward(target, n, eta)
        e = e + 1
    N.save_weights()
    print("train with n : " + str(n) + " and NbNeuronnes = " + str(NbNeuronnes) + " and eta = " + str(eta))

def UtilisationPostTraining(n, NbNeuronnes):
    N.load_weights()
    g = 0
    Find = 0
    NotFind = 0
    FalseFind = 0
    Goodrjt = 0
    for i in x_test:
        i = np.array(i)
        value = N.forward(i, n, NbNeuronnes)
        if value > 0.5:
            value = 1
        else:
            value = 0
        if y_test[g] == 9 and value == 1:
            Find = Find + 1
        if y_test[g] == 9 and value == 0:
            NotFind = NotFind + 1
        if not y_test[g] == 9 and value == 1:
            FalseFind = FalseFind + 1
        if not y_test[g] == 9 and value == 0:
            Goodrjt = Goodrjt + 1
        g = g + 1
    print("9 trouvé à juste titre : " + str(Find) + " / 9 non trouvé : " + str(
        NotFind) + " / Erreur d'affirmation : " + str(FalseFind) + " / Bon rejet : " + str(Goodrjt))


"""
# test de perf :
N = ReseauNeuronne()
N.load_initial_weights()
train(1,[164], 0.05)
UtilisationPostTraining(1,[164])

# réduc du nombre de neuronne
N = ReseauNeuronne()
N.load_initial_weights()
train(1,[10],0.05)
UtilisationPostTraining(1, [10])

# réduc + une couche
N = ReseauNeuronne()
N.forward(T, 2,[164,164])
N.save_initial_weights()
train(2,[164,164],0.05)
UtilisationPostTraining(2, [164,164])

# modification du eta
N = ReseauNeuronne()
N.load_initial_weights()
train(2,[164,164],0.01)
UtilisationPostTraining(2, [164,164])

N = ReseauNeuronne()
N.load_initial_weights()
train(2,[164,164],0.1)
UtilisationPostTraining(2, [164,164])"""

N = ReseauNeuronne()
N.forward(T, 1,[164])
N.save_initial_weights()
train(1,[164],0.01)
UtilisationPostTraining(1, [164])












# environ 90% de taux de réussite
