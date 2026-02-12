import numpy as np
import struct
from array import array
import matplotlib.pyplot as plt
import random


# --- 1. MODIFICATION DU DATALOADER ---
class MnistDataloader(object):
    def __init__(self):
        # On pointe vers les fichiers EMNIST Letters (visibles dans ta capture)
        input_path = 'Datas_lettres'
        # Assure-toi que ces fichiers sont bien dans le dossier 'Datas'
        self.training_images_filepath = input_path + '/emnist-balanced-train-images-idx3-ubyte'
        self.training_labels_filepath = input_path + '/emnist-balanced-train-labels-idx1-ubyte'
        self.test_images_filepath = input_path + '/emnist-balanced-test-images-idx3-ubyte'
        self.test_labels_filepath = input_path + '/emnist-balanced-test-labels-idx1-ubyte'

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())

        images = []
        for i in range(size):
            # Récupération de l'image brute
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)

            # --- CRUCIAL POUR EMNIST ---
            # Les images EMNIST sont pivotées et inversées par rapport à MNIST classique.
            # On applique une transposition pour les remettre dans le bon sens.
            img = img.T

            images.append(img)

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)

class Neuronneclass:
    def __init__(self, neuronnes, app):
        self.neuronnes = neuronnes
        self.app = app
        self.poids = []
        self.biais = []
        for i in range(len(neuronnes) - 1):
            n = neuronnes[i]
            n_plus_un = neuronnes[i + 1]
            W = np.random.uniform(-np.sqrt(6 / n), np.sqrt(6 / n), size=(n, n_plus_un))
            self.poids.append(W)
            B = np.zeros((1, n_plus_un))
            self.biais.append(B)

    def feedforward(self,inputs):
        activation = inputs
        resultat = [activation]
        for i in range(len(self.poids)):
            w = self.poids[i]
            b = self.biais[i]
            z = np.dot(activation, w) + b
            activation = self.sigmoid(z)
            resultat.append(activation)
        return resultat

    def feedforwardneur(self, nombres, poids, biais, fonction):
        x = 0
        for i in range(len(nombres)):
            x += nombres[i] * poids[i]
        if fonction == 'tanh':
            return self.tanh(x + biais)
        if fonction == 'sigmoid':
            return self.sigmoid(x + biais)
        if fonction == 'relu':
            return self.relu(x + biais)

    def relu(self, x):
        return max(0, x)

    def relu_deriv(self,x):
        if x>0:
            return 1
        else:
            return 0
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return x * (1 - x)

    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def cout(self, liste_res, liste_res_attendu):
        cout = 0
        for i in range(len(liste_res)):
            cout += abs(liste_res_attendu[i] - liste_res[i])
        return cout

    def backwardpropagation(self,delta,resultat):
        w = self.poids
        for n in range(len(self.poids)):
            for i in range(len(self.poids[n])):
                for j in range(len(self.poids[n][i])):
                    w[n][i][j] = w[n][i][j] + self.app * resultat[n][i] * delta[n][j]

    def backwardpropagation2(self,delta,resultat):
        for n in range(len(self.poids)):
            gradient = np.dot(resultat[n].T,delta[n+1])
            self.poids[n] += self.app*gradient
            self.biais[n] += self.app * delta[n + 1]
    def delta(self,label,resultats):
        delta = [np.zeros_like(r) for r in resultats]
        w = self.poids
        delta[-1] = (label - resultats[-1])
        for i in range(len(self.poids)-2,-1,-1):
            for j in range(len(self.biais[i])):
                mat = np.dot(delta[i+1],w[i].T)
                somme = 0
                for ligne in range(len(mat)):
                    somme += mat[ligne]
                delta[i][j] = resultats[i][j]*(1-resultats[i][j])*somme
        return delta

    def delta_mat(self, label_vector,resultats):
        deltas = [np.zeros_like(r) for r in resultats]
        output = resultats[-1]
        error = (label_vector - output)
        deltas[-1] = error * self.sigmoid_deriv(output)
        for i in range(len(self.poids) - 1, -1, -1):
            error_prop = np.dot(deltas[i + 1], self.poids[i].T)
            d_activ = self.sigmoid_deriv(resultats[i])
            deltas[i] = error_prop * d_activ

        return deltas
    def softmax(self,liste):
        somme = 0
        for i in range(len(liste)):
            somme += liste[i]
        for i in range(len(liste)):
            liste[i] = liste[i]/somme


def input_Mnist_trad():
    dataloader = MnistDataloader()
    (x_train, label_train), (x_test, label_test) = dataloader.load_data()
    xtrain = np.array(x_train)
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtrain = xtrain / 255.0
    xtest = np.array(x_test)
    xtest = xtest.reshape(xtest.shape[0], -1)
    xtest = xtest / 255.0
    return (xtrain, label_train), (xtest, label_test)

(x_train, label_train), (x_test, label_test)=input_Mnist_trad()
def label_trad(label, classes=47):
    vecteur = np.zeros((1, classes))
    vecteur[0][label] = 1
    return vecteur

def trainingtesting(x_train,label_train,x_test, label_test,boucle=10):

    perc = Neuronneclass([784, 400, 150, 47], 0.12)
    #training
    for j in range(boucle):
        print(j)

        for i in range(len(x_train)):
            pixel = x_train[i].reshape(1, -1)
            label = label_trad(label_train[i])
            resultats = perc.feedforward(pixel)
            delta = perc.delta_mat(label, resultats)
            perc.backwardpropagation2(delta, resultats)
    #test
    somme = 0
    for i in range(len(x_test)):
        pixel=x_test[i].reshape(1, -1)

        resultats = perc.feedforward(pixel)
        reponse = np.argmax(resultats[-1])

        if reponse == label_test[i]:
            somme += 1
    print("L'IA reconnait suite a son entrainement :",(somme/len(label_test))*100,"% des images")

def traintrain(x_train,label_train,x_test, label_test,boucle=5):
    perc = Neuronneclass([784, 128, 64, 10], 0.05)
    # training
    for j in range(boucle):
        print(j)

        for i in range(len(x_train)):
            pixel = x_train[i].reshape(1, -1)
            label = label_trad(label_train[i])
            resultats = perc.feedforward(pixel)
            delta = perc.delta_mat(label, resultats)
            perc.backwardpropagation2(delta, resultats)
    
    # test
    somme=0
    for i in range(len(x_train)):
        pixel=x_train[i].reshape(1, -1)

        resultats = perc.feedforward(pixel)
        reponse = np.argmax(resultats[-1])

        if reponse == label_train[i]:
            somme += 1
    print("L'IA reconnait suite a son entrainement :",(somme/len(label_train))*100,"% des images")


trainingtesting(x_train, label_train,x_test, label_test)
#traintrain(x_train, label_train,x_test, label_test)



"""    def load_weights(self, filename="poids_reseau.npz"):  # pour utiliser les données obtenue avec l'entrainement précédant
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

        np.savez(filename, **all_params)"""
