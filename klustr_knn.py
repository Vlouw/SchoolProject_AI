# -------------------------------------------------------------
# Auteur  : Henri-Paul Bolduc
#           Ariel Hotz-Garber
#           Gael Lane Lepine
# Cours   : 420-C52-IN - AI 1
# TP 1    : Analyse KNN des images
# Fichier : klustr_knn.py
# -------------------------------------------------------------

# Importation des modules
# -------------------------------------------------------------
from klustr_engine import *
import numpy as np
# -------------------------------------------------------------

# Knn
# -------------------------------------------------------------
class Knn():
    def __init__(self, array, labels, classify_pt):
        self.distances = self.calcul_distance(array, classify_pt)
        self.labels = labels

    # calculer la distance entre tous les points et celui choisi
    def calcul_distance(self, array, classify_pt):
        classify_pt = np.reshape(classify_pt, (array.shape[0],1))
        return np.sqrt((np.sum((array - classify_pt)**2, axis=0)))
        
    def knn_classifiy(self, k, distance):
        if k < 1:
            self.k = 1
        elif k > self.distances.shape[0]:
            self.k = self.distances.shape[0]
        else:
            self.k = k

        self.distances_classees = np.argsort(self.distances)
        self.etiquettes_classees = np.take(self.labels, self.distances_classees)
        self.k_voisins = self.etiquettes_classees[:self.k]
        self.label_gagnant, self.frequency = np.unique(self.k_voisins, return_inverse = True)
        counts = np.bincount(self.frequency)
        maxpos = counts.argmax()
        
        return self.label_gagnant[maxpos]
# -------------------------------------------------------------