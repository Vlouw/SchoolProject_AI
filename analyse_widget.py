# -------------------------------------------------------------
# Auteur  : Henri-Paul Bolduc
#           Ariel Hotz-Garber
#           Gael Lane Lepine
# Cours   : 420-C52-IN - AI 1
# TP 1    : Analyse KNN des images
# Fichier : analyse_widget.py
# -------------------------------------------------------------

# Importation des modules
# -------------------------------------------------------------
import sys
import numpy as np
import matplotlib.pyplot as plt

from klustr_utils import qimage_argb32_from_png_decoding
from klustr_dao import *

from klustr_engine import *
from klustr_knn import *

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtCore import Slot

from PySide6.QtWidgets import  (QWidget, QGroupBox, QLabel, QGridLayout, QHBoxLayout, 
                                QVBoxLayout, QMessageBox, QPushButton, QComboBox, QSlider)
from PySide6.QtGui import QPixmap

from __feature__ import snake_case, true_property
# -------------------------------------------------------------

# View Widget
# -------------------------------------------------------------
class Projet1ViewWidget(QWidget):

    def __init__(self, klustr_dao, parent=None):        
        super().__init__(parent)

        self.klustr_dao = klustr_dao        
        if self.klustr_dao.is_available:
            self._setup_models()
            self._setup_gui()
        else:
            self._setup_invalid_gui()

    def _setup_models(self):
        # Data for Dataset Box
        # --------------------
        self.alldata = np.array(self.klustr_dao.available_datasets, dtype=object)
        
        self.dataset = np.array(self.klustr_dao.available_datasets, dtype=object)[:,1]
        self.dataset = self.dataset + ' [' + np.array(self.klustr_dao.available_datasets, dtype=object)[:,5].astype(str) + ']'
        self.dataset = self.dataset + ' [' + np.array(self.klustr_dao.available_datasets, dtype=object)[:,8].astype(str) + ']' 
    
    # Code Jean-Christophe si la data base n'est pas accessible
    # ---------------------------------------------------------    
    def _setup_invalid_gui(self):
        not_available = QLabel('Data access unavailable')
        not_available.alignment = Qt.AlignCenter
        not_available.enabled = False
        layout = QGridLayout(self)
        layout.add_widget(not_available)
        QMessageBox.warning(self, 'Data access unavailable', 'Data access unavailable.')
    
    # Setup interface graphique
    # -------------------------    
    def _setup_gui(self):        
        # QLabel
        # ------
        self.qlab_matgraph = QLabel()   
        
        qlab_catcount = QLabel("Category count:")
        qlab_trainimgcount = QLabel("Training image count:")
        qlab_testimgcount = QLabel("Test image count:")
        qlab_totalimgcount = QLabel("Training image count:")
        self.qlab_catcount_value = QLabel()        
        self.qlab_trainimgcount_value = QLabel()
        self.qlab_testimgcount_value = QLabel()
        self.qlab_totalimgcount_value = QLabel()        
        
        qlab_translated = QLabel("Translated:")
        qlab_rotated = QLabel("Rotated:")
        qlab_scaled = QLabel("Scaled:")
        self.qlab_translated_tf = QLabel()
        self.qlab_rotated_tf = QLabel()
        self.qlab_scaled_tf = QLabel()
        
        self.qlab_image = QLabel()        
        self.qlab_classify = QLabel()        
        
        self.qlab_k = QLabel()
        self.qlab_max = QLabel()
        
        # Slider
        # ------
        self.qslid_k = QSlider(Qt.Horizontal)
        self.qslid_max = QSlider(Qt.Horizontal)
        
        self.qslid_k.valueChanged.connect(self._qslid_k_change)
        self.qslid_max.valueChanged.connect(self._qslid_max_change)
        
        self.qslid_k.minimum = 1
        self.qslid_k.tick_interval = 1
        
        self.qslid_max.minimum = 1
        self.qslid_max.maximum = 10
        self.qslid_max.tick_interval = 1        
        self.qslid_max.value = 3 
        
        # Bouton
        # ------
        buttonAbout = QPushButton('About')
        buttonClassify = QPushButton('Classify')
        
        buttonAbout.clicked.connect(self._click_about)
        buttonClassify.clicked.connect(self._click_classify)
        
        # Combo Box
        # ---------
        self.cbox_singletest = QComboBox()
        self.cbox_singletest.currentIndexChanged.connect(self._cbox_singletest_index_change)
        
        self.cbox_dataset = QComboBox()
        self.cbox_dataset.currentIndexChanged.connect(self._cbox_dataset_index_change)  
        self.cbox_dataset.add_items(self.dataset)    
        
        # Group Box
        # ---------
        gbox_knn = QGroupBox('KNN Parameters')
        gbox_singletest = QGroupBox('Single Test')
        gbox_transformation = QGroupBox('Transformation')
        gbox_includeddataset = QGroupBox('Included in dataset')
        gbox_dataset = QGroupBox('Dataset')       
                
        # HBox et VBox
        # ------------
        hbox_knnparam_max = QHBoxLayout()
        hbox_knnparam_k = QHBoxLayout()
        vbox_knnparam = QVBoxLayout(gbox_knn)
        vbox_singletest = QVBoxLayout(gbox_singletest)
        hbox_transformation_3 = QHBoxLayout()
        hbox_transformation_2 = QHBoxLayout()
        hbox_transformation_1 = QHBoxLayout()
        hbox_includeddataset_4 = QHBoxLayout()        
        hbox_includeddataset_3 = QHBoxLayout()
        hbox_includeddataset_2 = QHBoxLayout()
        hbox_includeddataset_1 = QHBoxLayout()
        vbox_transformation = QVBoxLayout(gbox_transformation)
        vbox_includeddataset = QVBoxLayout(gbox_includeddataset)
        hbox_dataset = QHBoxLayout()
        vbox_dataset = QVBoxLayout(gbox_dataset)
        vbox_allinfo = QVBoxLayout()
        vbox_KNNClassification = QVBoxLayout()
        hbox_main = QHBoxLayout(self)
        
        # Add widget & layout
        # -------------------
        hbox_transformation_1.add_widget(qlab_translated)
        hbox_transformation_1.add_widget(self.qlab_translated_tf)
        hbox_transformation_2.add_widget(qlab_rotated)
        hbox_transformation_2.add_widget(self.qlab_rotated_tf)
        hbox_transformation_3.add_widget(qlab_scaled)
        hbox_transformation_3.add_widget(self.qlab_scaled_tf)
        
        hbox_includeddataset_1.add_widget(qlab_catcount)
        hbox_includeddataset_1.add_widget(self.qlab_catcount_value)
        hbox_includeddataset_2.add_widget(qlab_trainimgcount)
        hbox_includeddataset_2.add_widget(self.qlab_trainimgcount_value)
        hbox_includeddataset_3.add_widget(qlab_testimgcount)
        hbox_includeddataset_3.add_widget(self.qlab_testimgcount_value)
        hbox_includeddataset_4.add_widget(qlab_totalimgcount)
        hbox_includeddataset_4.add_widget(self.qlab_totalimgcount_value)
        
        vbox_transformation.add_layout(hbox_transformation_1)
        vbox_transformation.add_layout(hbox_transformation_2)
        vbox_transformation.add_layout(hbox_transformation_3)
        
        vbox_includeddataset.add_layout(hbox_includeddataset_1)
        vbox_includeddataset.add_layout(hbox_includeddataset_2)
        vbox_includeddataset.add_layout(hbox_includeddataset_3)
        vbox_includeddataset.add_layout(hbox_includeddataset_4)
        
        hbox_dataset.add_widget(gbox_includeddataset)
        hbox_dataset.add_widget(gbox_transformation)
        
        vbox_dataset.add_widget(self.cbox_dataset)
        vbox_dataset.add_layout(hbox_dataset)
        
        vbox_singletest.add_widget(self.cbox_singletest)
        vbox_singletest.add_widget(self.qlab_image)
        vbox_singletest.add_widget(buttonClassify)
        vbox_singletest.add_widget(self.qlab_classify)        
        
        hbox_knnparam_k.add_widget(self.qlab_k)
        hbox_knnparam_k.add_widget(self.qslid_k)
        
        hbox_knnparam_max.add_widget(self.qlab_max)
        hbox_knnparam_max.add_widget(self.qslid_max)
                
        vbox_knnparam.add_layout(hbox_knnparam_k)
        vbox_knnparam.add_layout(hbox_knnparam_max)        
        
        vbox_allinfo.add_widget(gbox_dataset)
        vbox_allinfo.add_widget(gbox_singletest)
        vbox_allinfo.add_widget(gbox_knn)
        vbox_allinfo.add_widget(buttonAbout)
        
        vbox_KNNClassification.add_widget(self.qlab_matgraph)
        
        hbox_main.add_layout(vbox_allinfo)
        hbox_main.add_layout(vbox_KNNClassification)    

        # Mise en forme
        # ---------------
        qlab_catcount.alignment = Qt.AlignLeft  
        qlab_trainimgcount.alignment = Qt.AlignLeft
        qlab_testimgcount.alignment = Qt.AlignLeft
        qlab_totalimgcount.alignment = Qt.AlignLeft
        self.qlab_catcount_value.alignment = Qt.AlignRight        
        self.qlab_trainimgcount_value.alignment = Qt.AlignRight  
        self.qlab_testimgcount_value.alignment = Qt.AlignRight  
        self.qlab_totalimgcount_value.alignment = Qt.AlignRight        
        
        qlab_translated.alignment = Qt.AlignLeft
        qlab_rotated.alignment = Qt.AlignLeft
        qlab_scaled.alignment = Qt.AlignLeft
        self.qlab_translated_tf.alignment = Qt.AlignRight  
        self.qlab_rotated_tf.alignment = Qt.AlignRight  
        self.qlab_scaled_tf.alignment = Qt.AlignRight
        
        self.qlab_image.style_sheet = 'QLabel { background-color : #313D4A; padding : 10px 10px 10px 10px; }'
        self.qlab_image.alignment = Qt.AlignCenter
        
        self.qlab_classify.alignment = Qt.AlignCenter
        
        self.qlab_k.alignment = Qt.AlignCenter
        self.qlab_max.alignment = Qt.AlignCenter
        
        qlab_catcount.set_fixed_size(120, 20)
        qlab_trainimgcount.set_fixed_size(120, 20)
        qlab_testimgcount.set_fixed_size(120, 20)
        qlab_totalimgcount.set_fixed_size(120, 20)
        self.qlab_catcount_value.set_fixed_size(50, 20)        
        self.qlab_trainimgcount_value.set_fixed_size(50, 20)
        self.qlab_testimgcount_value.set_fixed_size(50, 20)
        self.qlab_totalimgcount_value.set_fixed_size(50, 20)       
        
        qlab_translated.set_fixed_size(120, 20)
        qlab_rotated.set_fixed_size(120, 20)
        qlab_scaled.set_fixed_size(120, 20)
        self.qlab_translated_tf.set_fixed_size(50, 20)
        self.qlab_rotated_tf.set_fixed_size(50, 20)
        self.qlab_scaled_tf.set_fixed_size(50, 20)
        
        self.qlab_k.set_fixed_size(100, 10)
        self.qlab_max.set_fixed_size(100, 10)
    
    # Init le Graph
    # ---------------
    def mathgraph_init(self):
        my_dpi = 100
        width, height = 550, 550
        figure = plt.figure(figsize=(width / my_dpi, height / my_dpi), dpi=my_dpi)
        figure.set_size_inches(width / my_dpi, height / my_dpi)  
        self.canvas = FigureCanvas(figure)         
        
        self.axis = figure.add_subplot(projection='3d')        
        self.axis.set_title('KNN Classification')
        self.axis.set_xlabel('x')
        self.axis.set_ylabel('y')
        self.axis.set_zlabel('z')
        self.axis.set_xlim3d(0, 1.25)
        self.axis.set_ylim3d(0, 1)
        self.axis.set_zlim3d(0, 1)    
    
    # Afficher le graph
    # ---------------
    def mathgraph(self):                
        self.canvas.draw()
        w, h = self.canvas.get_width_height()
        buffer = self.canvas.buffer_rgba() 
        image = QtGui.QImage(buffer, w, h, w * 4, QtGui.QImage.Format_ARGB32)
        self.qlab_matgraph.pixmap = QtGui.QPixmap.from_image(image)    
    
    @Slot()
    def _cbox_dataset_index_change(self):
        # Update data of dataset box
        # --------------------------
        cbox_dataset_index = self.cbox_dataset.current_index
        cbox_dataset_title = self.alldata[cbox_dataset_index, 1]
        
        if (self.alldata[cbox_dataset_index,2]):
            self.qlab_translated_tf.text = "True"
        else :
            self.qlab_translated_tf.text = "False"
            
        if (self.alldata[cbox_dataset_index,3]):
            self.qlab_rotated_tf.text = "True"
        else :
            self.qlab_rotated_tf.text = "False"
            
        if (self.alldata[cbox_dataset_index,4]):
            self.qlab_scaled_tf.text = "True"
        else :
            self.qlab_scaled_tf.text = "False"
            
        self.qlab_catcount_value.text = str(self.alldata[cbox_dataset_index,5])
        self.qlab_trainimgcount_value.text = str(self.alldata[cbox_dataset_index,6])
        self.qlab_testimgcount_value.text = str(self.alldata[cbox_dataset_index,7])
        self.qlab_totalimgcount_value.text = str(self.alldata[cbox_dataset_index,8])
        
        # Update data of single test QComboBox
        # ------------------------------------
        self.dataset_image = np.array(self.klustr_dao.image_from_dataset(cbox_dataset_title, False), dtype=object)
        self.cbox_singletest.clear()
        self.cbox_singletest.add_items(self.dataset_image[:,3])
        
        # Analyse du data  
        # ---------------------------------        
        # Aller chercher le dataset_test
        self.dataset_test = np.array(self.klustr_dao.image_from_dataset(cbox_dataset_title, True), dtype=object)
        
        # Return all Labels in dataset_test and assign a color and symbol        
        self.dataset_label, frequency = np.unique(self.dataset_test[:,1], return_counts = True)
        qte_label = np.count_nonzero(self.dataset_label)
        
        # Set K Value Max
        self.qslid_k.value = 1
        self.qslid_k.maximum = np.sum(frequency)
        
        # Set couleur
        couleur = np.random.rand(qte_label, 3)
        
        # Set marqueur
        marqueur = []        
        for i in self.dataset_label:            
            random_temp = np.random.rand()
            if random_temp < 0.33: 
                marqueur.append('+')
            elif random_temp < 0.66: 
                marqueur.append('x')
            else: 
                marqueur.append('o')
        
        # Retourner les valeurs KNN de la liste d'image
        self.knn_values_x, self.knn_values_y, self.knn_values_z = KlustEngine(self.dataset_test[:,6]).extraire_coord()
        
        # Afficher les points ds le graphique
        self.mathgraph_init()
        
        compteur_frequency = 0
        compteur_i = 0
        for i in frequency: 
            self.axis.scatter(self.knn_values_x[compteur_frequency:compteur_frequency+i], self.knn_values_y[compteur_frequency:compteur_frequency+i], self.knn_values_z[compteur_frequency:compteur_frequency+i], color=couleur[compteur_i], marker=marqueur[compteur_i])
            compteur_frequency += i
            compteur_i += 1

        self.mathgraph()
    
    @Slot()
    def _cbox_singletest_index_change(self):
        self.cbox_singletest_index = self.cbox_singletest.current_index
        self.singletest_image = qimage_argb32_from_png_decoding(self.dataset_image[self.cbox_singletest_index,6])
        self.qlab_image.pixmap = QPixmap.from_image(self.singletest_image)
        self.qlab_classify.text = "Not classified"
        
    @Slot()
    def _click_classify(self):
        self.knn_points = np.array([self.knn_values_x, self.knn_values_y, self.knn_values_z])
        self.img_value_x, self.img_value_y, self.img_value_z = KlustEngine([self.dataset_image[self.cbox_singletest_index,6]]).extraire_coord()
        self.img_points = np.array([self.img_value_x, self.img_value_y, self.img_value_z])
        self.knn = Knn(self.knn_points, self.dataset_test[:,1], self.img_points)
        self.qlab_classify.text = self.knn.knn_classifiy(self.k_value, self.max_value)
        
    @Slot()
    def _click_about(self):
        qmsgbox = QMessageBox()
        qmsgbox.set_window_title("Projet 1 - C52 - About")
        qmsgbox.text =  "{:<150}".format("Ce logiciel est le projet #1 du cours C52.") + "\n\n" + \
                        "{:<150}".format("Il a ete realise par :") + "\n" + \
                        "{:<150}".format("\t- Henri-Paul Bolduc") + "\n" + \
                        "{:<150}".format("\t- Ariel Hotz-Garber") + "\n" + \
                        "{:<150}".format("\t- Gael Lane Lepine") + "\n\n" + \
                        "{:<150}".format("Il consiste a faire une analyse d'image 2D avec les concept suivants :") + "\n" + \
                        "{:<150}".format("\t- Application de Numpy") + "\n" + \
                        "{:<150}".format("\t- Application de MatPlotLib") + "\n" + \
                        "{:<150}".format("\t- Application de Pyside6 et des widgets") + "\n" + \
                        "{:<150}".format("\t- Analyse selon les principes KNN") + "\n" + \
                        "{:<150}".format("\t- Normalisation des axes/metriques KNN") + "\n\n" + \
                        "{:<150}".format("Nos 3 descripteurs de formes sont :") + "\n" + \
                        "{:<150}".format("\t- Indice de complexite") + "\n" + \
                        "{:<150}".format("\t\t- Aucune unite pour le domaine 0-1.2") + "\n" + \
                        "{:<150}".format("\t\t- Correspont a (4*pi*aire) divise par le perimetre au carre ") + "\n" + \
                        "{:<150}".format("\t- Indice de compacite") + "\n" + \
                        "{:<150}".format("\t\t- Aucune unite pour le domaine 0-1") + "\n" + \
                        "{:<150}".format("\t\t- Correspont a l'aire de la forme divisee par l'aire du cercle") + "\n" + \
                        "{:<150}".format("\t\t  qui l'encapsule ") + "\n" + \
                        "{:<150}".format("\t- Moyenne centroid/perimetre") + "\n" + \
                        "{:<150}".format("\t\t- Aucune unite pour le domaine 0-1") + "\n" + \
                        "{:<150}".format("\t\t- Correspont a la moyenne de la somme des distances") + "\n" + \
                        "{:<150}".format("\t\t  centroid/perimetre divise par le rayon qui l'encapsule") + "\n\n" + \
                        "{:<150}".format("Plus precisement, ce laboratoire permet de mettre en pratique les notions de :") + "\n" + \
                        "{:<150}".format("\t- Calcul vectoriel avec Numpy") + "\n" + \
                        "{:<150}".format("\t- Programmation oriente objet") + "\n\n" + \
                        "{:<150}".format("Un effort d'abstraction a ete fait pour ces points :") + "\n" + \
                        "{:<150}".format("\t- Generalisation du code KNN") + "\n" + \
                        "{:<150}".format("\t- Automatisation des couleurs et des marqueurs ds le graphique") + "\n\n" + \
                        "{:<150}".format("Finalement, l'ensemble de donnees le plus complexe que nous avons ete capable de resoudre est :") + "\n" + \
                        "{:<150}".format("\t- Zoo-Large")   
                        
        qmsgbox.exec()
        
    @Slot()
    def _qslid_k_change(self):
        self.k_value = self.qslid_k.value
        self.qlab_k.text = "K = " + str(self.k_value)
        
    @Slot()
    def _qslid_max_change(self):
        self.max_value = self.qslid_max.value/10
        self.qlab_max.text = "Max = " + str(self.max_value)
# -------------------------------------------------------------