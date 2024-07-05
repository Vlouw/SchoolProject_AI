# SchoolProject_AI

Ce logiciel est le projet #1 du cours C52.

Il consiste a faire une analyse d'image 2D avec les concept suivants :
- Application de Numpy
- Application de MatPlotLib
- Application de Pyside6 et des widgets
- Analyse selon les principes KNN
- Normalisation des axes/metriques KNN

Nos 3 descripteurs de formes sont :
- Indice de complexite
    - Aucune unite pour le domaine 0-1
    - Correspont a (4*pi*aire) divise par le perimetre au carre

- Indice de compacite
    - Aucune unite pour le domaine 0-1
    - Correspont a l'aire de la forme divisee par l'aire du cercle qui l'encapsule

- Moyenne centroid/perimetre
    - Aucune unite pour le domaine 0-1
    - Correspont a la moyenne de la somme des distances centroid/perimetre divise par le rayon qui l'encapsule

Plus precisement, ce laboratoire permet de mettre en pratique les notions de :
- Calcul vectoriel avec Numpy
- Programmation oriente objet

Un effort d'abstraction a ete fait pour ces points :
    - Generalisation du code KNN
    - Automatisation des couleurs et des marqueurs ds le graphique

Finalement, l'ensemble de donnees le plus complexe que nous avons ete capable de resoudre est :
    - Zoo-Large
