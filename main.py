# -------------------------------------------------------------
# Auteur  : Henri-Paul Bolduc
#           Ariel Hotz-Garber
#           Gael Lane Lepine
# Cours   : 420-C52-IN - AI 1
# TP 1    : Analyse KNN des images
# Fichier : main.py
# -------------------------------------------------------------

# Importation des modules
# -------------------------------------------------------------
import sys
from db_credential import PostgreSQLCredential
from klustr_dao import PostgreSQLKlustRDAO
#from klustr_widget import KlustRDataSourceViewWidget
from analyse_widget import Projet1ViewWidget
from PySide6.QtWidgets import QApplication

# MAIN
# -------------------------------------------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)

    credential = PostgreSQLCredential(host='jcd-prof-cvm-69b5.aivencloud.com', port=11702, database='data_kit', user='klustr_reader', password='h$2%1?')
    #credential = 1
    klustr_dao = PostgreSQLKlustRDAO(credential)
    #source_data_widget = KlustRDataSourceViewWidget(klustr_dao)
    source_data_widget = Projet1ViewWidget(klustr_dao)
    
    source_data_widget.show()
    source_data_widget.setWindowTitle("Projet 1 - C52 - KNN Classifier")

    sys.exit(app.exec_())
# -------------------------------------------------------------