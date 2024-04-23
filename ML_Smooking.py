# Import des bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# Charger les données d'entraînement et de test
data_entrainement = pd.read_csv('/kaggle/input/emsi-tabular-2024/train.csv')
data_test = pd.read_csv('/kaggle/input/emsi-tabular-2024/test.csv')

# Supprimer la colonne 'id' des données d'entraînement
data_entrainement.drop(columns=['id'], inplace=True)

# Définir les variables explicatives et la cible
variables_explicatives = data_entrainement.drop(columns=['smoking'])
cible = data_entrainement['smoking']

# Diviser les données en ensembles d'entraînement et de test
explicatives_entrainement, explicatives_test, cible_entrainement, cible_test = train_test_split(variables_explicatives, cible, test_size=0.2, random_state=100)

# Initialiser le modèle de classification Gradient Boosting
modele = GradientBoostingClassifier()

# Entraîner le modèle sur les données d'entraînement
modele.fit(explicatives_entrainement, cible_entrainement)

# Faire des prédictions sur l'ensemble de test
predictions = modele.predict(explicatives_test)

# Calculer le score ROC AUC
score_roc_auc = roc_auc_score(cible_test, predictions)
print(f"Score ROC AUC : {score_roc_auc}")

# Faire des prédictions sur les données de test
predictions_test = modele.predict(data_test.drop(columns=['id']))

# Créer un dataframe pour soumission
soumission = pd.DataFrame({'id': data_test['id'], 'smoking': predictions_test})

# Enregistrer les prédictions dans un fichier CSV
soumission.to_csv('soumission.csv', index=False)