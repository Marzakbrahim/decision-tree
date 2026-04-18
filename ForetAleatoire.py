# -*- coding: utf-8 -*-
"""
Created on Sun May 22 12:41:11 2022

@author: HP
"""




##### Bibliothèques :
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



###### Importer les données :
donnees = pd.read_excel("C:/Users/HP/Desktop/AssuranceData.xlsx")
donnees.head()    # ou donnees[0:5]





########## Extraire les attributs avec des valeurs non numériques :
Type_Dassurance=donnees.values[:,4]
Job=donnees.values[:,5]
Situation_Familiale=donnees.values[:,6]




########## Encodage entier des données non numériques : 
from sklearn.preprocessing import LabelEncoder


# 1 : Type_Dassurance
label_encoderonehot_Type_Dassurance = LabelEncoder()
integer_encoded_Type_Dassurance = label_encoderonehot_Type_Dassurance.fit_transform(Type_Dassurance)
integer_encoded_Type_Dassurance = integer_encoded_Type_Dassurance.reshape(len(integer_encoded_Type_Dassurance), 1)
#print(integer_encoded_Type_Dassurance)

#2 : Job
label_encoder_Job = LabelEncoder()
integer_encoded_Job = label_encoder_Job.fit_transform(Job)
integer_encoded_Job = integer_encoded_Job.reshape(len(integer_encoded_Job), 1)
#print(integer_encoded_Job)


# 3 : Situation_Familiale
label_encoder_Situation_Familiale = LabelEncoder()
integer_encoded_Situation_Familiale = label_encoder_Situation_Familiale.fit_transform(Situation_Familiale)
integer_encoded_Situation_Familiale=integer_encoded_Situation_Familiale.reshape(len(integer_encoded_Situation_Familiale),1)
#print(integer_encoded_Situation_Familiale)



########## # Ecodage Binaire :
from sklearn.preprocessing import OneHotEncoder

# 1 : Type_Dassurance
onehot_encoder_Type_Dassurance = OneHotEncoder(sparse=False)    
onehot_encoded_Type_Dassurance = onehot_encoder_Type_Dassurance.fit_transform(integer_encoded_Type_Dassurance)
#print(onehot_encoded_Type_Dassurance)


#2 : Job
onehot_encoder_Job = OneHotEncoder(sparse=False)
onehot_encoded_Job = onehot_encoder_Job.fit_transform(integer_encoded_Job)
#print(onehot_encoded_Job)


# 3 : Situation_Familiale
onehot_encoder_Situation_Familiale = OneHotEncoder(sparse=False)
onehot_encoded_Situation_Familiale = onehot_encoder_Situation_Familiale.fit_transform(integer_encoded_Situation_Familiale)
#print(onehot_encoded_Situation_Familiale)



########## Reconstruire le tableau de données qu'avec des features numériques :
donneesCible=donnees.values[:,-1].reshape(len(donnees.values[:,-1]),1)
donneesNew=np.hstack((onehot_encoded_Type_Dassurance,onehot_encoded_Job,onehot_encoded_Situation_Familiale,donneesCible))
donneesFinal=np.hstack((donnees.values[:,0:4],donneesNew)) 



#Séparation du target et features :
features_classes=donneesFinal[:,0:7]
Cible_classe=donneesFinal[:,-1]

#séparer les données : données pour entrainer le modèles et données pour tester et valider :
x_train, x_test, y_train, y_test  = train_test_split(features_classes,Cible_classe,test_size=0.25,random_state=42)

#train the model :
modele_rf = RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=None,min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0.0,max_features='auto',max_leaf_nodes=None,min_impurity_decrease=0.0,bootstrap=True,oob_score=False,n_jobs=None,random_state=None,verbose=0,warm_start=False,class_weight=None,ccp_alpha=0.0,max_samples=None,)
modele_rf.fit(x_train, y_train)

#tester le modèle :
print("test score :",modele_rf.score(x_test,y_test) ) # le résultat était 0.66666666 %




# L'importance des variables de notre modèle :
pd.DataFrame(modele_rf.feature_importances_,index =['Age', 'Revenu mensuel en euro', 'Cotisation annuelle en euros','Duree Contrat par jour', 'Type d Assurance', 'Profession','Situation Familiale'], columns = ["importance"]).sort_values("importance", ascending = False)

#Lepourcentage de bien classés :
from sklearn.metrics import accuracy_score  #, confusion_matrix
print(f"Le pourcentage de bien classés est de : {accuracy_score(y_test, modele_rf.predict(x_test))*100} %")