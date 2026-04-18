# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:09:55 2022

@author: HP
"""



##### Bibliothèques :
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree




###### Importer les données :
donnees = pd.read_excel("C:/Users/HP/Desktop/AssuranceData.xlsx")
donnees.head()    # ou donnees[0:5]
#donnees.info()   # pour savoir plusieurs informations sur notre Data
"""
def ImporterDonnees (NomFichier="C:/Users/HP/Desktop/AssuranceData.xlsx"):
    donnees = pd.read_excel(NomFichier)
    return donnees.head()    # ou donnees[0:5]
    #donnees.info()   # pour savoir plusieurs informations sur notre Data
ImporterDonnees() # pour tester
"""




########## Extraire les attributs avec des valeurs non numériques :
Type_Dassurance=donnees.values[:,4]
Job=donnees.values[:,5]
Situation_Familiale=donnees.values[:,6]
#print (type(Type_Dassurance))
#print(Type_Dassurance)


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
Target=donnees.values[:,-1].reshape(len(donnees.values[:,-1]),1)
New_donnees=np.hstack((onehot_encoded_Type_Dassurance,onehot_encoded_Job,onehot_encoded_Situation_Familiale,Target)) # merge des données transférer et la dernière colonne
donneesFinal=np.hstack((donnees.values[:,0:4],New_donnees))  # construire enfin un tableau de donées numériques


#Séparation du variable cible à des autres variables :
features_classes=donneesFinal[:,0:-1]
Cible_classe=donneesFinal[:,-1]
#features_classes[0:3]
#Cible_classe[0:15]






##### creation de modèle de préduction 
Notre_modele = DecisionTreeClassifier(criterion="gini",max_depth=7) #Déclaration de l'arbre de décision
Notre_modele.fit(features_classes,Cible_classe)
#Affichage de l'abre de décision obtenu après entraînement
plot_tree(Notre_modele, feature_names= ['Age','Revenu mensuel en euro','Cotisation annuelle en euros','Duree Contrat par jour','Type d Assurance','Profession','Situation Familiale'], class_names=["Oui","Non"],filled=True)
plt.show()

#affichage plus grand pour une meilleure lisibilité
import matplotlib.pyplot as plt
plt.figure(figsize=(40,40))
plot_tree(Notre_modele,feature_names= ['Age','Revenu mensuel en euro','Cotisation annuelle en euros','Duree Contrat par jour','Type d Assurance','Profession','Situation Familiale'], class_names=["Oui","Non"],filled=True)
plt.show()



#Test 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test  = train_test_split(features_classes,Cible_classe,test_size=0.25,random_state=42)

'''
# cross validation :
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
cross_val_score(RandomForestClassifier(),x_train,y_train,cv=5).mean()
k=np.arange(1,10)
train_score,val_score=validation_curve(DecisionTreeClassifier(random_state=42),x_train,y_train,'max_depth',k,cv=5)
plt.plot(k,val_score.mean(axis=1)) #pour motrer quelle valeur de max_depth va nous donner le mielleur score.




# pour avoir encore une mielleur performance on peut chercher la milleur valeur pour les 
#autres hyperparamètres (là on a un seul autre param :criterion) à l'aide de gridsearchcv'

from sklearn.model_selection import GridSearchCV
param_crid={'criterion':['gini','entropy'],'max_depth':np.arange(1,10)}
grid=GridSearchCV(DecisionTreeClassifier,param_crid,cv=5)
grid.fit(x_train,y_train) 

'''
import matplotlib.pyplot as plt
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
#cross_val_score(RandomForestClassifier(),x_train,y_train,cv=5).mean()
k=np.arange(1,10)
train_score,val_score=validation_curve(DecisionTreeClassifier(random_state=42),x_train,y_train,'max_depth',k,cv=5)
plt.plot(k,val_score.mean(axis=1)) #pour motrer quelle valeur de max_depth va nous donner le mielleur score.


from sklearn.model_selection import GridSearchCV
param_crid={'criterion':['gini','entropy'],'max_depth':np.arange(1,10)}
grid=GridSearchCV(DecisionTreeClassifier(max_depth=7),param_crid,cv=5)
grid.fit(x_train,y_train) 


from sklearn.model_selection import learning_curve
N,train_score,val_score=learning_curve(DecisionTreeClassifier(max_depth=7),x_train,y_train,train_sizes=np.linspace(0.2,1.0,10) ,cv=5)
print("N est :", N)


#Lepourcentage de bien classés :
from sklearn.metrics import accuracy_score #, confusion_matrix
print(f"Le pourcentage de bien classés est de : {accuracy_score(y_test, Notre_modele.predict(x_test))*100} %")

