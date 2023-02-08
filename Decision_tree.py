
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 11:42:54 2022

@author: HP
"""
##################
# Bibliothèques :#
 #################   
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV




#########################
# Importer les données :#
 ########################
donnees = pd.read_excel("C:/Users/HP/Desktop/AssuranceData.xlsx")
donnees.head()    # ou donnees[0:5]
donnees.info()   # pour savoir plusieurs informations sur notre 



##########################################################
# Extraire les attributs avec des valeurs non numériques :#
###########################################################
Type_Dassurance=donnees.values[:,4] # comme ça on aura des arrays dont on peut appliquer les fonctions de numpy
Type_Dassurance=Type_Dassurance.reshape(len(Type_Dassurance),1)

Job=donnees.values[:,5]
Job=Job.reshape(len(Job),1)

Situation_Familiale=donnees.values[:,6]
Situation_Familiale=Situation_Familiale.reshape(len(Situation_Familiale),1)

#print (type(Type_Dassurance))
#print(Type_Dassurance)


###############################################
# Ecodage Binaire des données non numériques :#
###############################################


# 1 : Type_Dassurance
onehot_encoder_Type_Dassurance = OneHotEncoder(sparse=False)    
onehot_encoded_Type_Dassurance = onehot_encoder_Type_Dassurance.fit_transform(Type_Dassurance)
#print(onehot_encoded_Type_Dassurance)


#2 : Job
onehot_encoder_Job = OneHotEncoder(sparse=False)
onehot_encoded_Job = onehot_encoder_Job.fit_transform(Job)
#print(onehot_encoded_Job)


# 3 : Situation_Familiale
onehot_encoder_Situation_Familiale = OneHotEncoder(sparse=False)
onehot_encoded_Situation_Familiale = onehot_encoder_Situation_Familiale.fit_transform(Situation_Familiale)
#print(onehot_encoded_Situation_Familiale)


#######################################################################
# Reconstruire le tableau de données qu'avec des features numériques :#
#######################################################################

Target=donnees.values[:,-1].reshape(len(donnees.values[:,-1]),1)
New_donnees=np.hstack((onehot_encoded_Type_Dassurance,onehot_encoded_Job,onehot_encoded_Situation_Familiale,Target)) # merge des données transférer et la dernière colonne
donneesFinal=np.hstack((donnees.values[:,0:4],New_donnees))  # construire enfin un tableau de donées numériques
donneesFinal[0:5,:]

#######################################################
#Séparation du variable cible à des autres variables :#
#######################################################

features_classes=donneesFinal[:,0:-1]
Cible_classe=donneesFinal[:,-1]
#features_classes[0:3]
#Cible_classe[0:15]

#########################################
##### creation de modèle de préduction :#
#########################################
x_train, x_test, y_train, y_test  = train_test_split(features_classes,Cible_classe,test_size=0.25,random_state=42)

Notre_modele = DecisionTreeClassifier(criterion="gini",max_depth=7) #Déclaration de l'arbre de décision
Notre_modele.fit(x_train,y_train)

##################################
# La validation de notre modèle :#
##################################

# cross validation :
cross_val_score(DecisionTreeClassifier(),x_train,y_train,cv=5).mean()
k=np.arange(1,10)
train_score,val_score=validation_curve(DecisionTreeClassifier(random_state=42),x_train,y_train,'max_depth',k,cv=5)
plt.plot(k,val_score.mean(axis=1)) 
#pour motrer quelle valeur de max_depth va nous donner le mielleur score.
# d'après la courbe, on constate que la valeur 5 de l'hyperparamètre "max_depth" va nous donner le meilleurs score.
param_crid={'criterion':['gini','entropy'],'max_depth':np.arange(1,10)}
grid=GridSearchCV(DecisionTreeClassifier(max_depth=7),param_crid,cv=5)
grid.fit(x_train,y_train) 

print(grid.best_score_) # pour avoir le best score :ça m'a donné 0.9336666666666666
print(grid.best_params_) # pour savoir les best valeurs des hyperparamètres, ça m'a donné : {'criterion': 'entropy', 'max_depth': 7}
# Donc on conserve le modèle avec les meilleurs paramères avec la commande suivante :
Notre_best_modele=grid.best_estimator_ 


#########
# test :#
#########
from sklearn.metrics import  confusion_matrix  #,accuracy_score 
confusion_matrix(y_test,Notre_best_modele.predict(x_test))
# ou Notre_best_modele.score(x_test,y_test)



#############################################################
#Affichage de l'abre de décision obtenu après entraînement :#
#############################################################
#plot_tree(Notre_best_modele, feature_names= ['Age','Revenu mensuel en euro','Cotisation annuelle en euros','Duree Contrat par jour','Type d Assurance','Profession','Situation Familiale'], class_names=["Oui","Non"],filled=True)
#plt.show()

######################################################
#affichage plus grand pour une meilleure lisibilité :#
######################################################
#import matplotlib.pyplot as plt
plt.figure(figsize=(40,40))
plot_tree(Notre_best_modele,feature_names= ['Age','Revenu mensuel en euro','Cotisation annuelle en euros','Duree Contrat par jour','Type d Assurance1','Type d Assurance2','Type d Assurance3','Profession1','Profession2','Profession2','Profession3','Profession4','Situation Familiale1','Situation Familiale2'], class_names=["Oui","Non"],filled=True)
plt.show()


##############################################################################
#Fonction de prédiction : Se servir de l'algorithme pour faire une prédiction#
############################################################################## 
def Prediction(Age,Revenu_mensuel_en_euro,Cotisation_annuelle_en_euros,Duree_contrat_par_jour,Type_D_Assurance,Profession,Situation_familiale):
    A_predire=[0 for i in range(13)]
    A_predire[0]=Age
    A_predire[1]=Revenu_mensuel_en_euro
    A_predire[2]=Cotisation_annuelle_en_euros
    A_predire[3]=Duree_contrat_par_jour
    if Type_D_Assurance=="auto":
        A_predire[4]=1.0
        A_predire[5]=0.0
        A_predire[6]=0.0
    if Type_D_Assurance=="auto_temporaire":
        A_predire[4]=0.0
        A_predire[5]=1.0
        A_predire[6]=0.0
    if Type_D_Assurance=="mutuelle":
        A_predire[4]=0.0
        A_predire[5]=0.0
        A_predire[6]=1.0
    if Profession=="Salarié":
        A_predire[7]=0.0
        A_predire[8]=0.0
        A_predire[9]=0.0
        A_predire[10]=1.0
    if Profession=="Retraité":
        A_predire[7]=0.0
        A_predire[8]=0.0
        A_predire[9]=1.0
        A_predire[10]=0.0
    if Profession=="Cadre":
        A_predire[7]=1.0
        A_predire[8]=0.0
        A_predire[9]=0.0
        A_predire[10]=0.0
    if Profession=="Etudiant":
        A_predire[7]=0.0
        A_predire[8]=1.0
        A_predire[9]=0.0
        A_predire[10]=0.0
    if Situation_familiale=="Marié":
        A_predire[11]=0.0
        A_predire[12]=1.0
    if Situation_familiale=="Célibataire":
        A_predire[11]=1.0
        A_predire[12]=0.0
    A_predire=np.array(A_predire)
    A_predire=A_predire.reshape(1,len(A_predire))
    return Notre_best_modele.predict(A_predire)

Prediction(33,3000,300,365,"mutuelle","Cadre","Marié")