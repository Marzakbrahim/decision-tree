##################
# Bibliothèques :#
 #################   
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder ,LabelEncoder
from sklearn.model_selection import train_test_split
from joblib import dump, load
from sklearn.pipeline import  make_pipeline
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler


#########################
# Importer les données :#
 ########################
donnees = pd.read_excel("C:/Users/HP/Desktop/AssuranceData.xlsx")
donnees.head()    # ou donnees[0:5]
#donnees.info()   # pour savoir plusieurs informations sur notre 


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


# 4 : Client ou pas (encodage entier):
Client_ou_pas=donnees.values[:,-1]
Tagrget= LabelEncoder()
integer_encoded_Target = Tagrget.fit_transform(Client_ou_pas)
integer_encoded_Target = integer_encoded_Target.reshape(len(integer_encoded_Target), 1)
#print(integer_encoded_Target)




#######################################################################
# Reconstruire le tableau de données qu'avec des features numériques :#
#######################################################################

#Target=donnees.values[:,-1].reshape(len(donnees.values[:,-1]),1)
Features=np.hstack((donnees.values[:,0:4],onehot_encoded_Type_Dassurance,onehot_encoded_Job,onehot_encoded_Situation_Familiale)) # merge des données transférer et la dernière colonne



#########################################
##### creation de modèle de préduction :#
#########################################
x_train, x_test, y_train, y_test  = train_test_split(Features,integer_encoded_Target,test_size=0.25,random_state=42)

modele_rf = RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=None,min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0.0,max_features='auto',max_leaf_nodes=None,min_impurity_decrease=0.0,bootstrap=True,oob_score=False,n_jobs=None,random_state=None,verbose=0,warm_start=False,class_weight=None,ccp_alpha=0.0,max_samples=None)
modele_rf.fit(x_train, y_train)


#tester le modèle :
print("test score :",modele_rf.score(x_test,y_test) ) # le résultat était 0.66666666 %



##############################################################
##### Intégrer la procédure dans un pipeline et l'utiliser. :#
##############################################################
X=donnees.iloc[:,0:-1]
Y=donnees.iloc[:,-1]
x_train, x_test, y_train, y_test  = train_test_split(X,Y,test_size=0.25,random_state=42)

features_qualitatives=x_train.iloc[:,4:7].columns
features_quantitatives=x_train.iloc[:,0:4].columns

#Pipeline :

pipeline_1=make_pipeline(MinMaxScaler())
pipeline_2=make_pipeline(OneHotEncoder(sparse=False))
pipeline_3=make_pipeline(LabelEncoder())
preprocessor=make_column_transformer((pipeline_1,features_quantitatives),(pipeline_2,features_qualitatives)) #,(pipeline_3,y_train)
model=make_pipeline(preprocessor,modele_rf)
model.fit(x_train,y_train)
model.score(x_test,y_test)




dump(model,"C:/Users/HP/Desktop/dossier_assur2.joblib")
Notre_model=load("C:/Users/HP/Desktop/dossier_assur2.joblib")
test=pd.read_excel("C:/Users/HP/Desktop/Classeur3.xlsx")
Notre_model.predict(test)

