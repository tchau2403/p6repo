# -*- coding: utf-8 -*-
"""
Created on Thu May  3 08:46:17 2018

@author: Thierry CHAUVIER
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

class T_roc_curve():
    """
    Return metrics on ROC curve for a binary classifier
    """

    def __init__(self,y_test, y_pred_proba):
        self.nom = "t_confusion_matrix"

        
        '''
         On peut résumer la courbe ROC par un nombre : "l'aire sous la courbe", aussi dénotée AUROC pour « Area Under the ROC », 
         qui permet plus aisément de comparer plusieurs modèles.
         Un classifieur parfait a une AUROC de 1 ; un classifieur aléatoire, une AUROC de 0.5
        '''
        y_pred_proba2 = y_pred_proba[:, 1]
        [self.fpr, self.tpr, self.thr] = metrics.roc_curve(y_test, y_pred_proba2)
        self.AUROC = metrics.auc(self.fpr, self.tpr)
        
    def graph(self):
            plt.plot(self.fpr, self.tpr, color='coral', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.title('ROC Curve')
            plt.xlabel('1 - specificite', fontsize=14)
            plt.ylabel('Sensibilite', fontsize=14)
            print('AUROC = ',self.AUROC)
        


        


class T_confusion_matrix():
    """
    Return metrics on confusion matrix for a binary classifier
    """

    def __init__(self,y_test, y_pred):
        self.nom = "t_confusion_matrix"

        self.cnf_matrix=metrics.confusion_matrix(y_test, y_pred)
        self.TN=self.cnf_matrix[0,0]
        self.FN=self.cnf_matrix[1,0]
        self.FP=self.cnf_matrix[0,1]
        self.TP=self.cnf_matrix[1,1]
        
        '''
        Le rappel ("recall"  en anglais), ou sensibilité ("sensitivity" en anglais), 
        est le taux de vrais positifs, c’est à dire la proportion de positifs 
        que l’on a correctement identifiés. 
        C’est la capacité de notre modèle à détecter tous les incendies.
        '''
        self.recall=self.TP/(self.TP+self.FN) 
        
        '''
        la précision, c’est-à-dire la proportion de prédictions correctes parmi 
        les points que l’on a prédits positifs. 
        C’est la capacité de notre modèle à ne déclencher d’alarme que pour un vrai incendie.
        '''
        self.precision=self.TP/(self.TP+self.FP)
        
        '''
         la "F-mesure", qui est leur moyenne harmonique.
        '''
        self.f_mesure=(2*self.TP)/((2*self.TP)+self.FP+self.FN)
        
        '''
        la spécificité ("specificity" en anglais), qui est le taux de vrais négatifs, 
        autrement dit la capacité à détecter toutes les  situations où il n’y a pas d’incendie. 
        C’est une mesure complémentaire de la sensibilité.     
        '''
        self.specificity=self.TN/(self.FP+self.TN)
        
        
    def graph(self):
        from mlxtend.plotting import plot_confusion_matrix
        fig, ax = plot_confusion_matrix(conf_mat=self.cnf_matrix,figsize=(20,20))
        plt.title('Confusion Matrix')
        plt.show()
        print("True Negative=",self.TN)
        print("True Positive=",self.TP)
        print("False Negative=",self.FN)
        print("False Positive=",self.FP)
        print("recall TP/(TP+FN) = ",self.recall)
        print("precision TP/(TP+FP)= ",self.precision)
        print("f_mesure (2*TP)/((2*TP)+FP+FN)= ",self.f_mesure)
        print("specificity TN/(FP+TN) = ",self.specificity)


def t_confusion_matrix(y_test, y_pred,aff=1):
    """
    Return metrics on confusion matrix for a binary classifier
    """
    cnf_matrix=metrics.confusion_matrix(y_test, y_pred)
    TN=cnf_matrix[0,0]
    FN=cnf_matrix[1,0]
    FP=cnf_matrix[0,1]
    TP=cnf_matrix[1,1]
    
    '''
    Le rappel ("recall"  en anglais), ou sensibilité ("sensitivity" en anglais), 
    est le taux de vrais positifs, c’est à dire la proportion de positifs 
    que l’on a correctement identifiés. 
    C’est la capacité de notre modèle à détecter tous les incendies.
    '''
    recall=TP/(TP+FN) 
    
    '''
    la précision, c’est-à-dire la proportion de prédictions correctes parmi 
    les points que l’on a prédits positifs. 
    C’est la capacité de notre modèle à ne déclencher d’alarme que pour un vrai incendie.
    '''
    precision=TP/(TP+FP)
    
    '''
     la "F-mesure", qui est leur moyenne harmonique.
    '''
    f_mesure=(2*TP)/((2*TP)+FP+FN)
    
    '''
    la spécificité ("specificity" en anglais), qui est le taux de vrais négatifs, 
    autrement dit la capacité à détecter toutes les  situations où il n’y a pas d’incendie. 
    C’est une mesure complémentaire de la sensibilité.     
    '''
    specificity=TN/(FP+TN)
    
    if aff > 0:

        from mlxtend.plotting import plot_confusion_matrix
        fig, ax = plot_confusion_matrix(conf_mat=cnf_matrix)
        plt.show()
        print("True Negative=",TN)
        print("True Positive=",TP)
        print("False Negative=",FN)
        print("False Positive=",FP)
        print("recall TP/(TP+FN) = ",recall)
        print("precision TP/(TP+FP)= ",precision)
        print("f_mesure (2*TP)/((2*TP)+FP+FN)= ",f_mesure)
        print("specificity TN/(FP+TN) = ",specificity)
    return(recall,specificity,precision)


def t_variance(X_features,X_components,nb,seuil=0.8,aff=1):
    """
    Calculate variance after dimension reduction
    """
    tot_var_X = np.sum(np.var(X_features,axis=0)) # variance des features
    tot_var_comp_nb = np.sum(np.var(X_components[:,0:nb],axis=0)) # cumul de la variance des N premières composantes principales
    tot_var_comp = np.sum(np.var(X_components,axis=0)) # cumul de la variance des composantes principales
    explained_variance = np.var(X_components,axis=0) # variance expliquée pour chaque composantes principales
    if np.sum(explained_variance) == 0:
        explained_variance_ratio = 0
    else:
        explained_variance_ratio = explained_variance / np.sum(explained_variance) # ratio de la variance expliquée pour chaque composantes principales
    explained_variance_ratio_cumsum = np.cumsum(explained_variance_ratio) # somme cumulée des ratios de la variance expliquée des composantes principales
    nb_comp = explained_variance_ratio_cumsum.shape[0]
    
    # recherche du nombre minimale de composantes principale pour atteindre le seuil demandé
    for i in range(0,explained_variance_ratio_cumsum.shape[0]):
        if explained_variance_ratio_cumsum[i] > seuil:
            nb_comp = i
            break
            
    explained_variance_ratio_cumsum = np.round(explained_variance_ratio_cumsum,2)
    if tot_var_comp == 0:
        pct_exp = 0
    else:
        pct_exp=tot_var_comp_nb/tot_var_comp # ratio de la variance expliquée par les N premières composantes principales
    if aff > 0:
        print("Nombre des features de X = ",X_features.shape[1])
        print("Total variance des features = %0.03f"%(tot_var_X))
        print("Total variance des composantes principales = %0.03f"%(tot_var_comp))
        print("Total variance des %i premières composantes principales = %0.03f"%(nb,tot_var_comp_nb))
        print("Pourcentage de la variance expliquée par les %i premières composantes principales = %0.03f "%(nb,pct_exp))
        print("Il faut les %i premières composantes principales pour expliquer au moins %0.00f pct de la variance"%(nb_comp+1,seuil*100))
        plt.figure(figsize=(5,5))
        plt.plot(explained_variance_ratio_cumsum)
        plt.ylabel("% explained variance ration ")
        plt.xlabel("Nb components")
        plt.show()
    return(nb_comp+1,pct_exp)

def t_is_numeric(obj):
    """
        This function test if arg is numeric
    """
    attrs = ['__add__', '__sub__', '__mul__', '__truediv__', '__pow__']
    return all(hasattr(obj, attr) for attr in attrs)

def t_analyze(df):
    """
        this function will analyze each columns with different metrics
    """
    # Create dataframe which will recieve metrics

    v_columns=['nb_lignes','nb_lignes_distinctes','nb_doublons','nb_nan','pct_nan','nb_val_num','pct_val_num','nb_val_alpha','pct_val_alpha',
               'Max','Min',"Ecart Type","Moyenne","quantile_25","quantile_50","quantile_75"]
    result=pd.DataFrame(np.zeros((len(df.columns.values),len(v_columns))), 
                        index=df.columns.values, 
                        columns=v_columns)

    
    for ind,col in enumerate(df.columns.values):
        print("Column : {} : {}".format(ind,col))
        result['nb_lignes'][col] = len(df[col])
        result['nb_lignes_distinctes'][col] = len(df[col].value_counts())
        try:
            result['nb_doublons'][col] = len(df[col][df.groupby(col)[col].transform('count') > 1].unique())
        except:
            pass
        result['nb_nan'][col] = df[col].isnull().values.sum()
        result['nb_val_num'][col] = df[col][df[col].apply(lambda x: t_is_numeric(x))].count()
        result['nb_val_alpha'][col] = result['nb_lignes'][col] - result['nb_val_num'][col] - result['nb_nan'][col]
        
        result['pct_nan'][col]=result['nb_nan'][col]/result['nb_lignes'][col]
        result['pct_val_num'][col]=result['nb_val_num'][col]/result['nb_lignes'][col]
        result['pct_val_alpha'][col]=result['nb_val_alpha'][col]/result['nb_lignes'][col]
        
        
        # calcul des statistiques
        
        if result['nb_val_alpha'][col] == 0:
            result['Max'][col]=df[col].max()
            result['Min'][col]=df[col].min()
            result['Ecart Type'][col]=df[col].std()
            result['Moyenne'][col]=df[col].mean()
            result['quantile_25'][col]=df[col].quantile(q=0.25)
            result['quantile_50'][col]=df[col].quantile(q=0.5)
            result['quantile_75'][col]=df[col].quantile(q=0.75)
        else:
            result['Ecart Type'][col]=np.nan
            result['Max'][col]=np.nan
            result['Min'][col]=np.nan
            result['Moyenne'][col]=np.nan
            result['quantile_25'][col]=np.nan
            result['quantile_50'][col]=np.nan
            result['quantile_75'][col]=np.nan
    
    # detect values data types
    
    dtypeCount =[df.iloc[:,i].apply(type).value_counts() for i in range(df.shape[1])]
    dtcount = pd.DataFrame(dtypeCount)
    dtcount.columns=dtcount.columns.astype(str)
    result = pd.concat([result,dtcount], axis=1)
    result = pd.concat([result,df.dtypes], axis=1)
    result.rename(columns={0:'pandas_dtypes'}, inplace=True)
    result['pandas_dtypes']=result['pandas_dtypes'].astype('str')
    result.columns=result.columns.str.replace("'","")
    result.columns=result.columns.str.replace("<","")
    result.columns=result.columns.str.replace(">","")

    print ("End of analyze : ")

    return(result)
    
    
def main():
    pass

# =============================================================================
#  Start run
# =============================================================================

if __name__ == "__main__":
    main()