import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import confusion_matrix,cohen_kappa_score


def nan_replace(x: np.ndarray):
    is_nan = np.isnan(x)
    k = np.where(is_nan)
    x[k] = np.nanmean(x[:, k[1]], axis=0)

def nan_replace_t(t:pd.DataFrame):
    for coloana in t.columns:
        if t[coloana].isna().any():
            if is_numeric_dtype(t[coloana]):
                t.fillna({coloana:t[coloana].mean()},inplace=True)
            else:
                t.fillna({coloana:t[coloana].mode()[0]},inplace=True)

def calcul_metrici(y,y_,clase):
    cm = confusion_matrix(y,y_,labels=clase)
    t_cm = pd.DataFrame(cm,index=clase,columns=clase)
    t_cm["Acuratete"] = np.diag(cm)*100/np.sum(cm,axis=1)
    a_globala = np.sum(np.diag(cm))*100/len(y)
    a_medie = t_cm["Acuratete"].mean()
    cohen_kappa = cohen_kappa_score(y,y_,labels=clase)
    return pd.Series([a_globala,a_medie,cohen_kappa],["A.Globala","A.Medie","Index CK"],name="Acuratete"),t_cm

def salvare_matrice(x, nume_linii=None, nume_coloane=None, nume_fisier=None):
    t = pd.DataFrame(x, nume_linii, nume_coloane)
    if nume_fisier is not None:
        t.to_csv(nume_fisier)
    return t