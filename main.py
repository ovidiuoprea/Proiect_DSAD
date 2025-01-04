import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from scipy.stats import f

from functions import nan_replace_t

set_date = pd.read_csv("data_in/heart_disease.csv")

nan_replace_t(set_date)
predictori = list(set_date)[:-1]
tinta = list(set_date)[-1]

# Prelucrare date

set_date["Gender"] = (set_date["Gender"] == "Male").astype(int)

low_medium_high_mapping = {"Low": 1, "Medium": 2, "High": 3}
set_date["Exercise Habits"] = set_date["Exercise Habits"].map(low_medium_high_mapping)

set_date["Smoking"] = (set_date["Smoking"] == "Yes").astype(int)
set_date["Family Heart Disease"] = (set_date["Family Heart Disease"] == "Yes").astype(int)
set_date["Diabetes"] = (set_date["Diabetes"] == "Yes").astype(int)
set_date["High Blood Pressure"] = (set_date["High Blood Pressure"] == "Yes").astype(int)
set_date["Low HDL Cholesterol"] = (set_date["Low HDL Cholesterol"] == "Yes").astype(int)
set_date["High LDL Cholesterol"] = (set_date["High LDL Cholesterol"] == "Yes").astype(int)


set_date["Alcohol Consumption"] = set_date["Alcohol Consumption"].map(low_medium_high_mapping)
set_date["Stress Level"] = set_date["Stress Level"].map(low_medium_high_mapping)
set_date["Sugar Consumption"] = set_date["Sugar Consumption"].map(low_medium_high_mapping)

#Impartire train/ test

x_train, x_test, y_train, y_test = (
    train_test_split(set_date[predictori], set_date[tinta], test_size=0.3))

# Discriminarea liniara

# Construire model

model_lda = LinearDiscriminantAnalysis()
model_lda.fit(x_train, y_train)

# Analiza pe setul de invatare
clase = model_lda.classes_
f_ = model_lda.priors_
print(clase, f_)
n = len(x_train)
m = len(predictori)
q = len(clase)

# Calcul putere de predictie predictori
t = np.cov(x_train.values, rowvar=False)
g = model_lda.means_ - np.mean(x_train.values, axis=0)
b = g.T @ np.diag(f_) @ g
w = t - b
f_p = (np.diag(b) / (q - 1)) / (np.diag(w) / (n - q))
p_values = 1 - f.cdf(f_p, q - 1, n - q)
validare_predictori = p_values < 0.01
print(validare_predictori)
t_predictori = pd.DataFrame(
    {
        "Putere discriminare": f_p,
        "P_Values": p_values
    }, index=predictori
)
t_predictori.to_csv("data_out/Predictori.csv")
