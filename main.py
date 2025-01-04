import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from scipy.stats import f
from sklearn.naive_bayes import GaussianNB

from grafice import plot_distributie, show, scatter_scoruri

from functions import nan_replace_t, calcul_metrici, salvare_matrice

pd.set_option('display.float_format', '{:.3f}'.format)

set_date = pd.read_csv("data_in/healthcare-dataset-stroke-data.csv", index_col=0)
# set_date = set_date[6000:]

nan_replace_t(set_date)
# predictori = list(set_date)[:-1]
predictori = list(["age", "avg_glucose_level", "hypertension"])
tinta = list(set_date)[-1]

# Prelucrare date

# set_date["Gender"] = (set_date["Gender"] == "Male").astype(int)
#
# low_medium_high_mapping = {"Low": 1, "Medium": 2, "High": 3}
# set_date["Exercise Habits"] = set_date["Exercise Habits"].map(low_medium_high_mapping)
#
# set_date["Smoking"] = (set_date["Smoking"] == "Yes").astype(int)
# set_date["Family Heart Disease"] = (set_date["Family Heart Disease"] == "Yes").astype(int)
# set_date["Diabetes"] = (set_date["Diabetes"] == "Yes").astype(int)
# set_date["High Blood Pressure"] = (set_date["High Blood Pressure"] == "Yes").astype(int)
# set_date["Low HDL Cholesterol"] = (set_date["Low HDL Cholesterol"] == "Yes").astype(int)
# set_date["High LDL Cholesterol"] = (set_date["High LDL Cholesterol"] == "Yes").astype(int)
#
# set_date["Alcohol Consumption"] = set_date["Alcohol Consumption"].map(low_medium_high_mapping)
# set_date["Stress Level"] = set_date["Stress Level"].map(low_medium_high_mapping)
# set_date["Sugar Consumption"] = set_date["Sugar Consumption"].map(low_medium_high_mapping)

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
validare_predictori = p_values < 0.05
print(validare_predictori)
t_predictori = pd.DataFrame(
    {
        "Putere discriminare": f_p,
        "P_Values": p_values
    }, index=predictori
)
t_predictori.to_csv("data_out/Predictori.csv")

# giif not all(validare_predictori):
#     print("Filtrare predictori!")
#     predictori = np.array(predictori)[validare_predictori]

# Analiza grafica a modelului pe setul de testare
z = model_lda.transform(x_test)
nr_discriminatori = min(q - 1, m)

t_z = salvare_matrice(z, x_test.index,
                      ["z" + str(i + 1) for i in range(nr_discriminatori)],
                      "data_out/z.csv")
t_zg = t_z.groupby(by=y_test.values).mean()
for i in range(q - 1):
    plot_distributie(z, y_test, clase, i)
if q > 2:
    for i in range(1, q - 1):
        scatter_scoruri(z, y_test, t_zg.values, clase, k2=i, etichete=x_test.index)

# Testare
y_ = model_lda.predict(x_test)
t_predictii_test = pd.DataFrame(index=x_test.index)
t_predictii_test[tinta] = y_test
t_predictii_test["LDA"] = y_

a_lda, t_cm_lda = calcul_metrici(y_test, y_, clase)
a_lda.to_csv("data_out/Acuratete_lda.csv")
t_cm_lda.to_csv("data_out/Mat_conf.csv")

# Predictie
set_aplicare = pd.read_csv("data_in/healthcare-dataset-stroke-data.csv", index_col=0)

print(set_aplicare[predictori])

predictie_lda = model_lda.predict(set_aplicare[predictori])
set_aplicare["Predictie LDA"] = predictie_lda

# Discriminarea Bayesiana
model_b = GaussianNB()
model_b.fit(x_train, y_train)

# Testare
y_b_ = model_b.predict(x_test)
a_b, t_cm_b = calcul_metrici(y_test, y_b_, clase)
a_b.to_csv("data_out/Acuratete_b.csv")
t_cm_b.to_csv("data_out/Mat_conf_b.csv")
t_predictii_test["Bayes"] = y_b_

t_predictii_test.to_csv("data_out/Predictii_test.csv")
err_lda = t_predictii_test[y_ != y_test]
err_bayes = t_predictii_test[y_b_ != y_test]
err_lda.to_csv("data_out/Err_lda.csv")
err_bayes.to_csv("data_out/Err_bayes.csv")

# Predictie
predictie_b = model_b.predict(set_aplicare[predictori])
set_aplicare["Predictie Bayes"] = predictie_b

set_aplicare.to_csv("data_out/Predictie.csv")

show()
