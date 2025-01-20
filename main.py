import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from scipy.stats import f
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import seaborn as sns

from grafice import plot_distributie, show, scatter_scoruri

from functions import nan_replace_t, calcul_metrici, salvare_matrice

pd.set_option('display.float_format', '{:.3f}'.format)
culori = sns.color_palette("Spectral", as_cmap=True)

set_date = pd.read_csv("data_in/stars_train.csv")

nan_replace_t(set_date)
predictori = list(["Temperature","L","R","A_M","Spectral_Class"])
tinta = list(set_date)[-1]


# Prelucrare date
print("Prelucrare date:\n")
print("Valori distincte posibile pentru Spectral Class: ")
print(set_date["Spectral_Class"].unique(), "\n")
spectral_class_mapping = {"M": 1, "A": 2, "F": 3, "B": 4, "O": 5, "K": 6, "G": 7}
set_date["Spectral_Class"] = set_date["Spectral_Class"].map(spectral_class_mapping)

#Impartire train/ test
x_train, x_test, y_train, y_test = (
    train_test_split(set_date[predictori], set_date[tinta], test_size=0.7))

# Discriminarea liniara

# Construire model
model_lda = LinearDiscriminantAnalysis()
model_lda.fit(x_train, y_train)

# Analiza pe setul de invatare
clase = model_lda.classes_
probabilitati_clase = model_lda.priors_
print("Clasele si probabilitatile a priori asociate: ")
print(clase, probabilitati_clase, "\n")
n = len(x_train)
m = len(predictori)
q = len(clase)

# Calcul putere de predictie predictori
t = np.cov(x_train.values, rowvar=False)
g = model_lda.means_ - np.mean(x_train.values, axis=0)
b = g.T @ np.diag(probabilitati_clase) @ g
w = t - b
f_p = (np.diag(b) / (q - 1)) / (np.diag(w) / (n - q))
p_values = 1 - f.cdf(f_p, q - 1, n - q)
validare_predictori = p_values < 0.01
print("Semnificatia predictorilor - True = semnificativ, False = nesemnificativ")
print(validare_predictori, "\n")
t_predictori = pd.DataFrame(
    {
        "Putere discriminare": f_p,
        "P_Values": p_values
    }, index=predictori
)
t_predictori.to_csv("data_out/Predictori.csv")


# Analiza grafica a modelului pe setul de testare
# Scoruri discriminante:
z = model_lda.transform(x_test)
nr_discriminatori = min(q - 1, m)

t_z = salvare_matrice(z, x_test.index,
                      ["z" + str(i + 1) for i in range(nr_discriminatori)],
                      "data_out/z.csv")
t_zg = t_z.groupby(by=y_test.values).mean()
for i in range(nr_discriminatori):
    plot_distributie(z, y_test, clase, i,culori)
if q > 2:
    for i in range(1, nr_discriminatori):
        scatter_scoruri(z, y_test, t_zg.values, clase, k2=i, etichete=x_test.index, culori=culori)

#Calculare putere discriminare variabile discriminante
f_values = []
p_values = []

for i in range(nr_discriminatori):
    scoruri = z[:, i]
    s_b = 0
    s_w = 0
    medie_globala = np.mean(scoruri)
    for clasa in clase:
        indices = (y_test == clasa)
        scoruri_clasa = scoruri[indices]
        medie_clasa = np.mean(scoruri_clasa)
        s_b += len(scoruri_clasa) * (medie_clasa - medie_globala) ** 2
        s_w += np.sum((scoruri_clasa - medie_clasa) ** 2)
    f_value = (s_b / (q - 1)) / (s_w / (len(y_test) - q))
    f_values.append(f_value)
    p_value = 1 - f.cdf(f_value, q - 1, len(y_test) - q)
    p_values.append(p_value)

t_discriminare = pd.DataFrame({
    "Variabila discriminanta": [f"z{i+1}" for i in range(nr_discriminatori)],
    "Putere discriminare (F-value)": f_values,
    "P_Value": p_values
})
t_discriminare.to_csv("data_out/Putere_discriminare.csv", index=False)

# Testare - folosim modelul bazat pe LDA pentru ca are atat acuratetea globala mai mare, cat si indexul Cohen-Kappa mai bun
y_ = model_lda.predict(x_test)
t_predictii_test = pd.DataFrame(index=x_test.index)
t_predictii_test[tinta] = y_test
t_predictii_test["LDA"] = y_

a_lda, t_cm_lda = calcul_metrici(y_test, y_, clase)
a_lda.to_csv("data_out/Acuratete_lda.csv")
t_cm_lda.to_csv("data_out/Mat_conf.csv")

# Predictie
set_aplicare = pd.read_csv("data_in/stars_apply.csv")
set_aplicare["Spectral_Class"] = set_aplicare["Spectral_Class"].map(spectral_class_mapping)

predictie_lda = model_lda.predict(set_aplicare[predictori])
set_aplicare["Predictie LDA"] = predictie_lda

# Pentru ca valorile pentru accuracy sunt foarte mari, am presupus ca modelul poate fi overfitted
# Tinta din setul de testare
y_real = set_aplicare[tinta]
# acuratete pentru setul de testare
acc_test = accuracy_score(y_real, predictie_lda)
print(f"Acuratete pe setul de date de test: {acc_test:.2f}")

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

set_aplicare.to_csv("data_out/Predictie.csv")

show()