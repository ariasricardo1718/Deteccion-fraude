# Importing the libraries
# https://www.kaggle.com/ (Insurance Fraud Claims)

import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import pingouin as pg
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)


print("\n================ INICIO =====================\n")
df = pd.read_csv("fraud_oracle.csv")

print("\n **Se muestran los primeros cinco registros cargados:**")
print("---------------------------------")
resultado = df.head(5)
print(resultado)

print("\n **Estructura de la tabla de datos:**")
print("---------------------------------")
df.info()

print("\n **Columnas y sus tipos de datos:**")
print("---------------------------------")
for column in df:
    print(column,":\n",df[column].unique(),'\n')
    
# Eliminar la columna con el número de póliza, que no aporta información útil
df = df.drop(columns="PolicyNumber")

# Reemplazar en los campos 'Age', 'DayOfWeekClaimed', 'MonthClaimed'
# los valores de cero (0) por NaN (Not a Number), 
# que en Pandas representa un valor faltante o desconocido.
df[['Age', 'DayOfWeekClaimed', 'MonthClaimed']] = (
    df[['Age', 'DayOfWeekClaimed', 'MonthClaimed']]
    .replace(0, np.nan)
)

# Reemplazar los campos que contienen "none" por valor cero (0)
df[['Days_Policy_Accident','Days_Policy_Claim',
    'PastNumberOfClaims','NumberOfSuppliments']
   ] = df[['Days_Policy_Accident','Days_Policy_Claim',
           'PastNumberOfClaims','NumberOfSuppliments']
          ].replace('none', '0')
   
# Reemplazar los campos que contienen "new", por valor cero (0)
df['AgeOfVehicle'] = df['AgeOfVehicle'].replace('new', '0')
df['AddressChange_Claim'] = df['AddressChange_Claim'].replace('new', '0')

# Algunas características que son cadenas pero tienen significados numéricos. 
# Con "regex" se convierten estas características a tipo numérico.
pattern = r'(\d+)'
columns = ['NumberOfCars','NumberOfSuppliments','AgeOfPolicyHolder',
           'AgeOfVehicle','PastNumberOfClaims', 'Days_Policy_Claim', 
           'Days_Policy_Accident', 'VehiclePrice']
for column in columns:
    # Convert the column to string type
    df[column] = df[column].astype(str)
    
    # Use regex to extract the first number from each value
    df[column] = df[column].str.extract(pattern, expand=False)
    
df[['NumberOfCars','NumberOfSuppliments','AgeOfPolicyHolder',
    'AgeOfVehicle','PastNumberOfClaims', 'Days_Policy_Claim', 
    'Days_Policy_Accident', 'VehiclePrice']
   ] = df[['NumberOfCars','NumberOfSuppliments','AgeOfPolicyHolder',
           'AgeOfVehicle','PastNumberOfClaims', 'Days_Policy_Claim', 
           'Days_Policy_Accident', 'VehiclePrice']].astype('int')
   
## Probando la hipótesis
# Función plot_countplot_with_labels(data, x, hue):
# crear una gráfica de barras de frecuencias (countplot) usando Seaborn, 
# mostrar la cantidad de registros en cada barra y luego visualizar la gráfica
def plot_countplot_with_labels(data, x, hue):
    # Create the countplot
    ax = sns.countplot(data=df, x=x, hue=hue)

    # Add labels to the top of each bar
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(height, (p.get_x() + p.get_width() / 2, height), ha="center", va="bottom")

    # Display the plot
    plt.show()
    
df = df.dropna()

# Hipótesis nula (H0): 
#     El conjunto de datos no está sesgado hacia muestras no fraudulentas. 
#     La proporción de muestras fraudulentas es igual a la proporción de 
#     muestras no fraudulentas.
# ---
# Hipótesis alternativa (HA): 
#     El conjunto de datos está sesgado hacia muestras no fraudulentas. 
#     La proporción de muestras fraudulentas es diferente de la proporción de 
#     muestras no fraudulentas.

# Calculate observed frequencies
observed_freq = df['FraudFound_P'].value_counts()

# Calculate expected frequencies assuming no skewness (equal proportions)
expected_freq = pd.Series([len(df) / 2, len(df) / 2], index=[0, 1])

# Perform chi-square goodness-of-fit test
chi2, p_value = stats.chisquare(observed_freq, expected_freq)

print("Chi-square statistic:", chi2)
print("p-value:", p_value)

# sns.histplot(data=df, x="FraudFound_P")
# plt.show()

plt.figure(figsize=(8,5))

ax = sns.countplot(
    data=df,
    x="FraudFound_P",
    hue="FraudFound_P",
    palette=["#4C72B0", "#DD8452"],
    legend=False
)

for p in ax.patches:
    ax.annotate(
        f"{int(p.get_height())}",
        (p.get_x() + p.get_width()/2, p.get_height()),
        ha="center",
        va="bottom"
    )

plt.title("Cantidad de reclamaciones fraudulentas y no fraudulentas")
plt.xlabel("Fraude (0=No, 1=Sí)")
plt.ylabel("Número de reclamaciones")

plt.show()

sns.boxplot( y=df["Age"], x=df["FraudFound_P"] )
plt.xlabel("Fraude (0=No, 1=Sí)")
plt.ylabel("Edad")
plt.show()
    
print("\n================  FIN  =====================\n")

# ==============================================================
# MACHINE LEARNING - DETECCIÓN DE RECLAMACIONES FRAUDULENTAS
# ==============================================================

# --------------------------------------------------------------
# 1. Separar variables predictoras (X) y variable objetivo (y)
# --------------------------------------------------------------

X = df.drop(columns="FraudFound_P")
y = df["FraudFound_P"]


# --------------------------------------------------------------
# 2. Identificar variables categóricas y numéricas
# --------------------------------------------------------------

categorical_cols = X.select_dtypes(include=["object"]).columns
numeric_cols = X.select_dtypes(exclude=["object"]).columns

print("\nVariables categóricas:")
print(list(categorical_cols))

print("\nVariables numéricas:")
print(list(numeric_cols))


# --------------------------------------------------------------
# 3. Preparar el procesamiento de los datos
# --------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_cols
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_cols
        )
    ]
)


# --------------------------------------------------------------
# 4. Separar datos para entrenamiento y prueba
# --------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n================ DATOS DE ENTRENAMIENTO ================")
print("Registros para entrenamiento:", len(X_train))
print("Registros para prueba:", len(X_test))


# --------------------------------------------------------------
# 5. Definir los modelos
# --------------------------------------------------------------

modelos = {

    "Regresión Logística":
        LogisticRegression(
            max_iter=1000
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            random_state=42
        )
}


# --------------------------------------------------------------
# 6. Entrenar y evaluar los modelos
# --------------------------------------------------------------

resultados = []

modelos_entrenados = {}

for nombre, clasificador in modelos.items():

    print("\n======================================================")
    print(nombre)
    print("======================================================")

    modelo = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clasificador)
    ])

    # Entrenamiento
    modelo.fit(X_train, y_train)

    # Predicción
    y_pred = modelo.predict(X_test)

    # Guardar modelo
    modelos_entrenados[nombre] = modelo

    # Métricas
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    resultados.append({
        "Modelo": nombre,
        "Accuracy": accuracy,
        "Precision (Fraude)": precision,
        "Recall (Fraude)": recall,
        "F1 (Fraude)": f1
    })

    print("\nAccuracy:", round(accuracy, 4))
    print("Precision (Fraude):", round(precision, 4))
    print("Recall (Fraude):", round(recall, 4))
    print("F1 (Fraude):", round(f1, 4))

    print("\nReporte de clasificación:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "No fraude",
                "Fraude"
            ],
            zero_division=0
        )
    )


# --------------------------------------------------------------
# 7. Tabla comparativa de resultados
# --------------------------------------------------------------

df_resultados = pd.DataFrame(resultados)

print("\n\n======================================================")
print("COMPARACIÓN DE MODELOS")
print("======================================================")

print(
    df_resultados.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# --------------------------------------------------------------
# 8. Matrices de confusión
# --------------------------------------------------------------

for nombre, modelo in modelos_entrenados.items():

    y_pred = modelo.predict(X_test)

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No fraude", "Fraude"],
        yticklabels=["No fraude", "Fraude"]
    )

    plt.title(
        f"Matriz de confusión - {nombre}"
    )

    plt.xlabel("Predicción")
    plt.ylabel("Valor real")

    plt.show()


# --------------------------------------------------------------
# 9. Identificar el modelo con mayor Recall de fraude
# --------------------------------------------------------------

mejor_recall = df_resultados.loc[
    df_resultados["Recall (Fraude)"].idxmax()
]

print("\n\n======================================================")
print("RESULTADO SEGÚN RECALL DE FRAUDE")
print("======================================================")

print(
    "Modelo:",
    mejor_recall["Modelo"]
)

print(
    "Recall (Fraude):",
    round(
        mejor_recall["Recall (Fraude)"],
        4
    )
)


# --------------------------------------------------------------
# 10. Identificar el modelo con mayor F1 de fraude
# --------------------------------------------------------------

mejor_f1 = df_resultados.loc[
    df_resultados["F1 (Fraude)"].idxmax()
]

print("\n\n======================================================")
print("RESULTADO SEGÚN F1 DE FRAUDE")
print("======================================================")

print(
    "Modelo:",
    mejor_f1["Modelo"]
)

print(
    "F1 (Fraude):",
    round(
        mejor_f1["F1 (Fraude)"],
        4
    )
)

# ================================
# 6. Predicciones y ajuste de umbral
# ================================
print("\n/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/-")
y_pred_proba = modelo.predict_proba(X_test)[:, 1]

# Ajuste de umbral (ejemplo: 0.3 en lugar de 0.5)
umbral = 0.3
y_pred_adj = (y_pred_proba > umbral).astype(int)

# ================================
# 7. Evaluación
# ================================
print("\nReporte de clasificación con umbral ajustado:")
print(classification_report(y_test, y_pred_adj, target_names=["No fraude", "Fraude"]))

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred_adj))

print("\nROC-AUC:", roc_auc_score(y_test, y_pred_proba))

# ================================
# 2. Calcular curva Precision-Recall
# ================================
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

# ================================
# 3. Graficar curva
# ================================
plt.figure(figsize=(8,6))
plt.plot(recall, precision, marker='.', label=f'RandomForest (AP={avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall')
plt.legend()
plt.grid(True)
plt.show()

# ================================
# 4. Opcional: elegir umbral óptimo
# ================================
# Ejemplo: seleccionar el umbral donde recall >= 0.3
umbral_optimo = thresholds[(recall[:-1] >= 0.3)][0]
print("Umbral óptimo sugerido:", umbral_optimo)