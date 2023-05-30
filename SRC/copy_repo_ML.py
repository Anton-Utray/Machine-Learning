from sklearn.model_selection import train_test_split as tts 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


## Separar X and y en versiones test y version de entranmiento. Esta toma una propo de 80%entrenamiento y 20% de testeo. devuelve shape para comparar.

        # X_train, X_test, y_train, y_test = tts(X, y, 
        #                                        train_size=0.8, 
        #                                        test_size=0.2,
        #                                        random_state=42
        #                                       )


        # X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.model_selection import train_test_split as tts
def perform_train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = tts(X, y, train_size=train_size, test_size=test_size, random_state=random_state)
    
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

## Entrenar modelo. Cambiar modelo al gusto, aqui usa regresion logistica para determinar categorias. Lineal es para predicir numeros

                # model=LogisticRegression()  # inicia el modelo

                # model.fit(X_train, y_train) # entrena el modelo

                # y_pred = model.predict(X_test) # prediccion

from sklearn.linear_model import LogisticRegression

def train_and_predict_logistic_regression(X_train, y_train, X_test):
    model = LogisticRegression()  # Initialize the logistic regression model
    model.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(X_test)  # Predict the target variable for the test set
    return y_pred


## Matriz de confusión, aplica unicamente a modelos discretos de clasificación, no para lineales

                # y_pred = model.predict(X_test)

                # cm = confusion_matrix(y_test, y_pred)
                # print("Confusion Matrix:")
                # print(cm)

                # accuracy = accuracy_score(y_test, y_pred)
                # print("Accuracy Score:", accuracy)

from sklearn.metrics import confusion_matrix, accuracy_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)  # Predict the target variable for the test set
    
    cm = confusion_matrix(y_test, y_pred)  # Calculate the confusion matrix
    print("Confusion Matrix:")
    print(cm)
    
    accuracy = accuracy_score(y_test, y_pred)  # Calculate the accuracy score
    print("Accuracy Score:", accuracy)


#matriz de confusion en heatmap

                # plt.figure(figsize=(15, 8))

                # ax=sns.heatmap(cm(y_test, y_pred)/cm(y_test, y_pred).sum(), annot=True)

                # plt.title('Matriz confusion')
                # plt.ylabel('Verdad')
                # plt.xlabel('Prediccion')
                # plt.show();

import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)  # Calculate the confusion matrix

    plt.figure(figsize=(15, 8))  # Set the size of the figure

    # Plot the heatmap of the normalized confusion matrix
    ax = sns.heatmap(cm/cm.sum(), annot=True)

    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()

# EVALUACION DE ERRORES

## R2

from sklearn.metrics import r2_score as r2

            #r2(y_test, y_pred)

## Mean squared error

from sklearn.metrics import mean_squared_error as mse

            #mse(y_test, y_pred)

# Mean absolute error

from sklearn.metrics import mean_absolute_error as mae 

            #mae(y_test, y_pred)

# balanced accuracy 

from sklearn.metrics import balanced_accuracy_score as bas
            # balanced_acc = bas(y_test, y_pred)
            # print("Balanced Accuracy:", balanced_acc)

#precisiom score, si es multiclase aggreagar 'average=...' y indicar por cual de las clases empezar, nombre en string

from sklearn.metrics import precision_score as pes

            # precision = pes(y_test, y_pred)
            # print("Precision Score:", precision)

#recall score

from sklearn.metrics import recall_score as rec
            # recall=rec(y_train, y_pred)
            # print("Recall Score:", recall)

#F1 score

from sklearn.metrics import f1_score as f1
            # f1=rec(y_test, y_pred)
            # print("F1 Score:", f1)

#escalarizar los datos (ponerlos todo en un mismo rango para el modelo)

from sklearn.preprocessing import StandardScaler
            # scaler = StandardScaler() #se inicializa
            # scaler.fit(x)
            # customer_scaled = scaler.transform(x)

def scale_data(x):
    scaler = StandardScaler()  # Initialize the StandardScaler
    scaler.fit(x)  # Fit the scaler to the data
    x_scaled = scaler.transform(x)  # Transform the data using the scaler
    return x_scaled
#Evaluación numero de clusters para KMEANS

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def cluster_evaluation(data, max_clusters):
    wcss = []
    silhouette_scores = []

    for k in range(1, max_clusters + 1):
        if k == 1:
            labels = np.zeros(len(data))
            wcss.append(0)
            silhouette_scores.append(0)
        else:
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(data)
            wcss.append(kmeans.inertia_)
            silhouette_avg = silhouette_score(data, kmeans.labels_)
            silhouette_scores.append(silhouette_avg)

    # Elbow Method
    plt.plot(range(1, max_clusters + 1), wcss)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')
    plt.title('Elbow Method')
    plt.show()

    # Silhouette Score
    plt.plot(range(1, max_clusters + 1), silhouette_scores)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score Method')
    plt.show()




