import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.cm as cm

#cargar el dataset
digits = datasets.load_digits()

"""
target trae la etiqueta del numero y en images trae el arreglo que contiene la imagen del numero de 8x8 pixeles
"""
target = digits.target
images = digits.images

# class names =digits.target_names:
class names:

  #elegir un numero aleatorio dentro del numero de muestras
  sample = np.random.randint(target.shape [0])
  #lo mostramos (nomas para que vean como se ve)
  plt.imshow(images[sample], cmap=plt. get_cmap("Greys"))
  plt.title("Target: %i" % target[sample])

  """
  Aplanamos los ejemplos para cambiar de tener 1797 ejemplos de 8x8 a 1797 ejemplos de 64 elementos
  """

  x = images.reshape((target.shape[0], -1))
  #dividir el data set en entrenamiento y test
  xtrain, xtest, ytrain, ytest = train_test_split(x, target)
  #creamos el modelo de un Perceptron multicapa
  model= MLPClassifier (alpha = 1, max_iter=1000)
  #a entrenar
  model.fit(xtrain , ytrain)
  # Aplicar metrica al modelo
  print("Train: ", model.score(xtrain , ytrain ))
  print("Test: ", model.score(xtest , ytest ))
  #sacar la prediccion en la parte del test
  ytestpred = model.predict(xtest)
  #sacar el reporte de clasificacion

  print("Classification report: \n", metrics.classification_report (ytest , ytestpred ))
  # disp = metrics.plot.confusion.matrix(model, xtest, ytest, display.labels = class names, cmap = plt.cm.Blues ,)
  disp = metrics.plot.confusion.matrix(model, xtest, ytest, cmap = plt.cm.Blues ,)
  disp.ax.set.title("Confusion matrix , without normalization")
plt.show ()
