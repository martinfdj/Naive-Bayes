from sklearn import datasets


dataset = datasets.load_breast_cancer()
print(dataset)


print('Información en el dataset:')
print(dataset.keys())
print()


print('Características del dataset:')
print(dataset.DESCR)


X = dataset.data

y = dataset.target



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


from sklearn.naive_bayes import GaussianNB
algoritmo = GaussianNB()


algoritmo.fit(X_train, y_train)
y_pred = algoritmo.predict(X_test)


from sklearn.metrics import confusion_matrix
matriz = confusion_matrix(y_test, y_pred)
print('Matriz de Confusión:')
print(matriz)


from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print('Precisión del modelo:')
print(precision)
