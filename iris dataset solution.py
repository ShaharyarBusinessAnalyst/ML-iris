import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


url = r"C:\Users\Shaharyar\Downloads\data\iris_dataset.csv"
names = ['sepal-length','sepal-width','petal-length', 'petal-width','class']
dataset = pd.read_csv(url, names = names)

print(dataset.shape)

print(dataset.head(10))
dataset.describe()
print(dataset.dtypes)

#converting data type from objects to numeric
dataset['petal-length'] = pd.to_numeric(dataset['petal-length'], errors = 'coerce') 
dataset['petal-width'] = pd.to_numeric(dataset['petal-width'], errors = 'coerce')
dataset['sepal-length'] = pd.to_numeric(dataset['sepal-length'], errors = 'coerce') 
dataset['sepal-width'] = pd.to_numeric(dataset['sepal-width'], errors = 'coerce')
dataset.drop(0)

#box plot to get to know range
dataset.plot(kind='box', subplots = True, layout = (2,2), sharex = False, sharey = False)
plt.show()

#histogtam to find distribution type
dataset.hist()
plt.show()

array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.2
seed = 6
X_train, X_test, Y_train, Y_test = model_selectiontrain_test_split(X, Y, test_size = validation_size, random_state = seed)

seed  = 6
scoring = 'accuracy'

models = []
models.append(('LR',LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC()))

results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits = 10, random_state=seed, shuffle= True)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv = kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)