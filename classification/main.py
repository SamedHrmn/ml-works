import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

# 2.veri onisleme
# 2.1.veri yukleme

"""
veriler = load_iris()
veriler = pd.DataFrame(data=veriler.data,columns=veriler.feature_names)
"""

veriler = pd.read_csv('iris.csv')

x = veriler.iloc[:, 0:3].values  # bağımsız değişkenler
y = veriler.iloc[:, 5:] # bağımlı değişken


# verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0)
logr.fit(X_train, np.ravel(y_train))
y_pred = logr.predict(X_test)


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print("LogisticR. Confision:")
print(cm)


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(X_train, np.ravel(y_train))

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("KNN Confision:")
print(cm)


from sklearn.svm import SVC

svc = SVC(kernel='rbf')
svc.fit(X_train, np.ravel(y_train))

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print('SVC Confision:')
print(cm)


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, np.ravel(y_train))

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print('GNB Confision:')
print(cm)

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion='entropy')

dtc.fit(X_train, np.ravel(y_train))
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print('DTC Confision')
print(cm)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=10, criterion='entropy')
rfc.fit(X_train, np.ravel(y_train))

y_pred = rfc.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred)
print('RFC Confision')
print(cm)

FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
TP = np.diag(confusion_matrix)
TN = confusion_matrix.sum() - (FP + FN + TP)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP)
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)
print("FP"+str(FP))
print("FN"+str(FN))
print("TP"+str(TP))
print("TN"+str(TN))
# 7. ROC , TPR, FPR değerleri

y_proba = rfc.predict_proba(X_test)
from sklearn import metrics

fpr, tpr, thold = metrics.roc_curve(y_test, y_proba[:, 0],pos_label='Species')
print("False P:"+str(fpr))
print("True P:"+str(tpr))








