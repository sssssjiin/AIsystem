import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

users = pd.read_csv('C:/Users/a4005/Downloads/archive (1)/telecom_users.csv')
users.head()


users.drop(['Unnamed: 0','customerID'], axis = 1, inplace = True)

users.info()

for i in range(len(users)):
        if users['TotalCharges'][i] == " ":
            users.drop(i, inplace = True)
            
users['TotalCharges'] = users['TotalCharges'].apply(lambda x: float(x))
users[['tenure','MonthlyCharges','TotalCharges']].describe().T
users.columns

#데이터 프레임 생성
X = pd.get_dummies(data = users, columns=['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','Churn'],drop_first=True )

X.head()

y = X['Churn_Yes']

X.drop(['Churn_Yes'], axis = 1, inplace = True)


#Remove the columns with correlation of 1 to address multicollinearity
X.drop(['OnlineSecurity_No internet service','OnlineBackup_No internet service','DeviceProtection_No internet service','TechSupport_No internet service','StreamingTV_No internet service','StreamingMovies_No internet service'], axis = 1, inplace = True)

#Remove the columns with correlation of -1to address multicollinearity
X.drop(['MultipleLines_No phone service'], axis = 1, inplace = True)

#Intiating Scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 101)

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

#회귀분석
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
log_predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

# Printing Confusion Matrix
pd.DataFrame(confusion_matrix(y_test,log_predictions))
#Printing Classification Report
print(classification_report(y_test,log_predictions))

#knn
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)
knn_score = knn.score(X_test, y_test)

# Printing Confusion Matrix
pd.DataFrame(confusion_matrix(y_test,knn_pred))

print(classification_report(y_test,knn_pred))
print("knn 정확도 : ",knn_score)

#Random forest
from sklearn.ensemble import RandomForestClassifier
#n_estimators : 사용할 트리의 수
rfc = RandomForestClassifier(n_estimators=1000, max_depth = 8)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
rfc_score = rfc.score(X_test, y_test)

#Printing Confusion Matrix
pd.DataFrame(confusion_matrix(y_test,rfc_pred))

print(classification_report(y_test,rfc_pred))
print("rfc(RandomForest) 정확도 : ", rfc_score)

#svm
from sklearn.svm import SVC
#model = SVC(c=1, gamma=0.001)
model = SVC(gamma=0.05, C = 1)
model.fit(X_train,y_train)
svm_predictions = model.predict(X_test)
svm_score = model.score(X_test, y_test)

#Printing Confusion Matrix
pd.DataFrame(confusion_matrix(y_test,svm_predictions))

# Printing Classification Report
print(classification_report(y_test,rfc_pred))
print("SVM 정확도 : ", svm_score)








