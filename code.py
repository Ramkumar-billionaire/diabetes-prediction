import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns',None)
warnings.filterwarnings('ignore')

df = pd.read_csv('diabetes.csv')
df
df.duplicated().sum()

df.isnull().sum()
df.info()
df['BMI'] = df['BMI'].astype(int)
df.head()

df.columns = df.columns.str.lower()
df.head()

# Visualisation
# FEATURE VS TARGET

# FOR ALL COLUMNS

sns.set_theme(style='whitegrid',palette='pastel')

plt.figure(figsize=(20,8))
plt.subplot(2,3,1)
sns.barplot(x= df['pregnancies'],data = df , y = df['outcome'])
plt.show()

plt.figure(figsize=(20,8))
plt.subplot(2,3,2)
sns.lineplot(x= df['glucose'],data = df , y = df['outcome'])
plt.show()

plt.figure(figsize=(20,8))
plt.subplot(2,3,3)
sns.lineplot(x= df['bloodpressure'],data = df , y = df['outcome'])
plt.show()

plt.figure(figsize=(20,8))
plt.subplot(2,3,4)
sns.lineplot(x= df['skinthickness'],data = df , y = df['outcome'])
plt.show()

plt.figure(figsize=(20,8))
plt.subplot(2,3,5)
sns.lineplot(x= df['insulin'],data = df , y = df['outcome'])
plt.show()

plt.figure(figsize=(20,8))
plt.subplot(2,3,6)
sns.lineplot(x= df['bmi'],data = df , y = df['outcome'])
plt.show()

plt.figure(figsize=(20,8))
plt.subplot(2,4,1)
sns.lineplot(x= df['diabetespedigreefunction'],data = df , y = df['outcome'])
plt.show()

plt.figure(figsize=(20,8))
plt.subplot(2,4,2)
sns.lineplot(x= df['age'],data = df , y = df['outcome'])
plt.show()



# Remove columns which didnt correlated
df.drop(columns=['insulin','diabetespedigreefunction'],inplace=True,axis=1)
df.head()


# Split Data into Train and Test
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
X = df.drop(columns='outcome',axis=1)
y = df['outcome']
X.head()


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=50)

X_train


# machine Learning
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
model.score(X_test,y_test)


0.6948051948051948


from sklearn.model_selection import cross_val_score #import
cross_val_linear_model = cross_val_score(model,X_train,y_train,cv=10).mean()
cross_val_linear_model


0.7802221047065044

## KNN Classifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

knn_values=np.arange(1,40)
cross_val_knn=[]
for k in knn_values:
    knn_classifier=KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train_scaled,y_train)
    print("K value : ", k, " train score : ", knn_classifier.score(X_train_scaled,y_train)  ,"cross_val_score : ", cross_val_score(knn_classifier,X_train_scaled,y_train,cv = 10).mean())
    cross_val_knn.append(cross_val_score(knn_classifier,X_train_scaled,y_train,cv = 10).mean())

cross_val_knn_classifier=max(cross_val_knn)

print("The best K-Value is 16 and Cross_val_score is",cross_val_knn_classifier )


#Implementation
knn_classifier=KNeighborsClassifier(n_neighbors=16)
knn_classifier.fit(X_train_scaled,y_train)


cross_val_knn_classifier=cross_val_score(knn_classifier,X_train_scaled,y_train,cv=15).mean()
cross_val_knn_classifier

# Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
​
max_depth=np.arange(1,20)
cross_val_dt=[]
for d in max_depth:
    dt_classifier= DecisionTreeClassifier(max_depth=d, random_state=0)
    dt_classifier.fit(X_train,y_train)
    print("Depth : ", d, " train Score  : ", dt_classifier.score(X_train,y_train), "cross_val_score : ", cross_val_score(dt_classifier,X_train,y_train,cv = 10).mean())
    cross_val_dt.append(cross_val_score(dt_classifier,X_train,y_train,cv = 10).mean())

cross_val_dt_classifier=max(cross_val_dt)
print("The best depth is 4 and Cross_val_score is:",cross_val_dt_classifier)

# Implementation
dt_classifier=DecisionTreeClassifier(max_depth=4, random_state=0)
dt_classifier.fit(X_train,y_train)

# DecisionTreeClassifier
DecisionTreeClassifier(max_depth=4, random_state=0)
cross_val_dt_classifier=cross_val_score(dt_classifier,X_train,y_train,cv=10).mean()
cross_val_dt_classifier

0.7638551031200423

ftImp = list(zip(dt_classifier.feature_importances_, df.columns[:-1]))
imp = pd.DataFrame(ftImp, columns = ["Importance","Feature"])
imp.sort_values("Importance",ascending = False,inplace=True)
imp

# Random Forest
from sklearn.ensemble import RandomForestClassifier
​
max_depth=np.array([2,4,8,10,11,12,13,15,18,20])
cross_val_rf=[]
for d in max_depth:
    rf_classifier=RandomForestClassifier(max_depth=d, random_state=0)
    rf_classifier.fit(X_train,y_train)
    print("Depth : ", d, "cross_val_score : ", cross_val_score(rf_classifier,X_train,y_train,cv = 15).mean())
    cross_val_rf.append(cross_val_score(rf_classifier,X_train,y_train,cv = 15).mean())

cross_val_rf_classifier=max(cross_val_rf)
print("The best depth is 20 and Cross_val_score is:",cross_val_rf_classifier)


# Implementation
rf_classifier=RandomForestClassifier(max_depth=20, random_state=0)
rf_classifier.fit(X_train,y_train)

cross_val_rf_classifier=cross_val_score(rf_classifier,X_train,y_train,cv=15).mean()
cross_val_rf_classifier


# XG Boosting Classification
import xgboost as xgb
​
cross_val_xgb=[]
for lr in [0.01,0.05,0.08,0.1,0.2,0.25,0.3]:
    xgb_classifier = xgb.XGBClassifier(learning_rate = lr,n_estimators=100)
    xgb_classifier.fit(X_train,y_train)
    print("Learning rate : ", lr,"cross_val_score:", cross_val_score(xgb_classifier,X_train,y_train,cv = 15).mean())
    cross_val_xgb.append(cross_val_score(xgb_classifier,X_train,y_train,cv = 15).mean())

cross_val_xgb_classifier=max(cross_val_xgb)
print("The best Learning rate is 0.01 and Cross_val_score is:",cross_val_xgb_classifier)




# Implementation
xgb_classifier= xgb.XGBClassifier(learning_rate =0.01,n_estimators=100) # initialise the model
xgb_classifier.fit(X_train,y_train) #train the model

cross_val_xgb_classifier=cross_val_score(xgb_classifier,X_train,y_train,cv=15).mean()
cross_val_xgb_classifier

# CV Score for all Models
print("Cross Validation Score for Logistic Regression Model:",cross_val_linear_model)
print("Cross Validation Score for K-Nearest Neighbors Classification Model:",cross_val_knn_classifier)
print("Cross Validation Score for Decision Tree Classification Model: ",cross_val_dt_classifier)
print("Cross Validation Score for Random Forest Classification Model: ",cross_val_rf_classifier)
print("Cross Validation Score for Extreme-Gradient Boosting Classification Model: ",cross_val_xgb_classifier)
# Cross Validation Score for Logistic Regression Model: 0.7802221047065044
# Cross Validation Score for K-Nearest Neighbors Classification Model: 0.7735772357723577
# Cross Validation Score for Decision Tree Classification Model:  0.7638551031200423
# Cross Validation Score for Random Forest Classification Model:  0.7719512195121951
# Cross Validation Score for Extreme-Gradient Boosting Classification Model:  0.776829268292683

xgb_classifier.feature_importances_

# Important Features
sorted_idx = xgb_classifier.feature_importances_.argsort()
plt.figure(figsize=(10,8))
plt.barh(df.columns[sorted_idx], xgb_classifier.feature_importances_[sorted_idx])
plt.show()






