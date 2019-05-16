import pandas as pd
from sklearn import linear_model 
import joblib

df = pd.read_csv('train.csv')
selected_features = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch']

train = df['Survived']
v = df[selected_features].drop('Survived', axis = 1)

v['Age'].fillna(v['Age'].mean(), inplace = True)
v['Age'] = v['Age'].apply(round)
v['Age'] = v['Age'].apply(int)

v['TravelAlone'] = v['SibSp']+v['Parch']
for i in v.index: 
    if v.loc[i, 'TravelAlone'] == 0:
        v.loc[i, 'TravelAlone'] = 1
    else:
        v.loc[i, 'TravelAlone'] = 0

del v['SibSp']
del v['Parch']

v['Pclass'] = v['Pclass'].astype('object')

v = pd.get_dummies(v)

model= linear_model.LogisticRegression(C=1, penalty='l2', solver='liblinear')
model.fit(v, train)

filename = 'model.sav'
joblib.dump(model, filename)