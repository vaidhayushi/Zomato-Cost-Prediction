
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv(r'C:\Users\Hp\Desktop\zom\Zomato_df.csv')

df.drop('Unnamed: 0',axis=1,inplace=True)
print(df.head())
x=df.drop('average_cost_for_two',axis=1)
y=df['average_cost_for_two']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=353)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=105)
DTree=DecisionTreeRegressor(min_samples_leaf=.0001)
DTree.fit(x_train,y_train)
y_predict=DTree.predict(x_test)



import pickle
# Save the trained model using pickle
with open('model.pkl', 'wb') as model_file:
    pickle.dump(DTree, model_file)
model=pickle.load(open('model.pkl','rb'))
print(y_predict)
