from dataclasses import replace
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

name = open('2/iris_x.txt','r')
x = []
y = []
for i in name.readlines():
    string = i.split("\t")
    if("\n" in string):
        string.remove("\n")
    for j in range(0,len(string)):
        string[j] = float(string[j])
    x.append(string)
name.close

name = open('2/iris_y.txt','r')
for i in name.readlines():
    y.append(i[0])
name.close

X_train, X_test, y_train, y_test = train_test_split(x,y,random_state = 20220413)

reg =linear_model.LinearRegression()
reg.fit(X_train,y_train)
y_predict = reg.predict(X_test)
score = reg.score(X_test,y_test)
mse = mean_squared_error(y_test, reg.predict(X_test))
print("Mse =",mse)