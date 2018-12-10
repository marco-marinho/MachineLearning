import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

sc_y = StandardScaler()

y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
y_train_pred = lr.predict(X_train)

plt.scatter(sc_X.inverse_transform(X_train), sc_y.inverse_transform(y_train), color='red')
plt.plot(sc_X.inverse_transform(X_train), sc_y.inverse_transform(y_train_pred), color='blue')
plt.title("Salary vs Experience")
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(sc_X.inverse_transform(X_test), sc_y.inverse_transform(y_test), color='red')
plt.plot(sc_X.inverse_transform(X_train), sc_y.inverse_transform(y_train_pred), color='blue')
plt.title("Salary vs Experience")
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

