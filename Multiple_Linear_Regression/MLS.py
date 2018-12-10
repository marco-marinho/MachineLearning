import pandas as pnd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import statsmodels.formula.api as sm


##########################

data = pnd.read_csv('50_Startups.csv')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

#######################################

ct = ColumnTransformer([
    ('data', 'passthrough', slice(0, 3)),
    ('country', OneHotEncoder(), [3])
])

X = ct.fit_transform(X).astype(np.float)
X = X[:, :-1]

###############################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)

###################################################

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

y_pred_view = sc_y.inverse_transform(y_pred)
y_test_view = sc_y.inverse_transform(y_test)
###################################################

X = np.append(np.ones((np.shape(X)[0], 1), dtype=np.int64), X, 1)
X_opt = X[:, [0,1,3]]
regressor_OLS = sm.OLS(y, X_opt).fit()

regressor_OLS.summary()