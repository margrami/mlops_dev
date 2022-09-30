import numpy as np
from regressions import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=100, 
                                n_features=1, 
                                noise=40, 
                                random_state=4)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size =0.2, 
                                                    random_state=1234)

reg = LinearRegression(learning_rate = 0.01)
reg.fit(X_train, y_train)
prediction = reg.predict(X_test)

# draw the data and the LR all toguether 

y_pred_line = reg.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure()
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color='black', linewidth=1, label='Prediction')
plt.show()



#if __name__=='__main__':
#    main()