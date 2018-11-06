import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

counts = [
    79300000
    , 112700000
    , 168200000
    , 244085350
    , 346817846
    , 436852506
]

x = []
y = []

for (xi, yi) in enumerate(counts):
    features = []
    for i in range(3):
        features.append(pow(xi, i))
    x.append(features)
    y.append(yi)
print(x)
print(y)
x = np.array(x)
y = np.array(y)
# Create linear regression object
regression = linear_model.LinearRegression()
regression.fit(x, y)

# create the testing set
x_test_range = range(len(x), 3 + len(x))
x_test = []
for xi in x_test_range:
    features = []
    for i in range(3):
        features.append(pow(xi, i))
    x_test.append(features)
print(x_test)

print('Coefficients: \n', regression.coef_)
print('Intercept: \n', regression.intercept_)

print("Mean squared error: %.2f"
      % np.mean((regression.predict(x) - y) ** 2))

print('Variance score: %.2f' % regression.score(x, y))

y_predicted = regression.predict(x_test)

print('Next few numbers in the series are')
for p in y_predicted:
    print(p)

plt.scatter(range(len(counts)), counts, color='black')
plt.scatter(x_test_range, y_predicted, color='red')

plt.show()
