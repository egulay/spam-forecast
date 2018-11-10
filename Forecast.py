import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import datetime

from OutputFormatter import output_formatter


def main():
    black_friday = [
        58000000
        , 86300000
        , 133900000
        , 201437526
        , 311247035
        , 405677657
    ]
    cyber_monday = [
        79300000
        , 112700000
        , 168200000
        , 244085350
        , 346817846
        , 436852506
    ]
    print(output_formatter.BOLD + "Execution: Black Friday Forecast" + output_formatter.END)
    execute_forecast(black_friday)
    print(output_formatter.BOLD + "********************************" + output_formatter.END)

    print(output_formatter.BOLD + "Execution: Cyber Monday Forecast" + output_formatter.END)
    execute_forecast(cyber_monday)
    print(output_formatter.BOLD + "********************************" + output_formatter.END)


def execute_forecast(inputs=None):
    if inputs is None:
        inputs = []
    x = []
    y = []

    for (xi, yi) in enumerate(inputs):
        features = []
        for i in range(3):
            features.append(pow(xi, i))
        x.append(features)
        y.append(yi)
    print(x)
    print(y)
    x = np.array(x)
    y = np.array(y)
    # linear regression
    regression = linear_model.LinearRegression()
    regression.fit(x, y)

    # testing
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

    print(output_formatter.BOLD + "Forecast for next " + str(
        np.count_nonzero(y_predicted)) + output_formatter.END)
    for p in y_predicted:
        print(output_formatter.PURPLE + "{:,}".format(p) + output_formatter.END)

    plt.scatter(range(len(inputs)), inputs, color="black")
    plt.scatter(x_test_range, y_predicted, color="purple")

    plt.show()


if __name__ == "__main__":
    main()
