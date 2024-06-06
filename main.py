import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR


class WageRegression:

    def __init__(self):
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None

    def load_data(self, train_file, test_file):
        # Load training data
        df = pd.read_csv(train_file)
        df = df.fillna(0)
        df['HourlyRate'] = df['MonthlyEarnings'] / (4 * df['AveWeeklyHours'])
        self.X = df.drop(['MonthlyEarnings', 'AveWeeklyHours', 'HourlyRate'], axis=1)
        self.y = df['HourlyRate']

        # Load test data
        df_test = pd.read_csv(test_file)
        self.X_test = df_test.drop(['MonthlyEarnings', 'AveWeeklyHours'], axis=1)
        self.y_test = df_test['MonthlyEarnings'] / (4 * df_test['AveWeeklyHours'])

    def linear_regression(self):
        lin_reg = LinearRegression()
        lin_reg.fit(self.X, self.y)

        # Compute training error
        y_pred = lin_reg.predict(self.X)
        error = ((self.y - y_pred) ** 2).sum()
        print("Training Error (SSE):", error)

        # Print coefficients
        print("Coefficients:")
        for i, coef in enumerate(lin_reg.coef_):
            print(self.X.columns[i], ":", coef)

        # Predict on test data
        y_pred = lin_reg.predict(self.X_test)

        # Compute test error
        error = ((self.y_test - y_pred) ** 2).sum()
        print("Test Error (SSE):", error)

    def svr(self):
        # SVR with polynomial kernel
        svr_poly = SVR(C=100, kernel='poly')
        svr_poly.fit(self.X, self.y)

        print("SVR Poly Training SSE:", ((svr_poly.predict(self.X) - self.y) ** 2).sum())
        print("SVR Poly Testing SSE:", ((svr_poly.predict(self.X_test) - self.y_test) ** 2).sum())

        # SVR with RBF kernel
        svr_rbf = SVR(C=100, kernel='rbf')
        svr_rbf.fit(self.X, self.y)

        print("SVR RBF Training SSE:", ((svr_rbf.predict(self.X) - self.y) ** 2).sum())
        print("SVR RBF Testing SSE:", ((svr_rbf.predict(self.X_test) - self.y_test) ** 2).sum())


# Usage
regressor = WageRegression()
regressor.load_data('wages-train.csv', 'wages-test.csv')
regressor.linear_regression()
regressor.svr()