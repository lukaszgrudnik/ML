import matplotlib.pyplot as plt
import numpy as np
import time

# Generates pseudo-random data:
X = np.random.rand(500) * 2 - 1.0
Y = np.random.rand(500) * 2 - 1.0
D = np.array([1 if Y[i] > 3 * x + 0.3 else 0 for i, x in enumerate(X)])

# Splits into two classes:
red = [X[D.astype(bool)], Y[D.astype(bool)]]
green = [X[np.invert(D.astype(bool))], Y[np.invert(D.astype(bool))]]

# Plot
fig, ax = plt.subplots()
plt.scatter(red[0], red[1], color='r', marker='o')
plt.scatter(green[0], green[1], color='green', marker='o')
dot, = ax.plot(0, 0, color='k', marker='o')
line, = ax.plot(0, 0, color='k')

plt.show(block=False)
# plt.show()

# Perceptron:
class Perceptron:

    def __init__(self, X1, X2, D):
        self.tol = 0.0001
        self.d = D
        self.eta = 0.002
        self.x = np.array([[1, X1[i], X2[i]] for i, _ in enumerate(X1)])
        self.w = np.random.rand(3) * 0.01 - 0.1

    def f(self, x, w):
        return 0 if np.sum(np.multiply(x, w)) < 0 else 1

    def train(self):
        for _ in range(0, 20):
            for t, x in enumerate(self.x):
                y = self.f(x, self.w)
                dot.set_ydata(x[2])
                dot.set_xdata(x[1])
                if y == 0 and self.d[t] == 1:
                    self.w = self.w + self.eta * x
                elif y == 1 and self.d[t] == 0:
                    self.w = self.w - self.eta * x
                a_plt = - self.w[1] / self.w[2]
                b_plt = - self.w[0] / self.w[2]
                x_plt = np.linspace(-1.0, 1.0, num=100)
                y_plt = x_plt * a_plt + b_plt
                mask = np.array([1 if -2 <= y <= 2 else 0 for y in y_plt]).astype(bool)
                line.set_ydata(y_plt[mask])
                line.set_xdata(x_plt[mask])
                plt.pause(0.01)
                err = self.evaluate_test(self.x, self.d)
                if err < self.tol:
                    break

    def evaluate_test(self, xx, d):
        y = [self.f(xx, self.w) for x in xx]
        sol = [y == d[t] for t, y in enumerate(y)]
        counter = 0
        for val in sol:
            if val == False:
                counter += 1
        return counter / len(sol)


perceptron = Perceptron(X, Y, D)
perceptron.train()
