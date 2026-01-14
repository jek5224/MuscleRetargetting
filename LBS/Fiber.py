import matplotlib.pyplot as plt
import numpy as np

# Data for the x-axis
x = [0, 1, 1, 0]

# Data for the y-axis
y = [0, 0, 1, 1]

# plt.scatter(x, y, color='black')

# b_a = np.array([[0, 0], [0, 0.5], [0, 1]])
# b_b = np.array([[1, 0], [1, 0.5], [1, 1]])

b_a_I = np.array([[1, 0], [0, 0], [0, 0.5]])
b_b_I = np.array([[1, 0.25], [1, 0.4], [1, 0.5]])
b_a_II = np.array([[1, 1], [0, 1], [0, 0.5]])
b_b_II = np.array([[1, 0.75], [1, 0.6], [1, 0.5]])
control_points_set = []
control_points_set.append([b_a_I, b_b_I])
control_points_set.append([b_a_II, b_b_II])

plt.gca().set_aspect('equal', adjustable='box')
plt.plot([x[0], x[1], x[2], x[3], x[0]], [y[0], y[1], y[2], y[3], y[0]], color='black')

numfibs = 30
def b(i, s, b_a, b_b):
    if s >= 0 and s <= numfibs:
        return b_a[i] + (1 - s / numfibs) * (b_b[i] - b_a[i])
    elif s > numfibs:
        return b_a[i]
    else:
        return None

# for s in range(numfibs + 1):
#     b_s = b(s, b_a, b_b)
#     plt.scatter(b_s[:, 0], b_s[:, 1], marker='o', color='green')

def w(i, s):
    if i == 0 or i == 2:
        return 1.0
    elif i == 1:
        return s + 1.0
    
def B(i, t):
    return 2 / (np.math.factorial(2 - i) * np.math.factorial(i)) * (1 - t) ** (2 - i) * t ** i

def fib(t, s, b_a, b_b):
    numerator = [w(i, s) * b(i, s, b_a, b_b) * B(i, t) for i in range(3)]
    denominator = [w(i, s) * B(i, t) for i in range(3)]

    numerator = sum(numerator)
    denominator = sum(denominator)

    return numerator / denominator

num_t = 100
ts = np.linspace(0, 1, num_t)

for b_a, b_b in control_points_set:
    for s in range(numfibs + 1):
        fib_s = []
        for t in ts:
            fib_s.append(fib(t, s, b_a, b_b))
        plt.plot([point[0] for point in fib_s], [point[1] for point in fib_s], color='green')
        # plt.scatter([point[0] for point in fib_s], [point[1] for point in fib_s], color='green')

    plt.scatter([point[0] for point in b_a], [point[1] for point in b_a], marker='o', color='blue')
    plt.scatter([point[0] for point in b_b], [point[1] for point in b_b], marker='o', color='red')

plt.show()