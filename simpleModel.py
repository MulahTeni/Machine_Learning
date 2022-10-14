import math
import numpy as np


def compute_model_output(x, w, b):  # f(x) with updated w and b
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb


def compute_cost(x, y, w, b):   # f(x) - y[i]
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i]) ** 2
    total_cost = 1 / (2 * m) * cost

    return total_cost


def compute_gradient(x, y, w, b):   # Computes derivative of cost
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, num_iters):   # Updates w and b
    b = b_in
    w = w_in
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        b = b - alpha * dj_db
        w = w - alpha * dj_dw
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Ite No. {i}     Final w = {w}   Final b = {b}")

    return w, b


x_train = np.array([3.0, 5.0, 7.0])     # data to train w and b
y_train = np.array([9.0, 14.0, 21.0])   # data to train w and b
w_init = 0                              # first w
b_init = 0                              # first b
ites = 10                               # number of iteration
tmp_alpha = 10.9e-3                     # alpha for gradinet descent
w_final, b_final = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha, ites)
xf = np.array([4])
result_1 = compute_model_output(xf, w_final, b_final)
print(f"x = 4  \nf(x) = {result_1}")
