import numpy as npy
import pandas as pnd
import matplotlib.pyplot as plt  

from sklearn.linear_model import LinearRegression

_data = pnd.read_csv('data.csv')

# with sklearn
x1 = _data.iloc[:, 0].values.reshape(-1, 1)
y1 = _data.iloc[:, 1].values.reshape(-1, 1)

linear_regressor = LinearRegression
linear_regressor.fit(x1, y1)

xf1 = 15.0
y_pred1 = linear_regressor.predict(npy.array([[xf1]]))
print(y_pred2)

Y_pred2 = linear_regressor.predict(x1
plt.scatter(x2, y2)
plt.plot(x2, Y_pred2, color='red')
plt.show()


# handmade

# Updates w and b
def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    b = b_in
    w = w_in
    
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        b = b - alpha * dj_db
        w = w - alpha * dj_dw
        
    #print(f"Final w = {w}   Final b = {b}")
    
    return w, b
    
# computes gradient
def compute_gradient(x, y, w, b):   
    m = x.shape[0]
    res_w = 0
    res_b = 0
    
    for i in range(m):
        f_i = w * x[i] + b
        dw_i = (f_i - y[i]) * x[i]
        db_i = f_i - y[i]
        res_b += db_i
        res_w += dw_i
        
    res_w /= m
    res_b /= m

    return res_w, res_b
    
# predict y with updated w and b
def compute_model_output(x, w, b):  
    m = x.shape[0]
    f_wb = np.zeros(m)
    
    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb

ites = 100                               # number of iteration
tmp_alpha = 1.0e-3                       # alpha for gradinet descent
w_final, b_final = gradient_descent(x, y, 0, 0, tmp_alpha, ites)
xf = npy.array([15])
result = compute_model_output(xf, w_final, b_final)
print(f"f(15) = {result}")

plt.scatter(x, y)
y_pred = w_final*x + b_final
plt.plot(x, y_pred, color='red')
plt.xlabel('Experience (year)')
plt.ylabel('Salary ($)')
plt.show()
