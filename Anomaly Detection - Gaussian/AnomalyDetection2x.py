# imports
import numpy as npy
import pandas as pnd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# training data
_data = pnd.read_csv('data.csv')

# make model
model = GaussianMixture(n_components=1, covariance_type='full')

model.fit(_data[['Point', 'Time']])

_data['Anomaly Score'] = -model.score_samples(_data[['Point', 'Time']])

# print non anomaly data
print(_data)

# cross validatiton data
crossval_data = pnd.read_csv("crossValidation.csv")

# calculate anomaly score
crossval_data['Anomaly Score'] = -model2.score_samples(crossval_data[['Point', 'Time']])

# predict
crossval_data['Anomaly'] = npy.where(crossval_data['Anomaly Score'] > 20, 1, 0)

# print cross val anomaly status
print(crossval_data)



# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(crossval_data['Time'], crossval_data['Point'], c=crossval_data['Anomaly'], cmap='coolwarm')
plt.xlabel('Time (min)')
plt.ylabel('Point')
plt.title('Anomaly Detection')
plt.colorbar(label='Anomaly Status')
plt.show()