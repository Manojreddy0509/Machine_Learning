import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

time_studied=np.array([20,50,32,65,23,43,10,5,22,35,29,5,56]).reshape(-1,1)
test_scores=np.array([56,83,47,93,47,82,45,78,55,67,57,4,12]).reshape(-1,1)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
#model.fit(time_studied, test_scores)
#print(model.predict(np.array([[10]]).reshape(-1,1)) ) # Predict score for 40 hours of study
#plt.scatter(time_studied, test_scores, color='blue', label='Data Points')
#plt.plot(np.linspace(0,70,100).reshape(-1,1), model.predict(np.linspace(0,70,100).reshape(-1,1)), "r")

#plt.show()
# Split the data into training and testing sets
time_train, time_test, score_train, score_test = train_test_split(time_studied, test_scores, test_size=0.2)
# Fit the model to the training data
model.fit(time_train, score_train)
# Predict scores for the test set
predictions = model.predict(time_test)
# Print the predictions
print("Predictions for test set:", predictions.flatten())
print(model.score(time_test, score_test))  # Print the model's score on the test set