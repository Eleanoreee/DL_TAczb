import numpy as np

# Generate some sample data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.rand(100, 1)

# Linear regression using gradient descent
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    for i in range(iterations):
        # Calculate the predictions
        predictions = X.dot(theta)
        
        # Calculate the error
        error = predictions - y
        
        # Update the coefficients (theta) using gradient
        gradient = X.T.dot(error) / m
        theta -= learning_rate * gradient
        
    return theta

# Add a bias term (intercept) to the input features
X_b = np.c_[np.ones((100, 1)), X]

# Initialize coefficients (theta) with zeros
theta = np.zeros((2, 1))

# Set the learning rate and number of iterations
learning_rate = 0.1
iterations = 1000

# Perform gradient descent to find the optimal coefficients
theta = gradient_descent(X_b, y, theta, learning_rate, iterations)

# Display the coefficients (theta)
print("Optimal Coefficients (theta):")
print(theta)