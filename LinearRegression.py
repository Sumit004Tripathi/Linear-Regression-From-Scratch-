import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('x_vs_y_experience_salary.csv')
print(data)

# Mean Squared Error Function
def meanSquaredError(m, b, points):
    totalError = 0
    for i in range(len(points)):
        x = points.iloc[i].experience
        y = points.iloc[i].salary
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))  

# Gradient Descent Function
def gradientDescent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    N = float(len(points))

    for i in range(int(N)):  
        x = points.iloc[i].experience
        y = points.iloc[i].salary
        m_gradient += -(2/N) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/N) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L

    return m, b

# Initialize Parameters
m = 0
b = 0
L = 0.0001  # Learning rate
epochs = 1000  # Number of iterations

# Perform Gradient Descent
for i in range(epochs):
    m, b = gradientDescent(m, b, data, L)

print(f"Slope (m): {m}, Intercept (b): {b}")

# Plot the data points
plt.scatter(data.experience, data.salary, color="black")

# Plot the regression line
plt.plot(data.experience, [m * x + b for x in data.experience], color="red")

# Show the plot
plt.xlabel("Experience (years)")
plt.ylabel("Salary ($)")
plt.title("Experience vs. Salary Regression")
plt.show()
