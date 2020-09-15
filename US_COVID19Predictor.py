import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('national-history.csv')

#print(df)

X = df[['day', 'totalTestResultsIncrease']].values
y = df['positiveIncrease'].values

print("US_COVID19Predictor")
model = LinearRegression()
model.fit(X, y)
ans = input("Do you want to see the pattern of the pandemic? ")
if ans=='y':
    plt.figure(figsize=(10, 6))
    plt.scatter(df['totalTestResultsIncrease'], df['positiveIncrease'])
    plt.title('Pattern of the Pandemic')
    plt.xlabel('totalTestResultsIncrease')
    plt.ylabel('positiveIncrease')
    plt.xlim(1, 1.5e+6)
    plt.ylim(0, 1.0e+5)
    plt.plot(X, model.predict(X), color='red')
    
    plt.show()
ans = input("Do you want to see the general developement of the pandemic day to day? ")
if ans == 'y':
    plt.scatter(df['day'], df['positiveIncrease'])
    plt.title('PositiveIncrease Per Day')
    plt.xlabel('day')
    plt.ylabel('positiveIncrease')
    plt.xlim(1, 2.0e+2)
    plt.ylim(0, 1.0e+5)
    plt.show()

#Slope Coefficient/ theta_1
print("Slope = " + str(model.coef_))


#Intercept
print("Intercept = " + str(model.intercept_))


#Prediction
print("Day 200, Among 678,645 test results "+str(model.predict([[365, 678645]]))+" positive people.")
print("Day 245, Among 1,000,000 test results "+str(model.predict([[245, 1000000]]))+" positive people.")
print("Day 320, Among 1,000,000 test results "+str(model.predict([[320, 1000000]]))+" positive people.")
print("Day 400, Among 700,000 test results "+str(model.predict([[400, 700000]]))+" positive people.")
print("Day 500, Among 700,000 test results "+str(model.predict([[500, 700000]]))+" positive people.")

#Score
print("Accuracy of the program = "+str(model.score(X, y)))
