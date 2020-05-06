import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import pylab as pl
from sklearn import linear_model
from sklearn.metrics import r2_score


#Get the dataframe to work with the number of deaths
df = pd.read_csv("ccaa_covid19_fallecidos_long.csv")


#Select only the cases in Madrid from the number of deaths dataset
Madrid_df_muertes=df[df['CCAA']=='Madrid']

print(Madrid_df_muertes)

#Get the dataframe to work with the number of cases
df2 = pd.read_csv("ccaa_covid19_casos_long.csv")

#Select only the cases in Madrid from the number of cases dataset
Madrid_df_casos=df2[df2['CCAA']=='Madrid']

print(Madrid_df_casos)

#Have now 73 days to analysis in our data from the Madrid Community

#Plot the Number of cases vs number of days
plt.scatter(range(len(Madrid_df_casos)),Madrid_df_casos.total,color='blue')
plt.title("Number of cases (Madrid) vs days")
plt.xlabel("Days")
plt.ylabel("Number of Cases")
plt.show()

#Plot the Number of deaths vs number of days
plt.scatter(range(len(Madrid_df_casos)),Madrid_df_muertes.total,color='red')
plt.title("Number of deaths (Madrid) vs days")
plt.xlabel("Days")
plt.ylabel("Number of deaths")
plt.show()

#Select 80% of the data to the train set and 20% for the test set
msk = np.random.rand(len(Madrid_df_casos)) <0.8
train = Madrid_df_muertes[msk]
train_2 = Madrid_df_casos[msk]
test = Madrid_df_muertes[~msk]
test_2 = Madrid_df_casos[~msk]

#Plot the number of deaths vs number of cases
plt.scatter(train_2.total, train.total, color='red', label='training data')
plt.legend(loc='upper left')
plt.title("Number of Deaths vs Number of cases (Madrid)")
plt.xlabel("Number of cases")
plt.ylabel("Number of deaths")
plt.scatter(test_2.total, test.total, color='green', label='testing data')
plt.legend(loc='upper left')
plt.show()

#Now we know that the behaviour is more or less linear

#Apply the Linear Regression model
#We upload the model into the variable regr
regr=linear_model.LinearRegression()

#We treat the train data with np.asanyarray
train_x = np.asanyarray(train_2[['total']])
train_y = np.asanyarray(train[['total']])

#Apply the Linear Regression model
regr.fit(train_x, train_y)

#now obtain the coefficients of your model
print('Coefficients; ', regr.coef_)
print('Intercept: ', regr.intercept_)

# i.e: y = -373.4907455 + 0.12483009*x

#Plot the training and testing set with the line created in the model
plt.scatter(train_2.total, train.total, color='red', label='training data')
plt.legend(loc='upper left')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-b')
plt.title("Number of Deaths vs Number of cases (Madrid)")
plt.xlabel("Number of cases")
plt.ylabel("Deaths")
plt.scatter(test_2.total, test.total, color='green', label='testing data')
plt.legend(loc='upper left')
plt.show()


#Evaluate the model
#Select the testing values for number of cases and number of deaths
test_x = np.asanyarray(test_2[['total']])
test_y = np.asanyarray(test[['total']])

#Obtain the prediction
test_y_hat = regr.predict(test_x)

#Obtain the MAE, MSE and R2 score
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )

#i.e: 
#MAE = 337.63
#MSE = 166028.72
#R2 score = 0.98



