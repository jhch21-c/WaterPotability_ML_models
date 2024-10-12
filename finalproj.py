import pandas as pd
import numpy as np

from google.colab import files
files.upload()
#water_potability.csv is added to the file directory so that the path could be copied and accessed.

#training dataset
training_data = "/content/water_potability.csv"
#Creating a path for the code to access the water_potability.csv data file.

#printing the training dataset
df = pd.read_csv(training_data)
print(df.head())
df.isnull().sum()
#According to the dataframe, data points of ph, Sulfate, and Trihalomethanes has missing data for certain water samples.

"""The first true problem that was encountered was when analyzing the dataset, according to the dataset certain columns contain missing data for certain data samples. Certain water samples contain missing data on the ph, Sulfate, and Trihalomethanes of the water. Therefore; instead, of making up data which could lead to even more problematic and faulty results, it was a better idea to rather remove the columns from the dataset all together. So therefore, the dataset only includes Hardness, Solids, Chloramines, Conductivity, Organic cabron, Turbidity of the water and removes the ph, Sulfate, and Trihalomethanes data of the water. Even though accuracy is lost as there are less data points to determine whether or not a water is potable, it is still better than using faulty made up data which leads to faulty results."""

#we must shuffle the data as we don't know if all the potatabiilty = 1 is at the bottom of the df
sns.countplot(x = 'Potability', data = df)
#Testing the count of potable and non-potable water to determine whether there is enough data for potable water and non-potable water
shuffled = df.sample(frac=1)
print(shuffled)
#Turns out the ratio of potable and non-potable data is around 0.4-0.6, there is enough data for both potable and non-potable water and there is no data that is disproportionatley represented

#splitting the data into X and Y
X_data = shuffled.drop(['ph','Sulfate', 'Trihalomethanes', 'Potability'], axis = 1)
#indexes of columns dropped due to faulty data (some indexes have no data in the columns)
#The X_data is basically all the data that is required to determine whether or not a water is potable, the Y_data will be the result of the X_Data which would detect if the water is potable or not.
Y_data = shuffled['Potability']
X_data = (X_data - X_data.mean()) / X_data.std()
#Without adjusting the dataset, a sparse categorigal loss function is not possible because it reqruiers the outputs to be between 0 and 1, if the number is 100+ the data will not fit into a sparse categorical loss function, therefore adjustments are required
print(X_data.head())

print(X_data.shape)


#Data analysis using sns.distplot and sns.displot

#Remember that potable water count is lower than non-potable water count as the ratio of non-potable water is higher than potable water
#We are not comparing, count, only the distribution of the chracteristics of the water and comparing it to potability.
import seaborn as sns

sns.distplot(shuffled['Solids'], kde = True, bins = 45)
sns.displot(data=shuffled, x=shuffled['Solids'], hue = shuffled['Potability'])
#Solids: Distribution of Potable water is skewed slightly skewed right more than non-potable water; however the distributions are very similar.

sns.distplot(shuffled['Hardness'], kde = True, bins = 45)
sns.displot(data=shuffled, x=shuffled['Hardness'], hue = shuffled['Potability'])
#Hardness: The distribution of non-potable water is more unimodal than potable water; however, distributions are still very similar

sns.distplot(shuffled['Chloramines'], kde = True, bins = 45)
sns.displot(data=shuffled, x=shuffled['Chloramines'], hue = shuffled['Potability'])
#Chloramines: The distribution of non-potable and potable water is very similar (Potable water slightly skewed to the left while non-potable water has no apparent skew)

sns.distplot(shuffled['Conductivity'], kde = True, bins = 45)
sns.displot(data=shuffled, x=shuffled['Conductivity'], hue = shuffled['Potability'])
#Conductivity: The distribution of non-potable and potable water is very similar

sns.distplot(shuffled['Organic_carbon'], kde = True, bins = 45)
sns.displot(data=shuffled, x=shuffled['Organic_carbon'], hue = shuffled['Potability'])
#Organic_carbon: The distribution of non-potable and potable water is very similar

sns.distplot(shuffled['Turbidity'], kde = True, bins = 45)
sns.displot(data=shuffled, x=shuffled['Turbidity'], hue = shuffled['Potability'])
#Organic_carbon: The distribution of non-potable and potable water is very similar

"""**Data Analysis through seaborn distribution plots**

It was logical to attempt to find certain characteristics with potable and non-potable water. By comparing the distribution of characteristics in potable and non-potable water and potable water, it is easier to find certain patterns and trends that may explain the reason for the machine being able to dissect certain characteristics of potable and non-potable water. For example, if the distribution of Organic Carbon is very different among potable and non-potable water, it is significantly easier for models to detect that water with higher x-characteristics will not be potable. The opposite could be said true, if the distributions are very similar (which is this case), it may be significantly more difficult for models to detect whether the water is potable as characteristics among potable and non-potable water are so similar. As the distribution is so similar among all 6 characteristics, it is now upon the model to determine the certain combinations of characteristics which would lead to water potability or not, and that is for the model to determine. However, as distribution of certain characteristics is so similar among potable and non-potable water, accuracy rates regardless of any model may be significantly lower as the differences between potable and non-potable water are so obscure.
"""

#Check is Y_data is evenly distributed with no anomolies
print(Y_data)
print(Y_data.shape)

#Splitting the data into training and testing datasets, the creator of the dataset only gave one csv file of all the data
#However, it is absolutley neccesary to split the data set into 0.80 training data and 0.20 testing data so that there is enough data to train and test the data
#Shuffling was neccesary for this reason, to ensure that not all the "Potabliity=1" is not at the end of the csv file where the data will be split into train/test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size = 0.2, random_state = 12)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape) #X_train.shape is (2620,6) The correct dimmensions if the traning_size = 0.8
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#scaling the data with StandardScaler in order to allow the data to be able to be proccesed more easily machine-learning models.


print(X_train) #print to see if the data has been processed properly.

"""**First Model: Neural Network - Keras Sequential Model**"""

#Model 1
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import BatchNormalization
model = Sequential()

# The seqeuntial model in Keras is the classical model used when programming Neural Networks in Keras, providing a linear stack of layers where there is only one input and one ouput within each layer, giving it simplicitiy when programming.
model.add(Dense(32, input_dim = X_train.shape[1], activation= 'relu'))
#Input dimensions is basically the column of the X_dataset
#32 nodes, the same as MNIST, and activation function relu.
model.add(Dense(32, activation = 'relu'))
model.add(BatchNormalization())
#Batch normalization in order to allow for "noramlization", adding a layer of batch normalization will help run the model faster and more stable.
model.add(Dense(2, activation = 'softmax'))
model.add(BatchNormalization())
model.add(Dropout(0.7))
model.add(Dense(2, activation = 'softmax')) # Output layer
#4 Dense layers, signs of overfitting, possibly too large of a model is used
#Dropout layer is added in order to decrease the chances of overfitting.
opt = keras.optimizers.Adam(learning_rate = 0.01)
#Learning rate lowered to 0.01 in order to prevent overfitting
model.compile( optimizer = opt , loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'], )
model.fit(X_train, Y_train, batch_size = 5, epochs = 60, shuffle = True, verbose = 2 )
#Loss stuck at 0.6831 after 4 epochs with no signs of loss decreasing after that (loss has reached its maximum) and accuracry is around -55%, not a very accurate model and should be retested for higher accuracy
#Because loss has reached its maximum accuracy will no longer improve

#Inital issue was that even though loss was decreasing there were no signs of accuracy increasing, indicating overfitting.

modelResult = model.evaluate(X_test, Y_test)
print('Testing Accuracy: ', modelResult[1]*100)
#Testing accuracy = 61.2, could be improved

"""Trials Model 1:

Trial 1: Overfitting loss goes down but accuracy doesn't go up, but used the ph, sulfate,and Trihalomethanes columns in the data which was faulty as there was empty colums with no data, therefore it was innapropiate to assume that the indexes with no data = 0/median. Therefore I removed the columns as the data was faulty.

Trial 2: Overfitting, data is changed but the model is definitley overfitting, to prevent overfitting added a dropout layer. A new problem emerged as the loss didn't stop decreasing at 0.6931; therefore, from here a new model should be tested that could be more effective for the dataset
"""

#Model 2

from keras.callbacks import EarlyStopping

#Adding a EarlyStopping method allows for the machine to stop training when val_loss is constant or increasing, a custom earlystopping method is made in order to increase the patience of the early stopping method (allows the machine to run more epochs without stopping)
custom_early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=8,
    min_delta=0.001,
    mode='max'
)


model2 = Sequential()
model2.add(Dense(2, input_dim = X_train.shape[1], activation= 'relu'))
#Smaller dense layer in order to prevent overfitting. layer with 2 nodes is extremely small.
model2.add(Dropout(0.5))
model2.add(BatchNormalization())
model2.add(Dense(2, activation = 'softmax'))
model2.compile( optimizer = 'adam' , loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model2Data = model2.fit(X_train, Y_train, batch_size = 5, epochs = 100, validation_split=0.25, callbacks = [custom_early_stopping],  verbose = 2)

#First problem is that the machine keeps increasing it's epochs even if accuracy is dropping, therefore, there must be a way to stop the model when accuracy starts falling

modelResult2 = model2.evaluate(X_test, Y_test)
print('Testing Accuracy: ', modelResult2[1]*100)
#Testing accuracy = 61.28, could be improved

"""Model 2

Trial 1: Using a smaller model, the problem that is currently occuring is that the machine still keeps training even if accuracy is decreasing, there must be a method to automatically stop increasing epochs if accuracy keeps falling, keras.earlystopping must be implemented
"""

#Model 3: Putting it all together. including matplotlib to help graph out the loss to see if loss has terminally halted

#Comments are not repeated as they have been refrenced in previous models. The role of each layer, and feature is only combined.

#Large model with dropout layers.

import matplotlib.pyplot as plt
custom_early_stopping = EarlyStopping(
    monitor='val_Accuracy',
    patience=8,
    min_delta=0.001,
    mode='max'
)
#Early stopping still implemented: Saves memory and decreases chances of loosing accuracy rates due to overfitting
model3 = Sequential()

model3.add(Dense(100, input_dim = X_train.shape[1], activation= 'relu'))
model3.add(Dropout(0.8))
model3.add(BatchNormalization())
model3.add(Dense(100, activation = 'relu'))
model3.add(Dropout(0.8))
model3.add(Dense(1, activation = 'sigmoid'))

model3.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['Accuracy'])
model3Data = model3.fit(X_train, Y_train, batch_size = 5, epochs = 100, validation_split=0.25, callbacks = custom_early_stopping,  verbose = 2)
print(model3Data.history.keys())
#Plotting loss
#plotting loss to determine whether the loss has reached it's maximum.
plt.plot(model3Data.history['loss'])
plt.plot(model3Data.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'], loc='upper left')
plt.show()
#After plotting loss, it seems as if total loss and validation loss has reached the minimum, meaning the loss has reached it's asymptote, loss can no longer be reduced from the model.

modelResult3 = model3.evaluate(X_test, Y_test)
print('Testing Accuracy: ', modelResult3[1]*100)
#Testing accuracy = 61.28, could be improved

#Determining the indexes of the variables for the plt.plot()
print(model3Data.history.keys())



"""**Final Model Test evaluations**

Through implementing everything from the first trials of model1 to the 3rd trials of model3, the training and testing accuracy of the first model is still relatively low with merely low 60% training and testing accuracy rates. Overfitting was the first issue that was attempted to be solved; after adding more dropout layers and adjusting to make smaller neural networks the issue seemed to have been fixed. The problem was that even though loss kept decreasing the accuracy hardly increased (remained constant), which was a clear sign of overfitting from the initial model. After using a smaller model another problem has emerged, which was the sign that due to the high amount of epochs set, the model kept increasing epochs even though accuracy and val_accuracy kept decreasing, meaning the model kept re-training even with lower accuracy rates, again a clear sign of overfitting. The easiest way to solve this issue was to implement a Keras EarlyStopping method which would ensure that the model stopped if the val_accuracy was decreasing or remaining constant, automatically ensuring that the model would not overfit. Not only would EarlyStopping prevent overfitting, but it would also prevent unnecessary amounts of memory being used because of unnecessary re-running the model (increasing the epochs).

After best efforts to prevent overfitting through many ways, it became clear that it wasn’t actually the model’s fault that the accuracy was not rising after reaching the low 60% range. Instead the problem was that the training and testing accuracy of the model was low as the minimum loss was too high that it affected the model from increasing its accuracy rates. The accuracy rates of the model even after it has reached the minimum of the loss is still not very high, signaling a possible issue with nothing other than the data.

Data analysis has already indicated a prevalent issue with the data, that there is not really a way to sort potable and non-potable water as the distribution of the characteristics are so similar. Additionally, after having to remove ph, Sulfate, and Trihalomethanes data, the model had less data points to determine whether or not the water was actually potable or not, and therefore patterns and trends are less prevalent.

**Second model: KNeighborsClassifier Model (KNN)**

The KNN model is incredibly simple and easy to use meaning that only the training and testing data is required alongside the n_neighbors index. The n_neighbors can best be determined by graphing out all the data points and determining/calculating the distance between each data point; however, as that concept/method is out of the current curriculum (to determine the n_neighbors index with great precision), it was significantly more efficient/easier just to loop KNN models from n_neighbors values of 1 to 100 to determine what n_neighbors would lead to maximum accuracy rates. This was achieved through creating an array and storing all the accuracy of each model.
"""

#KNN Model testing

from sklearn.neighbors import KNeighborsClassifier

# Importing and fitting KNN classifier for k=3
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=100)
knn.fit(X_train,Y_train)
knn.score(X_test, Y_test) #Testing accuracy, only testing accuracy will be tested in KNN models.

# try K=1 through K=100 and record testing accuracy
k_range = range(1, 101)

# store all the trials with differing n_neighbors to determine which models determine the perfect fit in order to maximize accuracy rates
scores = []

# We use a loop through the range 1 to 100
# We append the scores in the array
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    scores.append(knn.score(X_test, Y_test))

print(scores)
max = np.max(scores) #Use python array methods to determine the maximum value and the index of the maximum value
max_index = scores.index(max)

print(max, max_index)
#Through the KNN model, we know that 0.614329 is the highest model accuracy rate when n_neighbors = 17

"""Evaluations

The KNN model lacks configurations compared to the Keras sequential model, and therefore it was hard to customize in order to create a model with the highest accuracy points. However, through looping through models with various n_neighbors, it was efficeint to find the n_neighbors index in order to maximize the testing accuracy rates. The conclusion was that the highest accuracy was 0.614329 with an n_neighbors index of 17.

Using a different model, the testing accuracy rate is still stuck in the low 60% range, meaning that after using two models, as the testing accuracy is basically stuck at the low 60% range, it is safer to assume that the relative low accuracy rates is not the fault of the models itself, but rather the consequence of faulty and insufficient input data.

**Final Model: Random Forest Classifier Model**
"""

#Last model: Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)
#n_estimators basically signifies # of trees in the forest, increasing n_estimators could lead to overfitting
#entropy is a type of random forest, other than the Gini has values inside [0, 0.5] while the entropy has [0,1]
classifier.fit(X_train,Y_train)
#Fit the random seed model, run the model through training data. Same as Keras Sequential model.
classifier.score(X_test, Y_test)
#Testing accuracy of the model.

classifier_range = range(1, 101)
classifier_scores = []
#Appending classifier scores into an array to determine which index would lead to maximum testing accuracy and the index of the max, which would lead to being able to find the right accuracy.

for i in classifier_range:
  classifier = RandomForestClassifier(n_estimators = i, criterion = 'entropy', random_state = 0)
  classifier.fit(X_train,Y_train)
  classifier_scores.append(classifier.score(X_test, Y_test))

print(classifier_scores) #Same array methods in order to find the maimum acc rate and testing accuracy
max_c = np.max(classifier_scores)
max_index_c = classifier_scores.index(max_c)

print(max_c, max_index_c)

"""I only used an extremely simple random forest classifier due to inexperience (There could be more customization with a random forest model (eg. min_sample_leaf) ; however, I have still been able to determine the best n_neighbors value and the accuracy rate of the value, which was 95 and 0.6402 respectively.

The random forest model als only determines a testing accuracy of around mid-60%, a relatively low testing accuracy rate, after running the data in 3 unique models, the machine still cannot determine an accuracy rate of above 80%, meaning that after trials of different models, it is becoming more apparent that the model may not be at fault, and rather that the input data is faulty.

# **CONCLUSION**
The conclusion is clear, that it is difficult to run a machine learning model through the characteristics of water that has been used to determine the potability of the water. The primary reason for the accuracy could be traced back to the difficulty of determining whether or not water is actually potable or not, as seen through the data analysis, it is clear that potable or non-potable water has very similar levels of each of the characteristics, and it is difficult to determine the levels of certain characteristics as they are so similar among non-potable and potable water. Every single minute detail matters when determining the potability of water, and therefore the attention to detail and precision required makes classification through machine learning ever more harder. The problem was exacerbated by needing to remove entire columns of data due to counts of missing data, and therefore, since the machine had less data points to reference while training, the consequence was that the model had a relatively lower accuracy rate. After using 3 unique machine learning models, and with all 3 of them having very similar testing accuracy rates, it is safe to conclude that the data is more problematic rather than the model itself.

To increase the accuracy of determining water potability, there could definitely be more accurate and less faulty data. The CSV file contained a significant proportion of data that was invalid (missing), and there were too few data points (columns, characteristics of water) that the model could reference to make a prediction. In addition, there could be certain characteristics of water that could be vastly different among potable and non-potable water, and by comparing characteristics machine learning models could more easily detect potable and non-potable water. If there are certain characteristics of water that would be vastly different among potable and non-potable water, detection would be significantly accurate and easier for the model.

Lastly, instead of determining potability as either 0 or 1, even though it sounds pretty unrealistic, the data could rather be the probability that the water is potable or not, which would allow for more accurate data analysis and determine the difference of a potable water dataset and a non-potable one, the difference would be clearer rather than making potability 0 or 1. Or instead of potability, the data could measure quality, as then there could be easier detection of which factors make water of higher quality; rather, than the computer having to do the arduous task of detecting the exact minute differences which make water potable or not.
"""