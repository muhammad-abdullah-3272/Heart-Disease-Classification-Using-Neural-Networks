#------------------------------------------------------------------------------
# Importing Necessary libraires  
#------------------------------------------------------------------------------
import pandas as pd

#------------------------------------------------------------------------------
# Importing Datasets
#------------------------------------------------------------------------------
from Data_Preprocessing import Data

#------------------------------------------------------------------------------
# Importing Classes from the file 'Classes'
#------------------------------------------------------------------------------
from Classes import Dense_Layer, ReLU_Activation, Sigmoid_Activation, BinaryCrossEntropy_Loss, GD_Optimizer, Categorical_Accuracy

#------------------------------------------------------------------------------
# Importing Model from the file 'Model'
#------------------------------------------------------------------------------
from Model import Model



## Reading the Dataset 
HeartData = pd.read_csv('heart.csv', encoding='unicode_escape')

# Instantiate the class Data
data = Data()

# Replace string to int
data.Replace(HeartData)

# Scale the data. Perform -1 to 1 Normalization
data.Scale()

# Split the data into Training, Validation and Test sets. Also can change the distribution below.
X_train, Y_train, X_val, Y_val, X_test, Y_test = data.Split(0.7, 0.85)


# Instantiate the Model
model = Model()

# Add Layers (Dense Layers and Activation Layers can be added here according to any configuration)
# Number of Neurons in each layer can also be changed, lambda can also be configured 
model.Add(Dense_Layer(11, 20, lamda=0.0003))
model.Add(ReLU_Activation())
model.Add(Dense_Layer(20, 1))
model.Add(Sigmoid_Activation())

# Set Loss, Optimizer (also set optimizer parameters) and Accuracy Objects
model.Set(loss=BinaryCrossEntropy_Loss(), optimizer=GD_Optimizer(alpha=0.05, decay=0), accuracy=Categorical_Accuracy())

# Finalize the Model
model.Finalize()

# Train the Model, Number of epochs can also be configured
model.Train(X_train, Y_train, validation_data=(X_val, Y_val), epochs=1000, print_every=100)

# Test the Model on Test Data set
model.Test(test_data=(X_test, Y_test))

# Save Optimized Parameters of the Model 
model.Save_Params('HeartDisease.params')