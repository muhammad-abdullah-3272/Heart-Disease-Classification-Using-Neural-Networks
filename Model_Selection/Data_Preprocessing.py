## Importing Necessary Libraries
import numpy as np
import pandas as pd
np.random.seed(2)

pd.options.mode.chained_assignment = None 

## Reading the Dataset 
data = pd.read_csv('heart.csv', encoding='unicode_escape')

def dataReplace(data):
    '''
    This fucntion converts string data type to int data type.
    The model takes the following arguments:
    
    Data (Features containing String data type)
    
    returns:
    Data (Features converted to Int data type)
    '''
    
    data['Sex'][data['Sex'] == 'F'] = 0                                    # Replace F with 0
    data['Sex'][data['Sex'] == 'M'] = 1                                    # Replace M with 1

    data['ChestPainType'][data['ChestPainType'] == 'ATA'] = 0              # Replace ATA with 0
    data['ChestPainType'][data['ChestPainType'] == 'NAP'] = 1              # Replace NAP with 1
    data['ChestPainType'][data['ChestPainType'] == 'ASY'] = 2              # Replace ASY with 2
    data['ChestPainType'][data['ChestPainType'] == 'TA'] = 3               # Replace TA with 3

    data['RestingECG'][data['RestingECG'] == 'Normal'] = 0                 # Replace Normal with 0
    data['RestingECG'][data['RestingECG'] == 'ST'] = 1                     # Replace ST with 1
    data['RestingECG'][data['RestingECG'] == 'LVH'] = 2                    # Replace LVH with 2

    data['ExerciseAngina'][data['ExerciseAngina'] == 'N'] = 0              # Replace N with 0
    data['ExerciseAngina'][data['ExerciseAngina'] == 'Y'] = 1              # Replace Y with 1

    data['ST_Slope'][data['ST_Slope'] == 'Up'] = 0                         # Replace Up with 0
    data['ST_Slope'][data['ST_Slope'] == 'Flat'] = 1                       # Replace Flat with 1
    data['ST_Slope'][data['ST_Slope'] == 'Down'] = 2                       # Replace Down with 2

    return data


def scale(data):
    '''
    This fucntion performs data scaling from -1 to 1 using min-max critera.
    The model takes the following arguments:
    
    Data (numpy array): Input dataset
    
    returns:
    Scaled Data (numpy array)
    '''
    
    dataScale = 2*((data - data.min()) / (data.max() - data.min())) - 1    # Feature Scaling from -1 to 1
    dataScale['HeartDisease'] = data['HeartDisease']                       # Not applying Scaling on Y
    return dataScale

data = dataReplace(data)                          # Calling the Data Conversion Function and assigning it to variable data
data = scale(data)                                # Calling the Feature Scaling Function and assigning it to variable data

# Splitting the Dataset into Train set(60%), Cross Validation set(20%) and Test set(20%).
train, val, test = np.split(data.sample(frac=1), [int(0.7 * len(data)), int(0.85 * len(data))])
#print(train.shape)
#print(val.shape)
#print(test.shape)



X_data = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG", "MaxHR", 
          "ExerciseAngina", "Oldpeak", "ST_Slope"]                # Extracting Features
Y_data = ["HeartDisease"]                                         # Extracting Labels

X_train = train[X_data]                                           # Assigning Features to X_train               
Y_train = train[Y_data]                                           # Assigning Features to Y_train

X_val = val[X_data]                                               # Assigning Features to X_val
Y_val = val[Y_data]                                               # Assigning Features to Y_val

X_test = test[X_data]                                             # Assigning Features to X_test
Y_test = test[Y_data]                                             # Assigning Features to Y_test


X_train = X_train.values                                          # Extracting values from X_train
Y_train = Y_train.values                                          # Extracting values from Y_train

X_val = X_val.values                                              # Extracting values from X_val
Y_val = Y_val.values                                              # Extracting values from Y_val

X_test = X_test.values                                            # Extracting values from X_test
Y_test = Y_test.values                                            # Extracting values from Y_test


print("Shape of X_train : ", X_train.shape)
print("Shape of Y_train : ", Y_train.shape)

print("Shape of X_val : ", X_val.shape)
print("Shape of Y_val : ", Y_val.shape)

print("Shape of X_test : ", X_test.shape)
print("Shape of Y_test : ", Y_test.shape)

