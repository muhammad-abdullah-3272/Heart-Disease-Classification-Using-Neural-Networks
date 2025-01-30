#------------------------------------------------------------------------------
# Importing Model from the file 'Model'
#------------------------------------------------------------------------------
from Model import Model

# Instantiate the Model
model = Model()

# Prediction Function
def prediction(X):
    '''
    Parameters
    ----------
    X : TYPE Numpy Array
        DESCRIPTION. Data Features

    Returns
    -------
    TYPE Numpy Array / Int
        DESCRIPTION. Predicted Output
    '''
    # Returns Prediction    
    return model.Predict(X)


if __name__ == "__main__":

    # Unseen Data Features
    features = [[37, 0, 1, 130, 211, 0, 0, 142, 0, 0, 0],
                [58, 1, 0, 136, 164, 0, 1, 99, 1, 2, 1],
                [39, 1, 0, 120, 204, 0, 0, 145, 0, 0, 0],
                [49, 1, 2, 140, 234, 0, 0, 140, 1, 1, 1],
                [42, 0, 1, 115, 211, 0, 1, 137, 0, 0, 0],
                [54, 0, 0, 120, 273, 0, 0, 150, 0, 1.5, 1]]
    

    # Predict the output of the given Features 
    y = prediction(features)
    print(y)
