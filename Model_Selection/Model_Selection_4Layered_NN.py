# Importing Necessary libraires  
import numpy as np
import matplotlib.pyplot as plt

# Importing Datasets
from Data_Preprocessing import X_train, Y_train, X_val, Y_val, X_test, Y_test


# Initializing Class for Dense layers in Neural Network
class Layer_Dense:
    # Layer Initialization
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l2=0):
        '''
        Parameters
        ----------
        n_inputs : TYPE Integer
            DESCRIPTION.
        n_neurons : TYPE Integer
            DESCRIPTION.
        weight_regularizer_l2 : TYPE Integer '0', optional
            DESCRIPTION. This function creates Network's layers and initializes wieghts and biases

        Returns
        -------
        None.

        '''
        # Initializing weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
        # Initializing regularization parameter (lambda)
        self.weight_regularizer_l2 = weight_regularizer_l2
    
    # Forward Pass
    def forward(self, xTrain, xVal):
        '''
        Parameters
        ----------
        xTrain : TYPE Numpy Array
        xVal : TYPE Numpy Array
            DESCRIPTION: This function takes training and validation datasset for forward pass

        Returns
        -------
        None.

        '''
        # Save Input Datasets values
        self.xTrain = xTrain
        self.xVal = xVal
        
        # Calculate outputs from input datasets, weights and biases
        self.output = np.dot(xTrain, self.weights) + self.biases
        self.val_output = np.dot(xVal, self.weights) + self.biases
        
        
    # Backward Pass
    def backward(self, dvalues):
        '''
        Parameters
        ----------
        dvalues : TYPE Numpy Array
            DESCRIPTION: This function performs backprop using chain rule

        Returns
        -------
        None.

        '''
        # Gradients on parameters
        self.dweights = np.dot(self.xTrain.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # Gradient on regularization, L2 Regularization on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights = self.dweights + 2 * self.weight_regularizer_l2 * self.weights
            
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
        
        
# ReLU Activation
class Activation_ReLU:
    # Forward Pass
    def forward(self, xTrain, xVal):
        '''
        Parameters
        ----------
        xTrain : TYPE Numpy Array
        xVal : TYPE Numpy Array
            DESCRIPTION: This function performs ReLU activation on training and validation datasset 
            while forward propagation

        Returns
        -------
        None.

        '''
        # Save input datasets values
        self.xTrain = xTrain
        self.xVal = xVal
        
        # Calculate outputs from input datasets using ReLU
        self.output = np.maximum(0, xTrain)
        self.val_output = np.maximum(0, xVal)
        
    # Backward Pass
    def backward(self, dvalues):
        '''
        Parameters
        ----------
        dvalues : TYPE Numpy Array
            DESCRIPTION: This function performs backprop on ReLU activation

        Returns
        -------
        None.

        '''
        # To modify original variabes, make a copy of it
        self.dinputs = dvalues.copy()
        
        # Zero gradient where input values are negative 
        self.dinputs[self.xTrain <= 0] = 0


# Sigmoid Activation
class Activation_Sigmoid:
    # Forward Pass
    def forward(self, xTrain, xVal):
        '''
        Parameters
        ----------
        xTrain : TYPE Numpy Array
        xVal : TYPE Numpy Array
            DESCRIPTION: This function performs Sigmoid activation on training and validation datasset 
            while forward propagation

        Returns
        -------
        None.

        '''
        # Remeber input datasets values
        self.xTrain = xTrain
        xTrain = xTrain.astype('float64')
        self.xVal = xVal
        xVal = xVal.astype('float64')
        
        # Calculate outputs from input datasets using Sigmoid
        self.output = 1 / (1 + np.exp(-xTrain)) 
        self.val_output = 1 / (1 + np.exp(-xVal)) 
        
    # Backward Pass
    def backward(self, dvalues):
        '''
        Parameters
        ----------
        dvalues : TYPE Numpy Array
            DESCRIPTION: This function performs backprop on Sigmoid activation

        Returns
        -------
        None.

        '''
        # Derivative: caculates from the output of Sigmoid Function 
        self.dinputs = dvalues * (1 - self.output)  * self.output
        
        
# Gradient Descent
class Optimizer_GD:
    # Initialize Optimizer - Set settings
    # Learning rate is set as 0.7
    def __init__(self, learning_rate=0.7, decay=0):
        '''
        Parameters
        ----------
        learning_rate : TYPE Int, optional
            DESCRIPTION. The default is 0.7.
        decay : TYPE Int, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        '''
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        
    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
      
    # Update Parameters
    def update_params(self, layer):
        '''
        Parameters
        ----------
        layer : TYPE Numpy Array
            DESCRIPTION: This function updates the parameters in each corresponding layer

        Returns
        -------
        None.

        '''
        layer.weights = layer.weights - self.current_learning_rate * layer.dweights
        layer.biases = layer.biases - self.current_learning_rate * layer.dbiases
    
    # Call once before any parameter update
    def post_update_params(self):
        self.iterations = self.iterations + 1


# Common Loss Class
class Loss:
    # Regularization Loss Calculation
    def regularization_loss(self, layer):
        '''
        Parameters
        ----------
        layer : TYPE Numpy Array
            DESCRIPTION.Regularization loss is calculated in each layer

        Returns
        -------
        regularization_loss : TYPE Int
            DESCRIPTION. Scalar value is returned as regularization loss

        '''
        regularization_loss = 0                     # 0 as default
        
        # L2 Regularization - weights, calculate only when factor greater than 0
        if layer.weight_regularizer_l2 > 0:         
            regularization_loss = regularization_loss + layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
        return regularization_loss
    
    # Calculate the data and regularization loss given model output and actual labels
    def calculate(self, output, y):
        '''
        Parameters
        ----------
        output : TYPE Numpy Array
            DESCRIPTION. Model Output (Predicted Values)
        y : TYPE Numpy Array
            DESCRIPTION. Actual labels

        Returns
        -------
        data_loss : TYPE Int
            DESCRIPTION. Mean Loss

        '''
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        return data_loss


# Binary Cross Entropy Loss
class Loss_BinaryCrossEntropy(Loss):
    # Forward Pass
    def forward(self, y_pred, y_true):
        '''
        Parameters
        ----------
        y_pred : TYPE Numpy Array
            DESCRIPTION. Predicted values
        y_true : TYPE Numpy Array
            DESCRIPTION. Actual labels

        Returns
        -------
        sample_losses : TYPE Numpy Array
            DESCRIPTION. loss for each sample in the data 

        '''
        # Clip data to prevent division by 0, clipping both sides to avoid mean dragging to any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # Calculate samle wise loss 
        sample_losses = -(y_true * np.log(y_pred_clipped) + ((1 - y_true) * np.log(1 - y_pred_clipped)))
        sample_losses = np.mean(sample_losses, axis=-1)
        return sample_losses
    
    # Backward Pass
    def backward(self, dvalues, y_true):
        '''
        Parameters
        ----------
        dvalues : TYPE Numpy Array
            DESCRIPTION. backprop data values
        y_true : TYPE Numpy Array
            DESCRIPTION. Actual labels

        Returns
        -------
        None.

        '''
        # Number of samples
        samples = len(dvalues)
        
        # Number of outputs in every sample
        outputs = len(dvalues[0])
        
        # Clip data to prevent division by 0, clipping both sides to avoid mean dragging to any value
        dvalues_clipped = np.clip(dvalues, 1e-7, 1 - 1e-7)
        
        # Calculate gradient
        self.dinputs = -(y_true / dvalues_clipped - (1 - y_true) / (1 - dvalues_clipped)) / outputs
        
        # Normalize gradient
        self.dinputs = self.dinputs / samples
        
    
    # Create dense layer with 11 input features and 20 output values
    dense1 = Layer_Dense(11, 20, weight_regularizer_l2=0.001)
    
    # Create ReLU activation (to be used with upper dense layer)
    activation1 = Activation_ReLU()
    
    # Create dense layer with 20 input values from above layer and 20 output values
    dense2 = Layer_Dense(20, 20, weight_regularizer_l2=0.001)
    
    # Create ReLU activation (to be used with upper dense layer)
    activation2 = Activation_ReLU()
    
    # Create dense layer with 20 input values from above layer and 1 output prediction value
    dense3 = Layer_Dense(20, 1)
    
    # Create Sigmoid activation (to be used with upper dense layer), for classification
    activation3 = Activation_Sigmoid()
    
    # Create Loss Function
    loss_function = Loss_BinaryCrossEntropy()
    
    # Create Optimizer Function
    optimizer = Optimizer_GD(decay=5e-4)
    
    trainCost = []               # Train Cost Array for cost from each iteration 
    valCost = []                 # Validation Cost Array for cost from each iteration 
    
    # Train in Loop
    for epoch in range(1001):
        
        # Perform forward pass on our training and validation data through this layer
        dense1.forward(X_train, X_val)
        
        # Perform forward pass through activation function, takes output of 1st dense layer
        activation1.forward(dense1.output, dense1.val_output)
        
        # Perform forward pass through 2nd dense layer, takes output of activation of 1st layer
        dense2.forward(activation1.output, activation1.val_output)
    
        # Perform forward pass through activation function, takes output of 2nd dense layer
        activation2.forward(dense2.output, dense2.val_output)
        
        # Perform forward pass through 3rd dense layer, takes output of activation of 2nd layer
        dense3.forward(activation2.output, activation2.val_output)
    
        # Perform forward pass through activation function, takes output of 3rd dense layer
        activation3.forward(dense3.output, dense3.val_output)
        
    
        # Calculate the training and validation data loss
        train_data_loss = loss_function.calculate(activation3.output, Y_train)
        val_data_loss = loss_function.calculate(activation3.val_output, Y_val)
        
        # Calculate regularization penalty in all dense layers
        regularization_loss = loss_function.regularization_loss(dense1) + \
                              loss_function.regularization_loss(dense2) + \
                              loss_function.regularization_loss(dense3)
        
        # Calculate the total loss
        loss = train_data_loss + regularization_loss
        
        # Append all training and validation losses in the arrays initialzed above
        trainCost.append(train_data_loss)
        valCost.append(val_data_loss)
        
        # Calculate accuracy from output of activaion3 and actual labels
        predictions = (activation3.output > 0.5) * 1
        accuracy = np.mean(predictions == Y_train) 
        
        if not epoch % 100:
            print(f'epoch: {epoch}, ' +
                  f'Accuracy: {accuracy:.3f}, ' +  
                  f'Loss: {loss:.3f}, ' +
                  f'Data Loss: {train_data_loss:.3f}, ' +
                  f'Regularization Loss: {regularization_loss:.3f}, ' +
                  f'Learning Rate: {optimizer.current_learning_rate:.3f}') 
                
        # Bacward pass from right to left (chain rule)
        loss_function.backward(activation3.output, Y_train)
        activation3.backward(loss_function.dinputs)
        dense3.backward(activation3.dinputs)
        activation2.backward(dense3.dinputs)
        dense2.backward(activation2.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)
        
        # Update weights and biases
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.update_params(dense3)
        optimizer.post_update_params()
    
       
plt.figure()
plt.plot(trainCost)
plt.plot(valCost)
plt.title("Training and Validation Costs")
plt.xlabel('Iterations')
plt.ylabel('Cost Error')
plt.legend(["Training Cost", "Validation Error"], loc ="upper right")
plt.show()