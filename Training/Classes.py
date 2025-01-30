#------------------------------------------------------------------------------
# Importing Necessary libraires  
#------------------------------------------------------------------------------
import numpy as np


#==============================================================================
# Initializing Class for Dense layers in Neural Network
#==============================================================================
class Dense_Layer:
    #--------------------------------------------------------------------------
    # Layer Initialization
    #--------------------------------------------------------------------------
    def __init__(self, n_inputs, n_neurons, lamda=0):
        '''
        Parameters
        ----------
        n_inputs : TYPE Integer
            DESCRIPTION.
        n_neurons : TYPE Integer
            DESCRIPTION.
        lamda : TYPE Integer '0', optional
            DESCRIPTION. This function creates Network's layers and initializes wieghts and biases

        Returns
        -------
        None.
        '''
        # Initializing weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
        # Initializing regularization parameter (lambda)
        self.lamda = lamda    
        
    #--------------------------------------------------------------------------
    # Forward Propagation
    #--------------------------------------------------------------------------
    def Forward(self, inputs, training):
        '''
        Parameters
        ----------
        inputs : TYPE Numpy Array
            DESCRIPTION: This function takes training datasset for forward pass
        training : TYPE optional
            DESCRIPTION. It is kept 'True' only when training is performed on training dataset.
                         Otherwise, it is kept 'False'

        Returns
        -------
        None.
        '''
        # Save Input Datasets values
        self.inputs = inputs
        
        # Calculate outputs from input datasets, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
        
    #--------------------------------------------------------------------------   
    # Backward Propagation
    #--------------------------------------------------------------------------
    def Backward(self, d_values):
        '''
        Parameters
        ----------
        d_values : TYPE Numpy Array
            DESCRIPTION: This function performs backprop using chain rule

        Returns
        -------
        None.
        '''
        # Gradients on parameters lambdas and biases
        self.d_weights = np.dot(self.inputs.T, d_values)
        self.d_biases = np.sum(d_values, axis=0, keepdims=True)
        
        # Gradient on regularization, L2 Regularization on weights
        if self.lamda > 0:
            self.d_weights = self.d_weights + 2 * self.lamda * self.weights
            
        # Gradient on values
        self.d_inputs = np.dot(d_values, self.weights.T)
        
    #--------------------------------------------------------------------------
    # Retrieve Layer Parameters
    #--------------------------------------------------------------------------
    def Get_Params(self):
        return self.weights, self.biases
    
    #--------------------------------------------------------------------------
    # Set Weights and Biases in a Layer Instance
    #--------------------------------------------------------------------------
    def Set_Params(self, weights, biases):
        self.weights = weights
        self.biases = biases
        

#==============================================================================
# Class for Input Layer of the Neura Netowk
#==============================================================================
class Input_Layer():
    #--------------------------------------------------------------------------
    # Forward Propagation
    #--------------------------------------------------------------------------
    def Forward(self, inputs, training):
        self.output = inputs
        
#==============================================================================        
# ReLU Activation
#==============================================================================
class ReLU_Activation:
    #--------------------------------------------------------------------------
    # Forward Propagation
    #--------------------------------------------------------------------------
    def Forward(self, inputs, training):
        '''
        Parameters
        ----------
        inputs : TYPE Numpy Array
            DESCRIPTION: This function performs ReLU activation on training datasset 
            while forward propagation

        Returns
        -------
        None.
        '''
        # Save input datasets values
        self.inputs = inputs
        
        # Calculate outputs from input datasets using ReLU
        self.output = np.maximum(0, inputs)
        
    #--------------------------------------------------------------------------    
    # Backward Propagation
    #--------------------------------------------------------------------------
    def Backward(self, d_values):
        '''
        Parameters
        ----------
        d_values : TYPE Numpy Array
            DESCRIPTION: This function performs backprop on ReLU activation

        Returns
        -------
        None.
        '''
        # To modify original variabes, make a copy of it
        self.d_inputs = d_values.copy()
        
        # Zero gradient where input values are negative 
        self.d_inputs[self.inputs <= 0] = 0

#==============================================================================
# Sigmoid Activation
#==============================================================================
class Sigmoid_Activation:
    #--------------------------------------------------------------------------
    # Forward Pass
    #--------------------------------------------------------------------------
    def Forward(self, inputs, training):
        '''
        Parameters
        ----------
        inputs : TYPE Numpy Array
            DESCRIPTION: This function performs Sigmoid activation on training datasset 
            while forward propagation

        Returns
        -------
        None.
        '''
        # Save input datasets values
        self.inputs = inputs
        inputs = inputs.astype('float64')
        
        # Calculate outputs from input datasets using Sigmoid
        self.output = 1 / (1 + np.exp(-inputs)) 
    
    #--------------------------------------------------------------------------
    # Backward Propagation
    #--------------------------------------------------------------------------
    def Backward(self, d_values):
        '''
        Parameters
        ----------
        d_values : TYPE Numpy Array
            DESCRIPTION: This function performs backprop on Sigmoid activation

        Returns
        -------
        None.
        '''
        # Derivative: caculates from the output of Sigmoid Function 
        self.d_inputs = d_values * (1 - self.output)  * self.output
        
    #--------------------------------------------------------------------------
    # Calculate Predictions for Outputs
    #--------------------------------------------------------------------------    
    def Predictions(self, outputs):
        return (outputs > 0.5) * 1

#==============================================================================        
# Gradient Descent
#==============================================================================
class GD_Optimizer:
    #--------------------------------------------------------------------------
    # Initialize Optimizer - Set settings
    # Learning rate (alpha) is set as 0.7
    #--------------------------------------------------------------------------
    def __init__(self, alpha=0.1, decay=0):
        '''
        Parameters
        ----------
        alpha : TYPE Int, optional
            DESCRIPTION. The default is 0.1.
        decay : TYPE Int, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.
        '''
        self.alpha = alpha
        self.current_alpha = alpha
        self.decay = decay
        self.iterations = 0
        
    #--------------------------------------------------------------------------    
    # Call once before any parameter updates
    #--------------------------------------------------------------------------
    def Pre_Update_Params(self):
        if self.decay:
            self.current_alpha = self.alpha * (1. / (1. + self.decay * self.iterations))
      
    #--------------------------------------------------------------------------    
    # Update Parameters
    #--------------------------------------------------------------------------
    def Update_Params(self, layer):
        '''
        Parameters
        ----------
        layer : TYPE Numpy Array
            DESCRIPTION: This function updates the parameters in each corresponding layer

        Returns
        -------
        None.
        '''
        layer.weights = layer.weights - self.current_alpha * layer.d_weights
        layer.biases = layer.biases - self.current_alpha * layer.d_biases
    
    #--------------------------------------------------------------------------
    # Function to call once before any update of parameters
    #--------------------------------------------------------------------------
    def Post_Update_Params(self):
        self.iterations = self.iterations + 1

#==============================================================================
# Class to Calculate Common Loss
#==============================================================================
class Loss:
    #--------------------------------------------------------------------------
    # Regularization Loss Calculation
    #--------------------------------------------------------------------------
    def Regularization_Loss(self):
        '''
        Parameters
        ----------
        None

        Returns
        -------
        loss_regularization : TYPE Int
            DESCRIPTION. Scalar value is returned as regularization loss
        '''
        regularization_loss = 0                     # 0 as default
        
        # Iterate all Trainable Layers
        for layer in self.trainable_layers:
            # L2 Regularization - weights, calculate only when factor greater than 0
            if layer.lamda > 0:         
                regularization_loss = regularization_loss + layer.lamda * np.sum(layer.weights * layer.weights)
        return regularization_loss
    
    #--------------------------------------------------------------------------
    # Remeber Trainable Layers
    #--------------------------------------------------------------------------
    def Trainable_Layers(self, trainable_layers):
        '''
        Parameters
        ----------
        trainable_layers : TYPE Numpy Array
            DESCRIPTION. Layers which are comprised of weights are saved as trainable layers

        Returns
        -------
        None.
        '''
        self.trainable_layers = trainable_layers
    
    #--------------------------------------------------------------------------
    # Calculate the data and regularization loss for given model output and actual labels
    #--------------------------------------------------------------------------
    def Calculate(self, output, y, *, include_lamda=False):
        '''
        Parameters
        ----------
        predicted : TYPE Numpy Array
            DESCRIPTION. Model Output (Predicted Values)
        labels : TYPE Numpy Array
            DESCRIPTION. Actual labels

        Returns
        -------
        data_loss : TYPE Int
            DESCRIPTION. Mean Loss
        '''
        # Calculate Sample Losses
        sample_losses = self.Forward(output, y)
        
        # Calculate Mean Loss
        data_loss = np.mean(sample_losses)
        
        # If just data loss; return it 
        if not include_lamda:
            return data_loss
        return data_loss, self.Regularization_Loss()

#==============================================================================
# Binary Cross Entropy Loss
#==============================================================================
class BinaryCrossEntropy_Loss(Loss):
    #--------------------------------------------------------------------------
    # Forward Propagation
    #--------------------------------------------------------------------------
    def Forward(self, y_pred, y_true):
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
        # Clip the data to avoid division by 0, clipping both sides to avoid mean dragging to any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # Calculate samle wise loss 
        sample_losses = -(y_true * np.log(y_pred_clipped) + ((1 - y_true) * np.log(1 - y_pred_clipped)))
        sample_losses = np.mean(sample_losses, axis=-1)
        return sample_losses
    
    #--------------------------------------------------------------------------
    # Backward Propagation
    #--------------------------------------------------------------------------
    def Backward(self, d_values, y_true):
        '''
        Parameters
        ----------
        d_values : TYPE Numpy Array
            DESCRIPTION. backprop data values
        y_true : TYPE Numpy Array
            DESCRIPTION. Actual labels

        Returns
        -------
        None.
        '''
        # Number of samples in the data
        samples = len(d_values)
        
        # Number of outputs in every sample
        outputs = len(d_values[0])
        
        # Clip data to avoid division by 0, clipping both sides to avoid mean dragging to any value
        dvalues_clipped = np.clip(d_values, 1e-7, 1 - 1e-7)
        
        # Calculate gradient
        self.dinputs = -(y_true / dvalues_clipped - (1 - y_true) / (1 - dvalues_clipped)) / outputs
        
        # Normalize gradient
        self.d_inputs = self.dinputs / samples
        
#==============================================================================
# Common Accuracy Class 
#==============================================================================
class Accuracy:
    #--------------------------------------------------------------------------
    # Calculate Accuracy given predictions and labels
    #--------------------------------------------------------------------------
    def Calculate(self, predictions, y):
        '''
        Parameters
        ----------
        predictions : TYPE Numpy Array
            DESCRIPTION.Predicted Values
        y : TYPE Numpy Array
            DESCRIPTION. Actual Values

        Returns
        -------
        accuracy : TYPE Int
            DESCRIPTION. Scalar Value to determine accuracy of the model
        '''
        comparisons = self.Compare(predictions, y)
        accuracy = np.mean(comparisons)
        return accuracy
        
#==============================================================================
# Accuracy Calculation for Classification Model
#==============================================================================
class Categorical_Accuracy(Accuracy):
    #--------------------------------------------------------------------------
    # No initialization Needed
    #--------------------------------------------------------------------------
    def init(self, y):
        pass
    
    #--------------------------------------------------------------------------
    # Compare Prediction to the Labels
    #--------------------------------------------------------------------------
    def Compare(self, predictions, y):
        '''
        Parameters
        ----------
        predictions : TYPE Numpy Array
            DESCRIPTION.Predicted Values
        y : TYPE Numpy Array
            DESCRIPTION. Actual Values

        Returns
        -------
        TYPE
            DESCRIPTION. Comparison of predicted and actual values
        '''
        return predictions == y