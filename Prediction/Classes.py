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
    def __init__(self, n_inputs, n_neurons):
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
    # Calculate Predictions for Outputs
    #--------------------------------------------------------------------------    
    def Predictions(self, outputs):
        return (outputs > 0.5) * 1