#------------------------------------------------------------------------------
# Importing Necessary libraires  
#------------------------------------------------------------------------------
import numpy as np
import pickle
from copy import deepcopy
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Importing Classes from the file 'Classes'
#------------------------------------------------------------------------------
from Classes import Input_Layer

#==============================================================================
# Class Model
#==============================================================================
class Model:
    
    def __init__(self):
        # Create a List of Network Objects
        self.layers = []
        
    #--------------------------------------------------------------------------
    # Add Layers to the Model
    #--------------------------------------------------------------------------
    def Add(self, layer):
        '''
        Parameters
        ----------
        layer : DESCRIPTION. Layers can be added in the network due to this function

        Returns
        -------
        None.
        '''
        self.layers.append(layer)
        
    #--------------------------------------------------------------------------
    # Set Loss, Optimizer and Accuracy
    #--------------------------------------------------------------------------
    def Set(self, *, loss, optimizer, accuracy):
        '''
        Parameters
        ----------
        * : DESCRIPTION. It notes that the subsequent parameters (loss and optimizer) are keywor arguments
        loss : DESCRIPTION. Setting Loss FUnction created earlier in 'Classes'
        optimizer : DESCRIPTION.Setting Optimizer Function, In our case is Gradient Descent
        accuracy : DESCRIPTION. Setting Accuracy Function

        Returns
        -------
        None.
        '''
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
        
    #--------------------------------------------------------------------------
    # Finalize the Model
    #--------------------------------------------------------------------------
    def Finalize(self):
        # Create and ser the Input Layer
        self.input_layer = Input_Layer()
        
        # # Count all the layers in the network
        layer_count = len(self.layers)
        
        # Initialize a list containing all trainable layers
        self.trainable_layers = []
        
        # Iterate the layers
        for i in range(layer_count):
            
            # If it is the first layer, then the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
                
            # Consider all layers except for the first and the last layer 
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
                
            # After last layer, the next object is the loss. Also save this object which is the output of the model
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
                
            # If layer contains an attribute 'weights', then its a trainable layer, add it to the list of trainabe layers
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
                
            # Update the loss object with trainable layers
            self.loss.Trainable_Layers(self.trainable_layers)
            
    #--------------------------------------------------------------------------
    # Get Weights of Each Layer, Append them and make a deepcopy
    #--------------------------------------------------------------------------
    def GetWeights(self):
        '''
        Returns
        -------
        TYPE List Object
            DESCRIPTION. Make a deepcopy of the weights and return the weights
        '''
        self.Weights = []
        
        # Iterate trainable layers and get their parameters
        for layer in self.trainable_layers:
            self.weight, self.biases = layer.Get_Params()
            self.Weights.append(self.weight)
            
        return deepcopy(self.Weights)

    #--------------------------------------------------------------------------
    # Tweak the Weights 
    #--------------------------------------------------------------------------
    def SetWeights(self, newWts):
        '''
        Parameters
        ----------
        newWts : TYPE List Object
            DESCRIPTION. Set the weights to a certrain value

        Returns
        -------
        None.
        '''

        for i in range(len(self.Weights)):
            self.Weights[i] = newWts[i]


    #--------------------------------------------------------------------------
    # Add a small Value to the Weights
    #--------------------------------------------------------------------------
    def AddEpsilon(self, wts, epsilonArr):
        '''
        Parameters
        ----------
        wts : TYPE Numpy Array
        epsilonArr : TYPE Numpy Array
            DESCRIPTION. This function adds a small value to the weights for the gradient checking

        Returns
        -------
        None.
        '''

        newWts = []
        for i in range(len(wts)):
            # Add a small value
            updatedWts = wts[i] + epsilonArr[i] 
            newWts.append(updatedWts)
        return newWts
    
    
    #--------------------------------------------------------------------------
    # Gradient Checking
    #--------------------------------------------------------------------------
    def Gradient_Check(self, X, y):
        '''
        Parameters
        ----------
        X : TYPE Numpy Array
            DESCRIPTION. Training Features
        y : TYPE Numpy Array
            DESCRIPTION. Actual Labels

        Returns
        -------
        None.
        '''
        # The small value that will be added to the weights
        epsilon = 1e-4                              
        
        # Save the original weights
        preserveWts = self.GetWeights() 
            
        epsilonArr = []
        epsilonGrads = []
        
        #Initialize Epsilon Array and its Gradient to zereos of shape 'preserved weights'
        for wts in preserveWts:
            epsilonArr.append(np.zeros(wts.shape))
            epsilonGrads.append(np.zeros(wts.shape))
        
        # For each individual wieghts in each layer of preserved weights, add and subtract epsilon value from the weights
        for i, wt in enumerate(preserveWts):
            
            # Rows and Columns of the trainable layers
            R, C = wt.shape
            
            # Iterate throgh rows 
            for r in range(R):
                
                # Iterate through columns
                for c in range(C):
                    
                    # For each weight in a trainable layer add epsilon to it
                    epsilonArr[i][r, c] = epsilon
                    newWts = self.AddEpsilon(preserveWts, epsilonArr)
                    
                    # Set the New Weights
                    self.SetWeights(newWts)
                    
                    #Perform Forward Pass and Claculate Loss
                    output = self.Forward(X, training=False)
                    J1 = self.loss.Calculate(output, y)
                    
                    # For the same weight in the layer subtract epsilon from it
                    epsilonArr[i][r, c] = -epsilon
                    newWts = self.AddEpsilon(preserveWts, epsilonArr)
                    
                    # Set the New Weights
                    self.SetWeights(newWts)
                    
                    # Perform Forward Pass and Calculate Loss
                    output = self.Forward(X, training=False)
                    J2 = self.loss.Calculate(output, y)
                    
                    # Get the gradient using gradient checking
                    epsilonGrads[i][r, c] = (J1-J2)/(2*epsilon)
                    
        # Set the original weights
        self.SetWeights(preserveWts)
        
        # Return gradients
        return epsilonGrads 

    #--------------------------------------------------------------------------
    # Train the Model
    #--------------------------------------------------------------------------
    def Train(self, X, y, *, validation_data=None, epochs=1, print_every=1):
        '''
        Parameters
        ----------
        X : TYPE Numpy Array
            DESCRIPTION. Training Features
        y : TYPE Numpy Array
            DESCRIPTION. Training Labels
        * : DESCRIPTION.
        validation_data : TYPE Numpy Arrays, optional
            DESCRIPTION. Validation Dataset. The default is None. 
        epochs : TYPE Int, optional
            DESCRIPTION. Number of times the training should be conducted. The default is 1.
        print_every : TYPE Int, optional
            DESCRIPTION. Loss, accuracy, etc must be printed after every described number. The default is 1.

        Returns
        -------
        None
        '''        
        # Initialize Accuracy object
        self.accuracy.init(y)
        
        trainCost = []               # Train Cost Array for cost from each iteration 
        valCost = []                 # Validation Cost Array for cost from each iteration
        trainAccuracy = []           # Training Accuracy Array
        valAccuracy = []             # Validation Accuracy Array
        
        # Training Loop
        for epoch in range (1, epochs+1):
            # Perform Forward Propagation
            output = self.Forward(X, training=True)
            
            # Calculate the Losses (data and regularization losses)
            train_data_loss, Regularization_Loss = self.loss.Calculate(output, y, include_lamda=True)
            
            # Append training loss 
            trainCost.append(train_data_loss)
            
            # Calculate total loss
            loss = train_data_loss + Regularization_Loss
            
            # Get Predictions and Calculate Accuracy
            predictions = self.output_layer_activation.Predictions(output)
            accuracy = self.accuracy.Calculate(predictions, y)
            trainAccuracy.append(accuracy)
            
            # Perform Back Propagation
            self.Backward(output, y)
            
            # Optimize and update the parameters
            self.optimizer.Pre_Update_Params()
            for layer in self.trainable_layers:
                self.optimizer.Update_Params(layer)
            self.optimizer.Post_Update_Params()
            
            # Print Summary of different parameters during training
            if not epoch % print_every:
                print(f'Epoch: {epoch}, ' +
                      f'Accuracy: {accuracy:.3f}, ' +  
                      f'Total Loss: {loss:.3f}, ' +
                      f'Data Loss: {train_data_loss:.3f}, ' +
                      f'Regularization Loss: {Regularization_Loss:.3f}')
                
            
            # Perform gradient for first 2 iterations
            if epoch < 2:
                gradCheck = self.Gradient_Check(X, y)
                for j in range(len(gradCheck)):
                    print("Gradient Check and Back Prop Difference for ", j, " Iteration.")
                    print("____________________", j , "____________________")
                    diff = gradCheck[j] - layer.d_inputs[j]
                    print(diff) # Print difference
        
            
            # If there is any Validation Data provided
            if validation_data is not None:
                # Recognize features and lables
                x_val, y_val = validation_data
                
                # Perform Forward Propagation without training
                output = self.Forward(x_val, training=False)
                
                # Calculate data loss
                val_data_loss = self.loss.Calculate(output, y_val)
                
                # Get Predictions and Determine Accuracy
                predictions = self.output_layer_activation.Predictions(output)
                val_accuracy = self.accuracy.Calculate(predictions, y_val)
                valAccuracy.append(val_accuracy)
                
                # Append the Validation Cost determined in every loop
                valCost.append(val_data_loss)
                
                # Print Validation Data Summary
                if not epoch % print_every:
                    print(f'Validation: ' +
                          f'Accuracy: {val_accuracy:.3f}, ' + 
                          f'loss: {val_data_loss:.3f}')
            
        # If Validation Cost is not none, then plot training and validation costs 
        if valCost != 0:
            plt.figure()
            plt.plot(trainCost)
            plt.plot(valCost)
            plt.title("Training and Validation Costs")
            plt.xlabel('Iterations')
            plt.ylabel('Cost Error')
            plt.legend(["Training Cost", "Validation Error"], loc ="upper right")
            plt.show()
            
            plt.figure()
            plt.plot(trainAccuracy)
            plt.plot(valAccuracy)
            plt.title("Training and Validation Accuracies")
            plt.xlabel('Iterations')
            plt.ylabel('Accuracy')
            plt.legend(["Training", "Validation"], loc ="upper right")
            plt.show()
            
        # Else, only print Training Cost
        else:
            plt.figure()
            plt.plot(trainCost)
            plt.title("Training Cost")
            plt.xlabel('Iterations')
            plt.ylabel('Cost Error')
            plt.show()
            
            plt.figure()
            plt.plot(trainAccuracy)
            plt.title("Training Accuracy")
            plt.xlabel('Iterations')
            plt.ylabel('Accuracy')
            plt.show()

    #--------------------------------------------------------------------------
    # Perform Forward Propagation
    #--------------------------------------------------------------------------        
    def Forward(self, X, training):
        '''
        Parameters
        ----------
        X : TYPE Numpy Array
            DESCRIPTION. Data set features
        training : TYPE Bool
            DESCRIPTION. Checks wether which dataset to train on and on which to perform forward pass only

        Returns
        -------
        layer.output: TYPE Numpy Array List
            DESCRIPTION. Performs forward propagation layer wise and returns list
        '''
        # Call Forward propagation, this will set the O/P property that the first layer in previos object is expecting
        self.input_layer.Forward(X, training)
        
        # Call Forward method in chain. Pass O/P of the previous layer as a parameter 
        for layer in self.layers:
            layer.Forward(layer.prev.output, training)
        return layer.output
    
    #--------------------------------------------------------------------------
    # Perform Back Propagation
    #--------------------------------------------------------------------------
    def Backward(self, output, y):
        '''
        Parameters
        ----------
        output : TYPE Numpy Array
            DESCRIPTION. Predicted Output
        y : TYPE Numpy Array
            DESCRIPTION. Actual Labels

        Returns
        -------
        None.
        '''
        # Call backprop on loss which is the last object in the model
        self.loss.Backward(output, y)
        
        # Call backprop in reverse order using chain rule passing d_inputs as a parameter 
        for layer in reversed(self.layers):
            layer.Backward(layer.next.d_inputs)
                
            
    #--------------------------------------------------------------------------
    # Test the trained Model with optimized parameters on an unseen data set
    #--------------------------------------------------------------------------
    def Test(self, test_data=None):
        '''
        Parameters
        ----------
        test_data : TYPE Numpy Arrays, optional
            DESCRIPTION. Test data set with features and labels. The default is None.

        Returns
        -------
        None
        '''
        # IF Test Dataset is provided
        if test_data is not None:
            # Recognize Features and Labels
            x_test, y_test = test_data
            
            # Perform forward prop without training
            output = self.Forward(x_test, training=False)
            
            # Calculate Loss
            test_data_loss = self.loss.Calculate(output, y_test)
            
            # Get Predictions and calculate Accuracy
            predictions = self.output_layer_activation.Predictions(output)
            accuracy = self.accuracy.Calculate(predictions, y_test)
            
            # Print Test Summary
            print(f'Test: ' +
                  f'Accuracy: {accuracy:.3f}, ' + 
                  f'loss: {test_data_loss:.3f}') 
            
            
    #--------------------------------------------------------------------------
    # Retrieves and returns parameters of trainable layers
    #--------------------------------------------------------------------------          
    def Get_Params(self):
        '''
        Returns
        -------
        params : TYPE List Numpy Array
            DESCRIPTION. Optimal weights and biases after training are fetched to be saved for future predictions
                         Parameters are saved layer wise
        '''
        # Create a list for parameters
        params = []
        
        # Iterate trainable layers and get their parameters
        for layer in self.trainable_layers:
            params.append(layer.Get_Params())
            
        # Return list of optimized parameters
        return params
    
    
    #--------------------------------------------------------------------------
    # Saves the Optimized Parameters to a file
    #--------------------------------------------------------------------------
    def Save_Params(self, path):
        '''
        Parameters
        ----------
        path : TYPE file. Binary Write Mode.
            DESCRIPTION. Parametrs are dumped into this file.

        Returns
        -------
        None.
        '''
        # Open a fie in the binary write mode and save parameters
        with open(path, 'wb') as f:
            pickle.dump(self.Get_Params(), f)