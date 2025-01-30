>> In order to train the model, run the Training.py file.
>> The following line of code in training.py is used to load the features and labels from the dataset:
	
	data = pd.read_csv('heart.csv', encoding='unicode_escape')
	'Changing the name of the .csv file will change the dataset.'

>> If the dataset split needs to be changed then the following line of code in the same file should be changed using:
	
	X_train, Y_train, X_val, Y_val, X_test, Y_test = data.Split(0.7, 0.85)
	'0.7 represents 70% distribution to the train set. While 0.85 means that out of 100%, 15% each is distributed 
	amongst validation and test sets.' 
  
>> Finally, when the data is split into different sets the model is called, layers and activation functions are added, 
	and lambda value can be changed below as. The model is then set to particular functions and then finalized. 
	The user can change the above-mentioned parameters as;

	model.Add(Dense_Layer(11, 20, lamda=0.0003))	
	model.Add(ReLU_Activation())
	model.Add(Dense_Layer(20, 1))
	model.Add(Sigmoid_Activation())

>> Model is set with required functions and values of alpha and decay can be changed here.
 	model.Set(loss=BinaryCrossEntropy_Loss(), optimizer=GD_Optimizer(alpha=0.05, decay=0), accuracy=Categorical_Accuracy()) 

>> Model is trained using below code. Number of iterations to train the model and print summary after every desired 
	number can be set using.
	
	model.Train(X_train, Y_train, validation_data=(X_val, Y_val), epochs=1000, print_every=100)

>> The model is then tested on test set and the parameters are saved into a file. 