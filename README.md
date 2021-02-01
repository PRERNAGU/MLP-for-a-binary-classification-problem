# MLP-vs-SVM-a-comparative-study
Motivation of the problem For the purpose of our study, we investigate a specific dataset from the German SocioEconomic Panel, a large panel study, extremely popular in Social Sciences Research for a classification problem (1). We choose to critique  Multi-Layer Perceptron  in a real-world scenario, with more complicated and noisy data (as opposed to using a popular machine learning dataset). The dataset contains sociodemographic information, as well as rarer psychometric variables, selected based on their importance in past research on behavioural prediction (2,3). Using the variable feature space, we will contrast the two models on the binary classification problem of predicting smoking patterns.


• Load "MLPFinal.mat"
• Read csv file: NN5000.csv containing dataset
• Partition the data into stratified train and stratified test set (class imbalance), with 30% test hold-out dataset.
• On the 70% training data, perform stratified k-Fold cross validation for model selection.
• ****%%Hyper-parameter tuning for MLP with single Hidden Layer%%*****
• In order to evaluate the dispersion of Test F1 score, every hyper-parameter combination is evaluated on the 30% HoldOut test set
• N=No. of neurons in hidden layer
• learn-rate combinations
• two optimisation algorithms
• "trainbr"=Bayesian Regularization
• "traingd"=Gradient Descent
• Two activation Functions: Log-sig and Poslin
• weightreg corresponds to weight momentum
• Create a Fitting Network & set number of neurons
• set cross-entrophy loss function to minimise back propagation cross entropy error in each iteration
• Set maximum no. of epochs to 400
• activation f(n) for input layers
• setting output layer as soft-max for classification
• Confusion Matrix, F1 -score, misclassification erorr; Precision and Recall for every out of sample validation fold
• Calculating the mean F1 -score, misclassification erorr; Precision and Recall for all out-of sample validation scores to
get cross-validation metrics
• Time calculation post K-fold cross validation for the given hyperparameter combination
• Evaluate every-hyper parameter-combination on hold-out test data to study dispersion of test scores\
• Save the grid-search in a csv file
• ****%%Hyper-parameter tuning for MLP with two Hidden Layer%%*****
• In order to evaluate the dispersion of Test F1 score, every hyper-parameter combination is evaluated on the 30% HoldOut test set
• N=No. of neurons in 1
st hiddenlayer
• N1=No. of neurons in 2
nd hiddenlayer
• Steps from last section follow
• save the grid-search results for two-hidden layers
