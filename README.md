# NetsPredict

Cross-validation predictions with the Elastic net, taking into account the structure of the data to organise the cross-validation folds (necessary e.g. when the data points are subjects and the subjects are family-related). It has also options to deal with riemannian spaces, and other things. 

The main function to use is nets_predict5 (1,2,3,4 correspond with previous versions with less functionalities and perhaps buggy). The documentation is within the function itself.

For training and testing in different data sets, use nets_train and nets_test. 
