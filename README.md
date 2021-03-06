
Missouri State University: CSC 535/635 - Data Mining

Professor: Dr. Jamil Saquer

Final Project: Patient Length of Stay Classifier

Contributors: David Gray, Braden Bagby, & Binh Le

Data Source: https://www.kaggle.com/nehaprabhavalkar/av-healthcare-analytics-ii


Instructions

1. run preprocess.py

	- Running this will preprocess and save the data.
	- After running there will be files named X_test.csv, X_train.csv, y_test.csv,
	  and y_train.csv in the 'processed_data' folder.

2. run train_mlp.py or train_knn.py

	- Running this will train and save the corresponding algorithm.
	- After running there will be a file named mlp.pkl or knn.pkl in the 'models'
	  folder.

3. run test_mlp.py or test_knn.py

	- Running this will produce a confusion matrix and classification report with
	  the algorithms accuracy, precision, recall, and f1-score.


Project Folders & Files Overview

- /code: contains the source code
	- tools.py: full of tools for training, testing, and visualizing
	- preprocess.py: preprocesses data
	- train_knn.py: trains the knn algorithm
	- train_mlp.py: trains the mlp algorithm
	- test_knn.py: tests the knn algorithm
	- test_mlp.py: tests the mlp algorithm
	- tsne.py: applies tsne to the data and saves the visual 
- /premade_examples: contains images, trained algorithms, and preprocessed data
	- knn.pkl: pretrained knn algorithm
	- mlp.pkl pretrained mlp algorithm
	- tsne.PNG: example results from tsne.py
	- knn_cm.png: example results from test_knn.py
	- mlp_cm.png: example results from test_mlp.py
	- knn_scores.png: example results from test_knn.py
	- mlp_scores.png: example results from test_mlp.py
- data.csv: original, unmodified healthcare data set for training and testing


Dependencies

1. pandas









