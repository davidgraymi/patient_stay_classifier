import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

"""
Author: David Gray
Description: Preprocesses data
Note: Run this program before training an algorithm.
"""

def preprocess(fp):
    print("------preprocessing data")
    useful_columns = ['Hospital_code', 'Hospital_type_code', 'City_Code_Hospital', 'Hospital_region_code', 'Available Extra Rooms in Hospital', 'Department', 'Ward_Type', 'Ward_Facility_Code', 'Bed Grade', 'City_Code_Patient', 'Type of Admission', 'Severity of Illness', 'Visitors with Patient', 'Age', 'Admission_Deposit']

    # Download the data set
    df = pd.read_csv(fp+"/data.csv", index_col='case_id')

    # Drop missing data
    df = df.dropna()
    # Keep important columns
    X = df.loc[:,useful_columns]
    y = df.loc[:,'Stay']
    # Binarize variables
    X = pd.get_dummies(X)

    # Split the data set into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1, test_size=0.2)

    # Standardize data
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Save preprocessed data
    processed_folder = "/processed_data"
    X_train = pd.DataFrame(X_train)
    X_train.to_csv(fp + processed_folder + "/X_train.csv", header=False, index=False)
    y_train = pd.DataFrame(y_train)
    y_train.to_csv(fp + processed_folder + "/y_train.csv", header=False, index=False)
    X_test = pd.DataFrame(X_test)
    X_test.to_csv(fp + processed_folder + "/X_test.csv", header=False, index=False)
    y_test = pd.DataFrame(y_test)
    y_test.to_csv(fp + processed_folder + "/y_test.csv", header=False, index=False)
    
filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
preprocess(filepath)