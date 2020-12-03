import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

"""
Author: David Gray
Description: Data Mining Project - Implementation of the TSNE algorithm to analyize clusterability of data.

##########################################################################################################################

        !!!   WARNING: THIS PROGRAM TAKES A SIGNIFICANT AMOUNT OF PROCESSING POWER AND MAY TAKE UP TO AN      !!!
        !!!   HOUR TO RUN. THE RESULTS HAVE ALREADY BEEN SAVED AND ARE LOCATED AT "/project/images/tsne.PNG"  !!!

##########################################################################################################################
"""

def preprocess(fp):
    print("------preprocessing data")
    useful_columns = ['Hospital_code', 'Hospital_type_code', 'City_Code_Hospital', 'Hospital_region_code', 'Department', 'Ward_Type', 'Ward_Facility_Code', 'Bed Grade', 'City_Code_Patient', 'Type of Admission', 'Severity of Illness']

    # Download the data set
    df = pd.read_csv(fp+"/data.csv", index_col='case_id')

    # Drop missing data
    df = df.dropna()
    # Keep important columns
    X = df.loc[:,useful_columns]
    y = df.loc[:,'Stay']
    # Binarize variables
    X = pd.get_dummies(X)
    # Standardize data
    X = StandardScaler().fit_transform(X)

    return X, y

def reduce2D(X):
    print("------reducing dimensions")
    steps = 1000
    rs = 1
    perp = 30
    tsne = TSNE(n_components=2, perplexity=perp, n_iter=steps, random_state=rs, verbose=True)
    components = tsne.fit_transform(X)

def plot(fp, X, y):
    colors={'0-10':'red','11-20':'orange','21-30':'yellow','31-40':'green','41-50':'blue','51-60':'purple','61-70':'pink','71-80':'grey','81-90':'teal','91-100':'sienna','More than 100 Days':'midnightblue'}
    print("------plotting data")
    plt.figure(figsize=(13, 13))
    plt.scatter(X, y, c=y.map(colors))
    plt.title(str(col)+" vs Stay")
    plt.xlabel(str(col))
    plt.ylabel("Patient Stay Time (Days)")
    plt.savefig(fp+"/images/tsne.PNG")
    plt.clf()

def full_package(fp):
    X, y = preprocess(fp)
    X = reduce2D(X)
    plot(fp, X, y)

filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
full_package(filepath)