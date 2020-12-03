import tools
import os

"""
Author: David Gray
Description: Train the knn algorithm.
"""

filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tools.train_knn(filepath)

"""
##########################################################################################################################

    !!!   WARNING: THIS PROGRAM TAKES A SIGNIFICANT AMOUNT OF PROCESSING POWER AND MAY TAKE UP TO AN HOUR TO    !!!
    !!!   RUN. THE RESULTS HAVE ALREADY BEEN SAVED AND ARE LOCATED AT "/535_project/premade_examples/knn.pkl"   !!!

##########################################################################################################################
"""