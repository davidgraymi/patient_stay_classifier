import tools
import os

"""
Author: David Gray
Description: Train the mlp algorithm.
"""

filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tools.train_mlp(filepath)