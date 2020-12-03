import tools
import os

"""
Author: David Gray
Description: Test the mlp algorithm.
"""

filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tools.test_mlp(filepath)