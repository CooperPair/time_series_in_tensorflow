"""
Module for visualizing the dataset that is going to be used using graph.
"""

from datetime import datetime
from pandas import read_csv
import matplotlib.pyplot as plt
import sys
import pandas as pd

data = pd.read_csv(sys.argv[1])
# original data - closing price
data['Adj Close'].plot(figsize=(15,8), title = 'Close Price', fontsize=14)
plt.show()