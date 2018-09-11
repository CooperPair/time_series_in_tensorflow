"""
Module for visualizing the dataset that is going to be used using graph.
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sys

data = pd.read_csv(sys.argv[1])

data = data['Adj Close']

n = data.min()
m = data.max()
data.plot(figsize=(10,6), title = 'Close Price', fontsize=14)

plt.xlim(1,2200)
plt.ylim(n,m)
plt.xlabel('Number of data',fontsize=20)
plt.ylabel('value',fontsize=20)
plt.title("Datasets Visualizatoin")
plt.grid(True)
plt.show()