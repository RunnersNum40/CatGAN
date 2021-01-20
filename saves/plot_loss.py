import seaborn as sns
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import pandas as pd
import re

# Save each line from training_log.txt which contains the loss each generation
with open("training_log.txt") as file:
    data = [line for line in file]

#Capture the four numbers in each line
data = [re.match(r"\D+ ([\d\.]+) \D+ ([\d\.]+) \D+ ([\d\.]+) \D+ ([\d\.]+)", x).groups() for x in data]

# Convert all the numbers to floats
for n, x in enumerate(data):
    try:
        data[n] = tuple(map(float, x))
    except:
        # If the conversion fails print what line
        print(n, x)

# Creata a Data Frame with each column of data
step, real, fake, gan = zip(*data) # zip can convert between row and col
df = {"real":real, "fake":fake, "GAN":gan}
df = pd.DataFrame(df, columns=df.keys())

sns.set_style()

sns.lineplot(data=df)

plt.show()

