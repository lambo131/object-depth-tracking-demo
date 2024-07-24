# importing required modules
import random
from itertools import count
from random import randint

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


x_vals = []
y_vals = []
index = count()

def animate(i):
    x_vals.append(next(index))
    y_vals.append(random.randint(0,5))
    if(len(x_vals)>3):
        x_vals.pop(0)
        y_vals.pop(0)
        print(x_vals)
    plt.plot(x_vals,y_vals)
anim = animation.FuncAnimation(plt.gcf(), animate, interval=1000)
plt.tight_layout
plt.show()
