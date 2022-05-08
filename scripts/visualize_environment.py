from cProfile import label
from turtle import color
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib
import numpy as np
from pulp import value
from scipy.special import erfinv

def visualize(xs, labels, objs, c=["r", "m", "g", "c", "k"]):
    fig, ax = plt.subplots()

    #create simple line plot
    ax.set(xlim=(-5, 15), ylim=(-5, 15))
    T = len(xs[0])
    #add rectangle to plot
    for t, obj in enumerate(objs):
        b = 1 - .5*(t/T)
        ax.add_patch(Rectangle((obj.x - obj.w/2, obj.y - obj.h/2), obj.w, obj.h, color=(0, 0, b)))
    
    for n, x in enumerate(xs):
        ax.scatter([i[0] for i in x], [i[1] for i in x], color=c[n], label=labels[n])
    #display plot
    ax.legend()
    plt.show()
