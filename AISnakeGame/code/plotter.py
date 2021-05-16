import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('fivethirtyeight')

x_vals = []
y_vals = []

index = count()


def animate(i):
    try:
        scores_data = pd.read_excel('./data/scores.xlsx')
        mean_scores_data = pd.read_excel('./data/mean_scores.xlsx')
        scores = list(scores_data['scores'].values)
        mean_scores = list(mean_scores_data['mean_scores'].values)
    except:
        return
    mean_scores= [round(x,2) for x in mean_scores]
    plt.cla()
    
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Scores')
    plt.plot(mean_scores, label='Mean Scores')
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    
    plt.legend(loc='upper left')
    plt.tight_layout()


ani = FuncAnimation(plt.gcf(), animate, interval=1000)
plt.tight_layout()
plt.show()

