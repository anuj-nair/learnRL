import matplotlib.pyplot as plt
import numpy as np


class plot:
    def __init__(self):
        plt.ion()  # Note this correction
        fig,self.ax = plt.subplots(1, 1, figsize=(15, 8))
        # plt.axis([0,1000,0,1])
        scores = list()
        mean_scores = list()

    def update(self, scores, mean_scores):
        mean_scores = [round(x, 2) for x in mean_scores]
        x = [i+1 for i in range(len(scores))]
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        self.ax.plot(x, scores)
        self.ax.plot(x, mean_scores)
        plt.ylim(ymin=0)
        plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
        plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
        plt.pause(0.00001)  # Note this correction
        plt.show()
