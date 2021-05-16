import os
import torch
import random
import pandas as pd
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model_2 import Linear_QNet, QTrainer
#from plotter_2 import plot


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate 1 - 0
        self.memory = deque(maxlen=MAX_MEMORY)  # poplef()
        self.model = Linear_QNet(11, 1024, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]

        # Point Block
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        point_r = Point(head.x + 20, head.y)
        point_l = Point(head.x - 20, head.y)

        # Point 2Block
#        point_2u = Point(head.x, head.y - 40)
#        point_2d = Point(head.x, head.y + 40)
#        point_2r = Point(head.x + 40, head.y)
#        point_2l = Point(head.x - 40, head.y)

        # Point 3Block
#        point_3u = Point(head.x, head.y - 60)
#        point_3d = Point(head.x, head.y + 60)
#        point_3r = Point(head.x + 60, head.y)
#        point_3l = Point(head.x - 60, head.y)

        # Direction Check
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        dir_r = game.direction == Direction.RIGHT
        dir_l = game.direction == Direction.LEFT

        state = [

            # danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),


            # danger 2straight
#            (dir_r and game.is_collision(point_2r)) or
#            (dir_l and game.is_collision(point_2l)) or
#            (dir_u and game.is_collision(point_2u)) or
#            (dir_d and game.is_collision(point_2d)),

            # danger 2right
#            (dir_u and game.is_collision(point_2r)) or
#            (dir_d and game.is_collision(point_2l)) or
#            (dir_l and game.is_collision(point_2u)) or
#            (dir_r and game.is_collision(point_2d)),

            # danger 2left
#            (dir_d and game.is_collision(point_2r)) or
#            (dir_u and game.is_collision(point_2l)) or
#            (dir_l and game.is_collision(point_2u)) or
#            (dir_r and game.is_collision(point_2d)),

            # danger 3straight
#            (dir_r and game.is_collision(point_3r)) or
#            (dir_l and game.is_collision(point_3l)) or
#            (dir_u and game.is_collision(point_3u)) or
#            (dir_d and game.is_collision(point_3d)),

            # danger 3right
#            (dir_u and game.is_collision(point_3r)) or
#            (dir_d and game.is_collision(point_3l)) or
#            (dir_l and game.is_collision(point_3u)) or
#            (dir_r and game.is_collision(point_3d)),

            # danger 3left
#            (dir_d and game.is_collision(point_3r)) or
#            (dir_u and game.is_collision(point_3l)) or
#            (dir_l and game.is_collision(point_3u)) or
#            (dir_r and game.is_collision(point_3d)),


            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array([state], dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # popleft if MAX_MEMORY is reached
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(
                self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: trade off exploration / exploration
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            state0 = torch.unsqueeze(state0,0)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():

    data_folder_path = './data'
    scores_filename = './data/scores.xlsx'
    mean_scores_filename = './data/mean_scores.xlsx'
    if os.path.exists(data_folder_path):
        if os.path.exists(scores_filename):
            os.remove(scores_filename)
        if os.path.exists(mean_scores_filename):
            os.remove(mean_scores_filename)
    else:
        os.makedirs(data_folder_path)
#    plot()
#    plt = plot()
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(
            state=state_old, action=final_move, reward=reward, next_state=state_new, done=done)

        # remember
        agent.remember(state=state_old, action=final_move,
                       reward=reward, next_state=state_new, done=done)

        if done:
            # train long memory
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            pd.DataFrame({'scores': plot_scores}).to_excel(
                scores_filename, index=False)
            pd.DataFrame({'mean_scores': plot_mean_scores}).to_excel(
                mean_scores_filename, index=False)


#            plt.update(scores=plot_scores, mean_scores=plot_mean_scores)


if __name__ == '__main__':
    train()
