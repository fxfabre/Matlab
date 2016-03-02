#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import random
import os

from gameState import GameState
from base_ai import BaseIA


ALPHA   = 0.5       # learning rate
GAMMA   = 0.8       # discount factor
EPSILON = 0.2
INF     = 999 * 1000 * 1000
FILE_Q_VALUES = 'q_val.csv'


class Ai_td_learning(BaseIA):

    def __init__(self):
        self.q_values = np.zeros([GameState.nb_states(), 3]) # 11.979 paramètres, 3 actions
        self.display = False

    def chooseMove(self, game_state: GameState, epsilon=0.0):
        """
        :param game_state: gameState object
        :param epsilon: valeur dans [0, 1]. 0 pour exploitation uniquement, 1 pour random move
        :return: 0 si pas de mouvement possible, 1, 2 ou 3 pour définir le mouvement à faire sinon
        """
        current_state_hash = game_state.__hash__()
        available_actions = []
        if game_state.nb_moutons_box1 > 0:
            available_actions.append(0)
        else:
            self.q_values[current_state_hash, 0] = -1000

        if game_state.nb_moutons_box2 > 0:
            available_actions.append(1)
        else:
            self.q_values[current_state_hash, 1] = -1000

        if game_state.nb_moutons_box3 > 0:
            available_actions.append(2)
        else:
            self.q_values[current_state_hash, 2] = -1000

        if len(available_actions) == 0:
            raise Exception("Game has ended. should not call chooseMove()")

        action = -1
        # Choose best score (exploitation)
        if random.random() > epsilon:
            q_values = self.q_values[current_state_hash, available_actions]
            idx_action = q_values.argmax()
            action = available_actions[ idx_action ]
        # Random move (exploration)
        else:
            idx_action = random.randint(0, len(available_actions) -1)
#            print( idx_action )
            action = available_actions[ idx_action ]
        return action + 1

    def learn(self):
        if os.path.exists(FILE_Q_VALUES):
            self.load()
        self._learn(2000, 1.0)  # Pure exploration. random moves
        for i in range(10):
            self._learn(2000, 0.4, nb_pilones=i)
        self._learn(10000, 0.2)  # Mix exploration - exploitation
        self.save()

    def _learn(self, nb_iter=1000, epsilon=0.2, nb_pilones=-1):
        while nb_iter > 0:
            nb_iter -= 1
            q_old = np.array( self.q_values )

            if nb_pilones > 0:
                game_state = GameState(self, nb_pilones=nb_pilones)
            else:
                game_state = GameState(self)

            r = 0
            while r >= 0:
                s = game_state.__hash__()

                a = self.chooseMove(game_state, epsilon)
                if a == 0:
                    print(game_state)
                    raise Exception("Choose move returns 0")

                # Move and call recordState() to update Q_values
                game_state.move(a)

            # Watch convergence of q_values
            diff = (self.q_values - q_old).__abs__().sum()
            print("Convergence des q_values. diff = " + str(diff))

        return


    def recordState(self, game_state, s, a, s_prime):

        if game_state.canMove():
            r = 0
        else:
            r = -100

        max_q_prime = self.q_values[s_prime, :].max()
        self.q_values[s, a-1] += ALPHA * (r + GAMMA * max_q_prime - self.q_values[s, a-1])

        return

    def save(self):
        with open(FILE_Q_VALUES, 'w') as f:
            for i in range(self.q_values.shape[0]):
                for j in range(self.q_values.shape[1]):
                    f.write(str(self.q_values[i,j]) + '\n')
        #self.q_values.tofile(FILE_Q_VALUES, ',')

    def load(self):
        #self.q_values = np.load(FILE_Q_VALUES, 'r', ',')
        with open(FILE_Q_VALUES, 'r') as f:
            for i in range(self.q_values.shape[0]):
                for j in range(self.q_values.shape[1]):
                    self.q_values[i,j] = float(f.readline().strip())



