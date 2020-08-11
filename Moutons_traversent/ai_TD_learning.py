#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import random
import time
import os

from gameState import GameState
from base_ai import BaseIA


ALPHA   = 0.5       # learning rate
GAMMA   = 0.8       # discount factor
EPSILON = 0.2
FILE_Q_VALUES = 'q_val.csv'

R_SUCCESS   = 10
R_MOVE      = 0
R_FAILED    = -100
R_MOVE_LOST = -10


class Ai_td_learning(BaseIA):

    def __init__(self):
        BaseIA.__init__(self)
        self.q_values = None

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
            print( game_state )
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
            action = available_actions[ idx_action ]

        if self.display:
            print("Current state :")
            print("  ", self.q_values[current_state_hash, :])
            print("  ", game_state)
            print("  Move :", action + 1)

        return action + 1

    def learn(self):
        self.display = False
        if os.path.exists(FILE_Q_VALUES):
            self.load()

        #for i in range(10):
        #    self._learn(500, 0.4, nb_pilones=i)
        self._learn(1000, 0.4)  # Mix exploration - exploitation
        self._learn(1000, 0.2)  # Mix exploration - exploitation

        self.save()

    def _learn(self, nb_loop, epsilon=0.2, nb_pilones=-1):
        q_old = np.array( self.q_values )
        while nb_loop > 0:
            nb_loop -= 1
            self.nb_iter += 1

            if nb_pilones > 0:
                game_state = GameState(self, nb_pilones=nb_pilones)
            else:
                game_state = GameState(self)

            while game_state.canMove():
                a = self.chooseMove(game_state, epsilon)

                # Move and call recordState() to update Q_values
                game_state.move(a)

        # Watch convergence of q_values
        diff = (self.q_values - q_old).__abs__().sum()
        print(nb_pilones, " pilones. diff =", str(diff))

        return

    def recordState(self, game_state, s, a, s_prime, has_moved):

        if s == s_prime:
            # Move lost
            if a == 1:
                self.nb_move1_lost += 1
            if a == 2:
                self.nb_move2_lost += 1
            if a == 3:
                self.nb_move3_lost += 1
            if self.display:
                if a == 1:
                    print("Move 1 lost")
                if a == 2:
                    print("Move 2 lost")
                if a == 3:
                    print("Move 3 lost")

        if game_state.isSuccess():
            r = R_SUCCESS * game_state.nb_piliers
        elif game_state.canMove():
            r = R_MOVE
        else:
            r = R_FAILED

        if not has_moved:
            r += R_MOVE_LOST

        max_q_prime = self.q_values[s_prime, :].max()
        self.q_values[s, a-1] += ALPHA * (r + GAMMA * max_q_prime - self.q_values[s, a-1])
        return

    def save(self):
        with open(FILE_Q_VALUES, 'w') as f:
            f.write(str(self.nb_iter) + '\n')
            for i in range(self.q_values.shape[0]):
                for j in range(self.q_values.shape[1]):
                    f.write(str(self.q_values[i,j]) + '\n')
        #self.q_values.tofile(FILE_Q_VALUES, ',')

    def load(self):
        if not (self.q_values is None):
            return

        self.q_values = np.zeros([GameState.nb_states(), 3]) # 11.979 paramètres, 3 actions
        self.nb_iter = 0

        with open(FILE_Q_VALUES, 'r') as f:
            self.nb_iter = float(f.readline().strip())
            for i in range(self.q_values.shape[0]):
                for j in range(self.q_values.shape[1]):
                    self.q_values[i,j] = float(f.readline().strip())

    def get_headersParams(self):
        parent_params = BaseIA.get_headersParams(self)
        self_params   = ['R_SUCCESS * nb_pilones', 'R_MOVE', 'R_FAILED']
        parent_params.extend( self_params )
        return list(map(str, parent_params))

    def getParameters(self):
        parent_params = BaseIA.getParameters(self)
        self_params   = [R_SUCCESS, R_MOVE, R_FAILED]
        parent_params.extend( self_params )
        return list(map(str, parent_params))

