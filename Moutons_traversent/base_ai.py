#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

from constants import *


class BaseIA:

    def __init__(self):
        self.nb_iter = 0
        self.display = False
        self.nb_move1_lost = 0
        self.nb_move2_lost = 0
        self.nb_move3_lost = 0

    def move4(self, gameState):
        self.chooseMove()

    def chooseMove(self, gameState):
        raise NotImplementedError()

    def learn(self):
        return

    def recordState(self, gameState, s, a, s_prime, has_moved):
        return

    def save(self):
        return

    def load(self):
        return

    def get_headersParams(self):
        return ['nb_iter_learn', 'nb_iter_play', 'move1_lost', 'move2_lost', 'move3_lost']

    def getParameters(self):
        base_params = [self.nb_iter, NB_ITER_EVAL_PERF * NB_PLAY, self.nb_move1_lost, self.nb_move2_lost, self.nb_move3_lost]
        return list(map(str, base_params))

    def computeScore(self, play_game):

        # Test score AI
        scores = np.zeros(NB_ITER_EVAL_PERF, dtype=float)
        for n in range(NB_ITER_EVAL_PERF):
            nb_success = 0
            self.nb_move1_lost = 0
            self.nb_move2_lost = 0
            self.nb_move3_lost = 0

            for i in range(NB_PLAY):
                if play_game(self):
                    if DISPLAY:
                        print("Game success")
                    nb_success += 1
                elif DISPLAY:
                    print("Game lost")

            print("Nombre success : {0:>3} / {1:>4}, Moves lost : {2:>4}, {3:>3}, {4:>3}".format(
                nb_success, NB_PLAY,
                self.nb_move1_lost, self.nb_move2_lost, self.nb_move3_lost
            ))
            scores[n] = nb_success / NB_PLAY
            with open('moves_lost.csv', 'a') as f:
                to_save = [  nb_success / NB_PLAY,
                    self.nb_move1_lost, self.nb_move2_lost, self.nb_move3_lost ]
                f.write(';'.join( map(str, to_save) ) + "\n")

        mean = scores.mean()
        std = scores.std()
        print("Average score :", mean)
        print("Sigma :", std)

        with open('scores.csv', 'a') as f:
            #f.write(';'.join( self.get_headersParams() ) + "\n")
            f.write(';'.join( self.getParameters()     ) + "\n")
        return

    def display_strategy(self, play_game):
        # Display AI's decisions
        self.display = True
        play_game(self)


