#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from base_ai import BaseIA

DISPLAY = True


class Ai_uniform(BaseIA):

    def __init__(self):
        BaseIA.__init__(self)

    def move4(self, gameState):
        move = 0
        if gameState.nb_moutons_box1 > 0:
            if gameState.nb_moutons_box2 == 0:
                move = 1
            elif gameState.nb_moutons_box3 == 0:
                move = 2

        if move == 0:
            move = gameState.nb_moutons[0:2].argmin() + 1

        if DISPLAY:
            print("({0}, {1}, {2}) => {3}".format(
                gameState.nb_moutons_box1,
                gameState.nb_moutons_box2,
                gameState.nb_moutons_box3,
                move
            ))
        return move

    def chooseMove(self, gameState):
        if gameState.nb_moutons_box1 == 0:
            if gameState.nb_moutons_box2 == 0:
                return 3
            if gameState.nb_moutons_box2 < gameState.nb_moutons_box3:
                return 3
            return 2
        if gameState.nb_moutons_box2 == 0:
            return 1
        if gameState.nb_moutons_box3 == 0:
            return 2
        return gameState.nb_moutons[0:3].argmax() + 1

    def learn(self):
        return

    def recordState(self, gameState, s, a, s_prime):
        return
