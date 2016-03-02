#!/usr/bin/python3
# -*- coding: utf-8 -*-

from gameState import GameState
from constants import DISPLAY

import numpy as np
import random
import math

from ai_random import Ai_random
from ai_uniform import Ai_uniform
from ai_TD_learning import Ai_td_learning


REPEAT  = 50
NB_PLAY = 1000

def play_game(ai):
    gameState = GameState(ai)
    if DISPLAY:
        print(gameState)
        print('')

    while gameState.canMove():
        value = random.randint(1, 6)
        gameState.move(value)

    return gameState.isSuccess()


def main():
    ai = Ai_td_learning()
    ai.learn()

    scores = np.zeros(REPEAT, dtype=float)
    for n in range(REPEAT):
        nb_success = 0
        for i in range(NB_PLAY):
            if DISPLAY:
                print('')
                print('')
                print("=== New Game :")

            if play_game(ai):
                if DISPLAY:
                    print("Game success")
                nb_success += 1
            elif DISPLAY:
                print("Game lost")

        print("Nombre success : {0} / {1}".format(nb_success, NB_PLAY))
        scores[n] = nb_success / NB_PLAY

    print("Average score :", scores.mean())
    print("Sigma :", scores.std())

if __name__ == '__main__':
    main()
