#!/usr/bin/python3
# -*- coding: utf-8 -*-

from gameState import GameState
from constants import *

import random
import time

from ai_random import Ai_random
from ai_uniform import Ai_uniform
from ai_TD_learning import Ai_td_learning


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
    # AI learn
    ai = Ai_td_learning()

    for i in range(REPEAT):
        print("Iteration ", i)
        time.sleep(10)

        ai.learn()

        ai.computeScore(play_game)
        print("End iteration ", i)

#    ai.display_strategy(play_game)



if __name__ == '__main__':
    main()
