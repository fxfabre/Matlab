#!/usr/bin/python3
# -*- coding: utf-8 -*-

from base_ai import BaseIA
import random

class Ai_random(BaseIA):

    def __init__(self):
        pass

    def chooseMove(self, gameState):
        return random.randint(1,3)

    def learn(self):
        return

    def recordState(self, gameState, s, a, s_prime):
        return


