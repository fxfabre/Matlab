#!/usr/bin/python3
# -*- coding: utf-8 -*-


class BaseIA:

    def move4(self, gameState):
        self.chooseMove()

    def chooseMove(self, gameState):
        raise NotImplementedError()

    def learn(self):
        return

    def recordState(self, gameState, s, a, s_prime):
        return

    def save(self):
        return

    def load(self):
        return

