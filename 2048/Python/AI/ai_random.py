#!/usr/bin/python3
# -*- coding: utf-8 -*-

from random import randrange
import sys

from time import sleep


class ai_random:

    def __init__(self):
        self._available_moves = ['left', 'right', 'up', 'down']
        file = open("LogPython.log", 'w')
        self.logFile = file

    def move_next(self, grid, gridHistory, scoreHistory):
        if grid.isGameOver:
            return ''
        n = randrange(len(self._available_moves))
        direction = self._available_moves[n]

        sleep(0.01)
        return direction

    def __del__(self):
        self.logFile.close()
        self.logFile = sys.stdout

