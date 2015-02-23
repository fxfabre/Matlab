#!/usr/bin/python3
# -*- coding: utf-8 -*-

from random import randrange


class ai_random:

    _available_moves = ['Left', 'right', 'up', 'down']

    def __init__(self):
        pass

    def move_next(self, grid):
        n = randrange(len(self._available_moves))
        direction = self._available_moves[n]
        return direction




