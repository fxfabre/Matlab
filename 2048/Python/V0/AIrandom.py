#!/usr/bin/python3

from Board import *
from random import *
from MyEnum import MoveDirection


class AIrandom:

    _N = 0

    def __init__(self, size:int):
        self._N = size

    def GetNextMove(self, board:Board) -> int:
        assert (len(board) == self._N*self._N), "Wrong board size"

        return randrange(self._N)

