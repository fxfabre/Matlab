#!/usr/bin/python3

import numpy as np
from CustomException import *
from random import *


class Board:
    """ 2048 board game manager """


    _N = 0
    _board = []


    ##################################
    # Constructor                    #
    ##################################
    def __init__(self, n:int=4):
        self._N = n
        self.SetBoard([0 for i in range(n*n)])
        seed()


    ##################################
    # Public methods                 #
    ##################################
    def CanMove(self) -> bool:

        if self._HasEmptyBox():
            return True
        return self._CanMergeNumber()

    def SetBoard(self, l:list):
        self._board = l

    def MoveLeft(self):
        for i in range(self._N):
            line = self._GetLine(i)
            self._ShiftLeft(line)
            self._SetLine(line, i)

    def MoveRight(self):
        for i in range(self._N):
            col = self._GetLine(i)
            col.reverse()
            self._ShiftLeft(col).reverse()
            self._SetLine(col, i)

    def MoveUp(self):
        for i in range(self._N):
            col = self._GetColumn(i)
            self._ShiftLeft(col)
            self._SetColumn(col, i)

    def MoveDown(self):
        for i in range(self._N):
            col = self._GetColumn(i)
            col.reverse()
            self._ShiftLeft(col).reverse()
            self._SetColumn(col, i)


    ##################################
    # Player functions : AI & Random #
    ##################################
    def InsertRandomNumber(self):
        """ Add a new value, 2 or 4 in an empty box """
        if self._HasEmptyBox() is False:
            raise EndGameException("End game")

        # generate new number, 2 or 4
        generated = self._GenerateNumber()

        # Find empty box
        idx = self._FindZeroIndex()

        # Set random value
        self._SetValue(idx, generated)
        return


    ##################################
    # Private functions              #
    ##################################
    @staticmethod
    def _GenerateNumber() -> int:
        """ Generate at random a new number, 2 or 4 """
        number = 4
        if random() < 0.9:
            number = 2
        return number

    def _HasEmptyBox(self) -> bool:
        """ Look for empty box """
        for i in filter(lambda x: x==0, self._board):
            return True
        return False

    def _CanMergeNumber(self) -> bool:
        """ Look for consecutive equals values """
        n = self._N
        for i in range(n):
            for j in range(n-1):
                if (self._board[i*n+j] == self._board[i*n+j+1]):
                    return True
        for i in range(n-1):
            for j in range(n):
                if (self._board[i*n+j] == self._board[i*n+j+n]):
                    return True
        return False

    def _FindZeroIndex(self) -> int:
        indexes = []
        for i in range(len(self._board)):
            if i == 0:
                indexes.append(i)

        # indexes is the list of empty box
        rand = randrange(len(self._board))
        return indexes[rand]

    def _SetValue(self, index:int, value:int):
        assert (index >= 0), "index must be positive, got " + index
        assert (index < len(self._board)), "Index out of bounds : " + index
        assert (self._board[index] != 0), "Expecting 0 value in position " + index
        self._board[index] = value

    def _ShiftLeft(self, l:list) -> list:
        assert (len(self._board) == self._N * self._N), "Wrong board size, N = {0}, size = {1}".format(self._N, len(self._board))
        n = len(l)

        # First, remove zeros
        idx = 0
        for j in range(n):
            if l[j] != 0:
                l[idx] = l[j]
                idx += 1
        while idx < n:
            l[idx] = 0
            idx += 1

        # Then merge cells
        idx = 0
        j = 0
        while j < n-1:
            if l[j] == 0:
                j += 1
            elif l[j] == l[j+1]:
                l[idx] = l[j]*2
                l[j+1] = 0
                idx += 1
                j += 2
            else:
                l[idx] = l[j]
                idx += 1
                j += 1

        if l[n-1] != 0:
            l[idx] = l[n-1]
            idx += 1
        while idx < n:
            l[idx] = 0
            idx += 1
        return l

    def _GetLine(self, number:int) -> list:
        assert (number < self._N), "Unable to get line {0}, max = {1}".format(number, self._N -1)
        begin = number * self._N
        end = begin + self._N
        return self._board[begin:end]

    def _GetColumn(self, number:int) -> list:
        assert (number < self._N), "Unable to get column {0}, max = {1}".format(number, self._N -1)
        i1 = number
        i2 = i1 + self._N
        i3 = i2 + self._N
        i4 = i3 + self._N
        return [self._board[i1], self._board[i2], self._board[i3], self._board[i4]]

    def _SetLine(self, newLine:list, number:int):
        assert(number < self._N), "Unable to set line {0}, max = {1}".format(number, self._N -1)
        assert(len(newLine) == self._N), "Wrong length of line to set : " + str(newLine)
        begin = number * self._N
        end = begin + self._N
        self._board[begin:end] = newLine

    def _SetColumn(self, newCol:list, number:int):
        assert(number < self._N), "Unable to set column {0}, max = {1}".format(number, self._N -1)
        assert(len(newCol) == self._N), "Wrong length of line to set : " + str(newCol)
        i1 = number
        i2 = i1 + self._N
        i3 = i2 + self._N
        i4 = i3 + self._N
        [self._board[i1], self._board[i2], self._board[i3], self._board[i4]] = newCol


    ##################################
    # Override                       #
    ##################################
    def __str__(self):
        return str(self._board)

    def __repr__(self):
        return repr(self._board)





