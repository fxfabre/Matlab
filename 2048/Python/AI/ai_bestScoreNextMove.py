#!/usr/bin/python3
# -*- coding: utf-8 -*-

from random import randrange
import sys
import copy
import tkinter as TK
import AI.GameGridLight as GGL


class ai_bestScoreNextMove:

    def __init__(self):
        self._available_moves = ['down', 'left', 'right', 'up']
        self._actions = {
            "left" : lambda b: b.move_tiles_left,
            "right": lambda b: b.move_tiles_right,
            "up"   : lambda b: b.move_tiles_up,
            "down" : lambda b: b.move_tiles_down
        }

    def move_next(self, gameBoard, gridHistory, scoreHistory):
        if gameBoard.grid.isGameOver:
            return ''

        print("\noriginal :")
        print(gameBoard)
        gridMatrix = gameBoard.grid.toIntMatrix()

        best_score = -1
        best_move=''
        for direction in self._available_moves:
            gameGrid = GGL.gameGridLight(0, 0, gridMatrix)
            current_score = gameGrid.moveTo(direction)
            print("Move done")

            print("Action {0:<5} => {1} pts".format(direction, current_score))
            if current_score > best_score:
                best_score = current_score
                best_move = direction

        return best_move
