#!/usr/bin/python3
# -*- coding: utf-8 -*-

from AI.Models.simulationNode import SimulationNode
import random
import AI.GameGridLight as GGL
import multiprocessing
import time


NB_PROCESS = 4
AVAILABLE_MOVES = ['down', 'left', 'right', 'up']
NB_SIMULATIONS_MC = 1000 # Number of random simulations
DEEP_SIMULATIONS  = 6    # Deep in the tree for MC simulations


class ai_parallelMC:

    def __init__(self):
        self.pool = multiprocessing.Pool(processes=NB_PROCESS)

    def move_next(self, gameBoard, gridHistory, scoreHistory):
        if gameBoard.grid.isGameOver:
            return ''

        grid = gameBoard.grid.toIntMatrix()
        params = [ [direction, grid] for direction in AVAILABLE_MOVES]

        scores = self.pool.map(runSimulation, params)
        print(scores)
        return getBestMove(scores)


def runSimulation(params):
    firstDirection = params[0]
    gridMatrix = params[1]

    scores = []

    grid = GGL.gameGridLight(matrix=gridMatrix)

    for i in range(NB_SIMULATIONS_MC):
        grid_MC = grid.clone()

        deep = 0
        score = 0
        while deep < DEEP_SIMULATIONS:
            if deep == 0:
                direction = firstDirection
            else:
                direction = random.choice( AVAILABLE_MOVES )
            current_score, have_moved = grid_MC.moveTo(direction)
            score += current_score

            if have_moved:
                deep += 1
                grid_MC.add_random_tile()
            else:
                deep = DEEP_SIMULATIONS + 1 # end

        scores.append(score)
    return getScore(scores)


def getScore(scores):
    return max(*scores)

def getBestMove(scores):
    best_score = -1
    best_move  = -1

    for i in range(len(scores)):
#        print(i, scores[i])
        if scores[i] > best_score:
            best_score = scores[i]
            best_move  = AVAILABLE_MOVES[i]

    print(best_move, best_score)
    return best_move

