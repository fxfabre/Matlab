#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from constants import DISPLAY, NB_MOUTONS, NB_PILIERS

class GameState:

    def __init__(self, ai, nb_pilones=NB_PILIERS):
        cdef np.ndarray self.nb_moutons = np.zeros(4, dtype=int)


        self.nb_moutons[0] = NB_MOUTONS
        cdef int self.nb_piliers = nb_pilones
        cdef np.ndarray self.movesLost = np.zeros(3)
        cdef np.ndarray self.totalMoves = np.zeros(3)
        self.ai = ai

        self.availables_moves = {
            1 : lambda : self.move1(),
            2 : lambda : self.move2(),
            3 : lambda : self.move3(),
            4 : lambda : self.move4(),
            5 : lambda : self.move5(),
            6 : lambda : self.move6()
        }

    def canMove(self):
        return (self.nb_moutons_box4 != NB_MOUTONS) and (self.nb_piliers > 0)

    def isGameLost(self):
        return (self.nb_piliers <= 0)

    def isSuccess(self):
        return (self.nb_moutons_box4 == NB_MOUTONS) and (self.nb_piliers > 0)

    #####################
    # Move functions
    #####################
    def move(self, action):
        if action == 0:
            print(self)
            raise Exception("Cannot move 0")
        if action < 4:
            s = self.__hash__() # current state
            has_moved = self.availables_moves[action]()
            s_prime = self.__hash__()
            self.ai.recordState(self, s, action, s_prime, has_moved)
            return has_moved
        return self.availables_moves[action]()

    def move1(self):
        if self.nb_moutons_box1 > 0:
            self.nb_moutons[0] -= 1
            self.nb_moutons[1] += 1
            if DISPLAY:
                print("Move 1 :", self)
            self.totalMoves[0] += 1
            return True
        if DISPLAY:
            print( "    Move 1 lost")
        self.movesLost[0] += 1
        return False

    def move2(self):
        if self.nb_moutons_box2 > 0:
            self.nb_moutons[1] -= 1
            self.nb_moutons[2] += 1
            if DISPLAY:
                print("Move 2 :", self)
            self.totalMoves[1] += 1
            return True
        if DISPLAY:
            print( "    Move 2 lost")
        self.movesLost[1] += 1
        return False

    def move3(self):
        if (self.nb_moutons_box3 > 0) and (self.nb_piliers > 0):
            self.nb_moutons[2] -= 1
            self.nb_moutons[3] += 1
            if DISPLAY:
                print("Move 3 :", self)
            self.totalMoves[2] += 1
            return True
        if DISPLAY:
            print( "    Move 3 lost")
        self.movesLost[2] += 1
        return False

    def move4(self):
        if DISPLAY:
            print("===> 4")
        a = self.ai.chooseMove(self)   # action : 1, 2 or 3
        return self.move(a)

    def move5(self):
        has_moved = self.move4()
        if self.canMove():
            return has_moved & self.move4()
        return has_moved

    def move6(self):
        if self.nb_piliers > 0:
            self.nb_piliers -= 1
        else:
            raise Exception("Cannot play, game has ended (no more pilones)")
        if DISPLAY:
            print("Move 6 :", self)
        return True

    #####################
    # Getters
    #####################
    @property
    def nb_moutons_box1(self):
        return self.nb_moutons[0]

    @property
    def nb_moutons_box2(self):
        return self.nb_moutons[1]

    @property
    def nb_moutons_box3(self):
        return self.nb_moutons[2]

    @property
    def nb_moutons_box4(self):
        return self.nb_moutons[3]

    #####################
    # override
    #####################
    def __hash__(self):
        hashValue = 0
        hashValue = (hashValue + self.nb_moutons_box1) * NB_MOUTONS
        hashValue = (hashValue + self.nb_moutons_box2) * NB_MOUTONS
        hashValue = (hashValue + self.nb_moutons_box3) * NB_MOUTONS
        return hashValue + self.nb_piliers

    def __str__(self):
        return "({0:>2d}, {1:>2d}, {2:>2d}, {3:>2d}), reste {4} pilones".format(
            self.nb_moutons_box1,
            self.nb_moutons_box2,
            self.nb_moutons_box3,
            self.nb_moutons_box4,
            self.nb_piliers
        )

    #####################
    # Static
    #####################
    @staticmethod
    def nb_states():
        return NB_MOUTONS * NB_MOUTONS * NB_MOUTONS * NB_MOUTONS * NB_PILIERS + 1




