#!/usr/bin/python3

from Board import *

board = Board(3)


####################################
def CanMoveTester(l:list, expect:bool=True):
    global board
    if (len(l) > 0):
        board.SetBoard(l)
    res = board.CanMove()
    status = ". KO"
    if res == expect:
        status = ""
    print("{0} : {1}{2}".format(board, res, status ))

def RunCanMoveTests():
    print(" *** Tests can move ? ***")
    CanMoveTester([])
    CanMoveTester([0,0,0, 0,0,0, 0,0,0])

    """ Almost empty boards """
    CanMoveTester([1,0,0, 0,0,0, 0,0,0])
    CanMoveTester([0,1,0, 0,0,0, 0,0,0])
    CanMoveTester([0,0,1, 0,0,0, 0,0,0])
    CanMoveTester([0,0,0, 1,0,0, 0,0,0])
    CanMoveTester([0,0,0, 0,1,0, 0,0,0])
    CanMoveTester([0,0,0, 0,0,1, 0,0,0])
    CanMoveTester([0,0,0, 0,0,0, 1,0,0])
    CanMoveTester([0,0,0, 0,0,0, 0,1,0])
    CanMoveTester([0,0,0, 0,0,0, 0,0,1])

    """ Full boards, with one 0 """
    CanMoveTester([0,2,3, 4,5,6, 7,8,9])
    CanMoveTester([1,0,3, 4,5,6, 7,8,9])
    CanMoveTester([1,2,0, 4,5,6, 7,8,9])
    CanMoveTester([1,2,3, 0,5,6, 7,8,9])
    CanMoveTester([1,2,3, 4,0,6, 7,8,9])
    CanMoveTester([1,2,3, 4,5,0, 7,8,9])
    CanMoveTester([1,2,3, 4,5,6, 0,8,9])
    CanMoveTester([1,2,3, 4,5,6, 7,0,9])
    CanMoveTester([1,2,3, 4,5,6, 7,8,0])

    """ Check move in lines """
    CanMoveTester([1,1,3, 4,5,6, 7,8,9])
    CanMoveTester([1,2,2, 4,5,6, 7,8,9])
    CanMoveTester([1,2,3, 4,4,6, 7,8,9])
    CanMoveTester([1,2,3, 4,6,6, 7,8,9])
    CanMoveTester([1,2,3, 4,5,6, 7,7,9])
    CanMoveTester([1,2,3, 4,5,6, 7,9,9])

    """ Check move in column """
    CanMoveTester([1,2,3, 1,5,6, 7,8,9])
    CanMoveTester([1,2,3, 4,2,6, 7,8,9])
    CanMoveTester([1,2,3, 4,5,3, 7,8,9])
    CanMoveTester([1,2,3, 4,5,6, 4,8,9])
    CanMoveTester([1,2,3, 4,5,6, 7,5,9])
    CanMoveTester([1,2,3, 4,5,6, 7,8,6])

    """ Not able to move """
    CanMoveTester([1,2,3, 4,5,6, 7,8,9], False)
    CanMoveTester([1,2,1, 2,1,2, 1,2,1], False)
    print()

RunCanMoveTests()


####################################
def MovetLeft(l:list) -> list:
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

def MoveLeftTester(l:list, expected:list):
    outputList = list(l)
    MovetLeft(outputList)
    status = "OK" if outputList == expected else "Expecting " + str(expected)
    print("{0} -> {1} : {2}".format(l, outputList, status))

def RunMoveLeftTests():
    MoveLeftTester([0,0,0,0], [0,0,0,0])
    MoveLeftTester([1,2,4,8], [1,2,4,8])

    MoveLeftTester([0,2,4,8], [2,4,8,0])
    MoveLeftTester([1,0,4,8], [1,4,8,0])
    MoveLeftTester([1,2,0,8], [1,2,8,0])
    MoveLeftTester([1,2,4,0], [1,2,4,0])

    MoveLeftTester([1,1,4,8], [2,4,8,0])
    MoveLeftTester([1,2,2,8], [1,4,8,0])
    MoveLeftTester([1,2,4,4], [1,2,8,0])

    MoveLeftTester([0,0,4,8], [4,8,0,0])
    MoveLeftTester([1,0,0,8], [1,8,0,0])
    MoveLeftTester([1,2,0,0], [1,2,0,0])

    MoveLeftTester([1,0,1,0], [2,0,0,0])
    MoveLeftTester([1,1,2,2], [2,4,0,0])
    MoveLeftTester([0,0,2,2], [4,0,0,0])
    MoveLeftTester([2,0,0,2], [4,0,0,0])

    print()

RunMoveLeftTests()


####################################
def MovetRight(l:list) -> list:
    l.reverse()
    MovetLeft(l).reverse()
    return l

def MoveRightTester(l:list, expected:list):
    outputList = MovetRight(list(l))
    status = "OK" if outputList == expected else "Expecting " + str(expected)
    print("{0} -> {1} : {2}".format(l, outputList, status))

def RunMoveRightTests():
    MoveRightTester([0,0,0,0], [0,0,0,0])
    MoveRightTester([1,2,4,8], [1,2,4,8])

    MoveRightTester([2,4,8,0], [0,2,4,8])
    MoveRightTester([1,0,4,8], [0,1,4,8])
    MoveRightTester([1,2,0,8], [0,1,2,8])
    MoveRightTester([0,1,2,4], [0,1,2,4])

    MoveRightTester([1,1,4,8], [0,2,4,8])
    MoveRightTester([1,2,2,8], [0,1,4,8])
    MoveRightTester([1,2,4,4], [0,1,2,8])

    MoveRightTester([4,8,0,0], [0,0,4,8])
    MoveRightTester([1,0,0,8], [0,0,1,8])
    MoveRightTester([1,2,0,0], [0,0,1,2])

    MoveRightTester([1,0,1,0], [0,0,0,2])
    MoveRightTester([1,1,2,2], [0,0,2,4])
    MoveRightTester([2,2,0,0], [0,0,0,4])

    print()

RunMoveRightTests()

