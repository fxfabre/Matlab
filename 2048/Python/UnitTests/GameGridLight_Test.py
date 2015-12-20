#!/usr/bin/python3
# -*- coding: utf-8 -*-


import unittest
import numpy as np
import AI.GameGridLight as GGL


class testConstructor(unittest.TestCase):

    def test_Initialisation(self):
        tab = np.array(
            [[0,0,0,0],
             [0,0,0,0],
             [0,0,0,0],
             [0,0,0,0]]
        )
        grid = GGL.gameGridLight(0, 0, tab)
        self.assertEqual(4, grid.rows)
        self.assertEqual(4, grid.columns)

        grid = GGL.gameGridLight(3, 4)
        self.assertEqual(3, grid.rows)
        self.assertEqual(4, grid.columns)
        self.assertTrue(grid.array2DEquals(np.zeros([3,4])))

        print("TestInitialisation success")


class TestMoves(unittest.TestCase):

    def test_MoveLeft(self):
        tab = np.array(
            [[2,2,0,0],
             [2,0,2,0],
             [2,0,0,2],
             [0,2,0,2]]
        )
        target = np.array(
            [[4,0,0,0],
             [4,0,0,0],
             [4,0,0,0],
             [4,0,0,0]]
        )
        grid = GGL.gameGridLight(0,0,tab)
        grid.moveLeft()
        self.assertTrue(grid.array2DEquals(target), str(grid.matrix) + "\n" + str(target))

        tab = np.array(
            [[2,2,0,2],
             [2,0,4,2],
             [0,2,0,4],
             [4,0,2,2],
             [2,0,2,4]]
        )
        target = np.array(
            [[4,2,0,0],
             [2,4,2,0],
             [2,4,0,0],
             [4,4,0,0],
             [4,4,0,0]]
        )
        grid = GGL.gameGridLight(0,0,tab)
        grid.moveLeft()
        self.assertTrue(grid.array2DEquals(target), str(grid.matrix) + "\n" + str(target))

        print("TestMoveLeft success")

    def test_MoveRight(self):
        tab = np.array(
            [[2,2,0,0],
             [2,0,2,0],
             [2,0,0,2],
             [0,2,0,2]]
        )
        target = np.array(
            [[0,0,0,4],
             [0,0,0,4],
             [0,0,0,4],
             [0,0,0,4]]
        )
        grid = GGL.gameGridLight(0,0,tab)
        grid.moveRight()
        self.assertTrue(grid.array2DEquals(target), str(grid.matrix) + "\n" + str(target))

        tab = np.array(
            [[2,2,0,2],
             [2,0,4,2],
             [0,2,0,4],
             [4,0,2,2]]
        )
        target = np.array(
            [[0,0,2,4],
             [0,2,4,2],
             [0,0,2,4],
             [0,0,4,4]]
        )
        grid = GGL.gameGridLight(0,0,tab)
        grid.moveRight()
        self.assertTrue(grid.array2DEquals(target), str(grid.matrix) + "\n" + str(target))

        print("TestMoveRight success")

    def test_MoveUp(self):
        tab = np.array(
            [[2,2,2,0,0,0],
             [2,0,0,2,2,0],
             [0,2,0,2,0,2],
             [0,0,2,0,2,2]]
        )
        target = np.array(
            [[4,4,4,4,4,4],
             [0,0,0,0,0,0],
             [0,0,0,0,0,0],
             [0,0,0,0,0,0]]
        )
        grid = GGL.gameGridLight(0,0,tab)
        grid.moveUp()
        self.assertTrue(grid.array2DEquals(target), "\n" + str(grid.matrix) + "\n" + str(target))

        tab = np.array(
            [[2,2,2,4,0,0],
             [2,4,0,2,2,4],
             [4,2,4,2,4,2],
             [0,4,2,4,2,2]]
        )
        target = np.array(
            [[4,2,2,4,2,4],
             [4,4,4,4,4,4],
             [0,2,2,4,2,0],
             [0,4,0,0,0,0]]
        )
        grid = GGL.gameGridLight(0,0,tab)
        grid.moveUp()
        self.assertTrue(grid.array2DEquals(target), "\n" + str(grid.matrix) + "\n" + str(target))

        tab = np.array(
            [[2,2,0,4],
             [2,0,2,0],
             [0,4,0,2],
             [2,2,4,2]]
        )
        target = np.array(
            [[4,2,2,4],
             [2,4,4,4],
             [0,2,0,0],
             [0,0,0,0]]
        )

        grid = GGL.gameGridLight(0,0,tab)
        grid.moveUp()
        self.assertTrue(grid.array2DEquals(target), "\n" + str(grid.matrix) + "\n" + str(target))

        print("TestMoveUp success")

    def test_MoveDown(self):
        tab = np.array(
            [[2,2,2,0,0,0],
             [2,0,0,2,2,0],
             [0,2,0,2,0,2],
             [0,0,2,0,2,2]]
        )
        target = np.array(
            [[0,0,0,0,0,0],
             [0,0,0,0,0,0],
             [0,0,0,0,0,0],
             [4,4,4,4,4,4]]
        )
        grid = GGL.gameGridLight(0,0,tab)
        grid.moveDown()
        self.assertTrue(grid.array2DEquals(target), "\n" + str(grid.matrix) + "\n" + str(target))

        tab = np.array(
            [[2,2,2,4,0,0],
             [2,4,0,2,2,4],
             [4,2,4,2,4,2],
             [0,4,2,4,2,2]]
        )
        target = np.array(
            [[0,2,0,0,0,0],
             [0,4,2,4,2,0],
             [4,2,4,4,4,4],
             [4,4,2,4,2,4]]
        )
        grid = GGL.gameGridLight(0,0,tab)
        grid.moveDown()
        self.assertTrue(grid.array2DEquals(target), "\n" + str(grid.matrix) + "\n" + str(target))

        tab = np.array(
            [[2,2,0,4],
             [2,0,2,0],
             [0,4,0,2],
             [2,2,4,2]]
        )
        target = np.array(
            [[0,0,0,0],
             [0,2,0,0],
             [2,4,2,4],
             [4,2,4,4]]
        )
        grid = GGL.gameGridLight(0,0,tab)
        grid.moveDown()
        self.assertTrue(grid.array2DEquals(target), "\n" + str(grid.matrix) + "\n" + str(target))

        print("TestMoveDown success")


class TestTools(unittest.TestCase):

    def test_ArrayDEquals(self):
        a = np.array(
            [[0,0,0],
             [0,0,0]]
        )
        b = np.array(
            [[0,0],
             [0,0]]
        )
        self.assertFalse(GGL.gameGridLight(0,0,a).array2DEquals(b))

        b = np.array(
            [[0,0,0],
             [0,0,0],
             [0,0,0]]
        )
        self.assertFalse(GGL.gameGridLight(0,0,a).array2DEquals(b))

        b = np.array(
            [[0,0,0],
             [0,0,0]]
        )
        self.assertTrue(GGL.gameGridLight(0,0,a).array2DEquals(b))

        a = np.array(
            [[1,3,0],
             [0,4,0]]
        )
        b = np.array(
            [[1,3,0],
             [1,4,0]]
        )
        self.assertFalse(GGL.gameGridLight(0,0,a).array2DEquals(b))

        a = np.array(
            [[4,3,0],
             [0,4,0]]
        )
        b = np.array(
            [[1,3,0],
             [0,4,0]]
        )
        self.assertFalse(GGL.gameGridLight(0,0,a).array2DEquals(b))

        a = np.array(
            [[1,3,0],
             [0,4,9]]
        )
        b = np.array(
            [[1,3,0],
             [0,4,0]]
        )
        self.assertFalse(GGL.gameGridLight(0,0,a).array2DEquals(b))

        a = np.array(
            [[1,3,0],
             [0,4,0]]
        )
        b = np.array(
            [[1,3,0],
             [0,4,0]]
        )
        self.assertTrue(GGL.gameGridLight(0,0,a).array2DEquals(b))

        print("TestArray2DEquals success")


if __name__ == '__main__':
    unittest.main()

