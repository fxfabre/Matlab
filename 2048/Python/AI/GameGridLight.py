#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np


class gameGridLight:

    def __init__(self, nbRows, nbColumns, matrix=None):
        if matrix is not None:
            nbRows = matrix.shape[0]
            nbColumns = matrix.shape[1]
            self.matrix = np.array(matrix)
        else:
            self.matrix = np.zeros([nbRows, nbColumns])

        self.rows = nbRows
        self.columns = nbColumns

    def getTileAt(self, row, column):
        return self.matrix[row, column]

    def moveTo(self, direction=""):
        direction = direction.lower()
        array_before_move = np.array(self.matrix)

        score = 0
        if direction == 'left':
            score = self.moveLeft()
        elif direction == 'right':
            score = self.moveRight()
        elif direction == 'up':
            score = self.moveUp()
        elif direction == 'down':
            score = self.moveDown()
        else:
            print("ERROR : Unknown direction : " + direction)

        if self.array2DEquals(array_before_move):
            return -10 # negative score : don't do this move
        return score

    def moveLeft(self):
        _at = self.matrix
        score = 0

        # Step 1 : fusion of equal tiles
        for _row in range(self.rows):
            for _column in range(self.columns - 1):

                # find same tile
                for _column_next in range(_column + 1, self.columns):
                    if _at[_row, _column_next] == 0:
                        continue
                    if _at[_row, _column] == _at[_row, _column_next]:
                        self.matrix[_row, _column] *= 2
                        self.matrix[_row, _column_next] = 0
                        score +=_at[_row, _column]
                    break

        # Step 2 : Move tiles
        for _row in range(self.rows):

            # Skip columns with > 0 tile
            _first_empty_column = self.columns # après la dernière colonne
            for _column in range(self.columns - 1):
                if _at[_row, _column] == 0:
                    _first_empty_column = _column
                    break

            # Move tiles
            for _column in range(_first_empty_column+1, self.columns):
                if _at[_row, _column] > 0:
                    self.matrix[_row, _first_empty_column] = _at[_row, _column]
                    _at[_row, _column] = 0
                    _first_empty_column += 1
        return score

    def moveRight(self):
        _at = self.matrix
        score = 0

        # Step 1 : fusion of equal tiles
        for _row in range(self.rows):
            for _column in range(self.columns - 1, 0, -1):

                # find same tile
                for _column_next in range(_column - 1, -1, -1):
                    if _at[_row, _column_next] == 0:
                        continue
                    if _at[_row, _column] == _at[_row, _column_next]:
                        self.matrix[_row, _column] *= 2
                        self.matrix[_row, _column_next] = 0
                        score +=_at[_row, _column]
                    break

        # Step 2 : Move tiles
        for _row in range(self.rows):

            # Skip columns with > 0 tile
            _first_empty_column = -1 # avant la premiere colonne
            for _column in range(self.columns - 1, -1, -1):
                if _at[_row, _column] == 0:
                    _first_empty_column = _column
                    break

            # Move tiles
            for _column in range(_first_empty_column-1, -1, -1):
                if _at[_row, _column] > 0:
                    self.matrix[_row, _first_empty_column] = _at[_row, _column]
                    _at[_row, _column] = 0
                    _first_empty_column -= 1
        return score

    def moveUp(self):
        _at = self.matrix
        score = 0

        # Step 1 : fusion of equal tiles
        for _column in range(self.columns):
            for _row in range(self.rows - 1):

                # find same tile
                for _row_next in range(_row + 1, self.rows):

                    if _at[_row_next, _column] == 0:
                        continue
                    if _at[_row, _column] == _at[_row_next, _column]:
                        self.matrix[_row, _column] *= 2
                        self.matrix[_row_next, _column] = 0
                        score +=_at[_row, _column]
                    break

        # Step 2 : Move tiles
        for _column in range(self.columns):

            # Skip columns with > 0 tile
            _first_empty_row = self.rows # après la dernière colonne
            for _row in range(self.rows - 1):
                if _at[_row, _column] == 0:
                    _first_empty_row = _row
                    break

            # Move tiles
            for _row in range(_first_empty_row+1, self.rows):
                if _at[_row, _column] > 0:
                    self.matrix[_first_empty_row, _column] = _at[_row, _column]
                    _at[_row, _column] = 0
                    _first_empty_row += 1
        return score

    def moveDown(self):
        _at = self.matrix
        score = 0

        # Step 1 : fusion of equal tiles
        for _column in range(self.columns):
            for _row in range(self.rows - 1, 0, -1):

                # find same tile
                for _row_next in range(_row - 1, -1, -1):

                    if _at[_row_next, _column] == 0:
                        continue
                    if _at[_row, _column] == _at[_row_next, _column]:
                        self.matrix[_row, _column] *= 2
                        self.matrix[_row_next, _column] = 0
                        score +=_at[_row, _column]
                    break

        # Step 2 : Move tiles
        for _column in range(self.columns):

            # Skip columns with > 0 tile
            _first_empty_row = -1 # avant la premiere ligne
            for _row in range(self.rows - 1, -1, -1):
                if _at[_row, _column] == 0:
                    _first_empty_row = _row
                    break

            # Move tiles
            for _row in range(_first_empty_row-1, -1, -1):
                if _at[_row, _column] > 0:
                    self.matrix[_first_empty_row, _column] = _at[_row, _column]
                    _at[_row, _column] = 0
                    _first_empty_row -= 1
        return score

    def pop_tile(self, tk_event=None, *args, **kw):
        """
            pops up a random tile at a given place;
        """
        # ensure we yet have room in grid
        if not self.is_full():
            # must have more "2" than "4" values
            _value = random.choice([2, 4, 2, 2])

            # set grid tile
            _row, _column = self.get_available_box()
            _tile = Game2048GridTile(self, _value, _row, _column)

            # make some animations
            _tile.animate_show()

            # store new tile for further use
            self.register_tile(_tile.id, _tile)
            self.matrix.add(_tile, *_tile.row_column, raise_error=True)

    def get_available_box (self):
        """
            looks for an empty box location;
        """

        # no more room in grid?
        if self.is_full():
            raise GG.GridError("no more room in grid")
        else:
            _at = self.matrix.get_object_at
            while True:
                _row = random.randrange(self.rows)
                _column = random.randrange(self.columns)
                if not _at(_row, _column):
                    break
            return (_row, _column)

    def array2DEquals(self, arrayOther):
        if self.matrix.shape != arrayOther.shape:
            return False
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                if self.matrix[i,j] != arrayOther[i,j]:
                    return False
        return True

    def __str__(self):
        return str(self.matrix)

    def __eq__(self, other):
        return self.array2DEquals(other.matrix)

