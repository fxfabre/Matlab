#!/usr/bin/python3
# -*- coding: utf-8 -*-


import random
import tkinter as TK
import tkinter.messagebox as MB
from tkinter import ttk
from time import sleep

import GUI.game2048_score as GS
import GUI.game2048_grid as GG

from AI.ai_random import *

class Game2048(TK.Tk):

    # component disposal padding
    PADDING = 10

    # number of tiles to show at startup
    START_TILES = 2


    def __init__ (self, **kw):
        # super class inits
        TK.Tk.__init__(self)

        # widget inits
        self.init_widget(**kw)

        # prevent from accidental displaying
        self.withdraw()

        # Init AI
        self._ai = ai_random()

    def center_window(self, tk_event=None, *args, **kw):
        """
            tries to center window along screen dims;
            no return value (void);
        """

        # ensure dims are correct
        self.update_idletasks()

        # window size inits
        _width = self.winfo_reqwidth()
        _height = self.winfo_reqheight()
        _screen_width = self.winfo_screenwidth()
        _screen_height = self.winfo_screenheight()

        # make calculations
        _left = (_screen_width - _width) // 2
        _top = (_screen_height - _height) // 2

        # update geometry
        self.geometry("+{x}+{y}".format(x=_left, y=_top))

    def init_widget(self, **kw):
        """
            widget's main inits;
        """

        # main window inits
        self.title("2048")
        self.protocol("WM_DELETE_WINDOW", self.quit_app)
        self.resizable(width=False, height=False)

        # look'n'feel
        ttk.Style().configure(".", font="sans 10")

        # inits
        _pad = self.PADDING

        # get 2048's grid
        self.grid = GG.Game2048Grid(self, **kw)

        # hint subcomponent
        self.hint = ttk.Label(self, text="Hint: use keyboard arrows to move tiles.")

        # score subcomponents
        self.score = GS.Game2048Score(self, **kw)
        self.hiscore = GS.Game2048Score(self, label="Highest:", **kw)

        # layout inits
        self.grid.pack(side=TK.TOP, padx=_pad, pady=_pad)
        self.hint.pack(side=TK.TOP)
        self.score.pack(side=TK.LEFT)
        self.hiscore.pack(side=TK.LEFT)

        # quit button
        ttk.Button(
            self, text="Play !", command=self.play_random,
        ).pack(side=TK.RIGHT, padx=_pad, pady=_pad)

        # new game button
        ttk.Button(
            self, text="New Game", command=self.new_game,
        ).pack(side=TK.RIGHT)

        # set score callback method
        self.grid.set_score_callback(self.update_score)

    def play_random(self, *args, **kw):
        self.grid.isGameOver = False

        i = 0
        nextMove = 'up'
        while len(nextMove) > 0:
            i += 1
            tk_event = TK.Event()
            nextMove = self._ai.move_next(self.grid)     # 'left', 'right', 'up' or 'down'
            tk_event.keysym = nextMove

            sleep(0.1)
            self.slot_keypressed(tk_event)

            self.grid.update()
        print("Game over in {0} iterations, stop game".format(i))

    def new_game(self, *args, **kw):
        """
            new game inits;
        """

        # no events now
        self.unbind_all("<Key>")

        # reset score
        self.score.reset_score()

        # reset grid
        self.grid.reset_grid()

        # make random tiles to appear
        for n in range(self.START_TILES):
            self.after(
                100 * random.randrange(3, 7), self.grid.pop_tile
            )

        # bind events
        self.bind_all("<Key>", self.slot_keypressed)

    def quit_app (self, **kw):
        """
            quit app dialog;
        """

        # ask before actually quitting
        if MB.askokcancel("Question", "Quit game?", parent=self):
            self.quit()

    def run (self, **kw):
        """
            actually runs the game;
        """

        # show up window
        self.center_window()
        self.deiconify()

        # init new game
        self.new_game(**kw)

        # enter the loop
        self.mainloop()

    def slot_keypressed (self, tk_event=None, *args, **kw):
        """
            keyboard input events manager;
        """

        # action slot multiplexer
        _slot = {
            "left" : self.grid.move_tiles_left,
            "right": self.grid.move_tiles_right,
            "up"   : self.grid.move_tiles_up,
            "down" : self.grid.move_tiles_down,
            "escape": self.quit_app,
        }.get(tk_event.keysym.lower())

        # got some redirection?
        if callable(_slot):
            _slot()
            # hints are useless by now
#            self.hint.pack_forget()

    def update_score (self, value, mode="add"):
        """
            updates score along @value and @mode;
        """
        # relative mode
        if str(mode).lower() in ("add", "inc", "+"):
            # increment score value
            self.score.add_score(value)

        # absolute mode
        else:
            # set new value
            self.score.set_score(value)

        # update high score
        self.hiscore.high_score(self.score.get_score())


# launching the game
if __name__ == "__main__":
    Game2048().run()
