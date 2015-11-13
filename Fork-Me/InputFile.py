#!/usr/bin/python3
# -*- coding: utf-8 -*-

from Status import Status

import os
import time


class InputFile:
    def __init__(self, filePath):
        self.filePath = filePath
        self.fileName = os.path.basename( filePath )
        self.nbStarts = 0

        self.dateLastCheckCarto = 0 # 01/01/1970
        self.timeSpanWaitCarto = -1 # Min time between 2 check carto

        self.status = Status.TO_START
        self.process = None
        self.logger = None

    ############################
    # Carto specific functions
    ############################
    def mustWaitForCarto(self):
        if self.timeSpanWaitCarto < 0:
            self.timeSpanWaitCarto = 0 # TODO : init timeSpanWaitCarto avec la valeur du flux XML
        if self.timeSpanWaitCarto == 0:
            return False
        return True

    def timeWaitForCarto(self):
        """
        Doit renvoyer le temps restant avant le prochain check carto (en secondes),
        en prenant en compte la date à laquelle on a réalisé le dernier check carto
        """
        if not self.mustWaitForCarto():
            print("_ERROR_ : Should not wait for carto")
            return 0
        timeToWait = self.dateLastCheckCarto + self.timeSpanWaitCarto - time.time()
        return max(0, timeToWait)

    def isCartoFinished(self, dbConnexion):
        self.dateLastCheckCarto = time.time()
        cartos = dbConnexion.checkCarto(self.getDate(), self.getFolio(), self.getCategorie(), self.getDescription())
        if len(cartos) > 0:
            return True
        return False

    ######################
    # XML Getters
    ######################
    def getNiceness(self):
        return 0
    def getDate(self):
        return '10/11/2015'
    def getFolio(self):
        return 333
    def getCategorie(self):
        return 'PROD'
    def getDescription(self):
        return 'Description'


    ########################
    # Overloaded functions #
    ########################
    def __str__(self):
        return str("Input file : " + self.fileName)

    def __repr__(self):
        return str("Input file : " + self.fileName)



