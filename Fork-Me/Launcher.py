#!/usr/bin/python3
# -*- coding: utf-8 -*-


import InputFiles
import InputFile
import LogManager
from Status import Status

import multiprocessing as mp
import os
import time
import random

import logging

class Launcher:

    ########################
    # Processing functions # Parent process
    ########################
    def __init__(self, inputFiles, **params):
        self.environment    = params['environment']
        self.processes      = params['processes']
        self.id             = params['id']
        self.binary         = params['binary']
        self.time           = params['time']
        self.maxStarts      = params['maxStarts']
        self.jsonErrorFile  = params['jsonErrorFile']

        self.categorie      = params['label']
        self.infra          = params['stress_test']

        self.parameters     = params
        self.inputFiles     = inputFiles
        self.logger         = logging.getLogger(params['programName'])

        self.eventHandlers  = None
        self.setDefaultEventHandler()

        self.__removeJsonErrorFile()

    def setDefaultEventHandler(self):
        self.eventHandlers = {
            'ignore'    : [],
            'compile'   : [],
            'to_start'  : [],
            'started'   : [],
            'finish'    : [],
            'finished'  : [],
            'end'       : [lambda : self.logger.info("Process main end !")],
            'xmlFilter' : [],
            'runCommand': []
        }

    def runEventhandler(self, eventName, *args, **kwargs):
        self.logger.debug("Running eventHandler {0}".format(eventName) )
        if eventName not in self.eventHandlers:
            return
        for function in self.eventHandlers[eventName]:
            self.logger.debug("EventHandler {0} : run function {1}".format(eventName, function.__name__))
            function(*args, **kwargs)

    def addEventHandler(self, eventName, function):
        self.logger.debug("Adding function {0} to eventHandler {1}".format(function.__name__, eventName))
        if eventName not in self.eventHandlers:
            self.eventHandlers[eventName] = []
        self.eventHandlers[eventName].append( function )

    def start(self):
        self.runEventhandler('to_start')
        self.logger.info("Starting Fork-Me")

        self.logger.debug("Creating semaphore with {0} process max".format(self.processes))
        semaphore = mp.Semaphore( self.processes )
        childParam = {
            'semaphore' : semaphore,
            'inputFile' : None,
            'time'      : self.time,
            'categorie' : self.categorie,
            'maxStarts' : self.maxStarts
        }

        for inputFile in self.inputFiles:
            childParam['inputFile'] = inputFile
            process = mp.Process(target=self.__startChildProcessing,
                                 name=inputFile.fileName,
                                 kwargs=childParam,
                                 daemon=True)
            inputFile.process = process
            process.start()
            self.logger.debug("Process for file {0} started".format(inputFile.fileName))

        self.logger.info("Waiting for all childs to end")
        self.inputFiles.waitAllProcess()
        self.runEventhandler('end')

    ########################
    # Processing functions # Child processes
    ########################
    def __startChildProcessing(self, **params):
        inputFile = params['inputFile']
        semaphore = params['semaphore']
        logger = LogManager.initializeLogger(self.parameters['log'], inputFile.fileName)
        inputFile.logger = logger

        logger.info("Process {0} for {1} started".format(os.getpid(), inputFile.fileName))

        while self.__canRunInputFile(inputFile, params):
            logger.debug("Processing {0} by {1}".format(inputFile.fileName, os.getpid()))
            inputFile.nbStarts += 1

            semaphore.acquire( block=True, timeout=None)

            try:
                self.__runChild(inputFile, params, logger)
            except Exception as e:
                logger.error("Exception in processing of " + inputFile.fileName)
                logger.exception(str(e))
                inputFile.status = Status.ERROR

            semaphore.release()
        logger.info("{0} ended".format(inputFile.fileName) )

    def __runChild(self, inputFile, params, logger):
        inputFile.status = Status.RUNNING
        logger.debug("Running file {0}".format(inputFile.fileName))

        time.sleep( random.uniform(2, 10) )

        inputFile.status = Status.SUCCESS
        logger.debug("Process for {0} finished".format(inputFile.fileName))
        return

    def __canRunInputFile(self, inputFile, params):
        """
        Return False if we must not run the file
        Return True  if we can run file now
        params : dico avec time, categorie, description, maxStarts, db
        """
        if inputFile.status == Status.SUCCESS:
            return False
        if inputFile.status == Status.RUNNING:
            return False

        self.runEventhandler('xmlFilter', inputFile)
        if inputFile.status == Status.IGNORE:
            self.runEventhandler('ignore', inputFile)
            return False

        if inputFile.nbStarts >= params['maxStarts']:
            inputFile.logger.debug("{0} a dépassé le nombre max de lancement. Exit".format(inputFile.fileName))
            inputFile.status = Status.ERROR
            return False

        inputFile.status = Status.TO_START

        if not inputFile.mustWaitForCarto():
            inputFile.logger.debug( "{0} : don't wait for carto".format(inputFile.fileName))
            return True

        cartoFinished = False
        while not cartoFinished:
            timeToWait = inputFile.timeWaitForCarto()
            inputFile.logger.debug("{0} : waiting {1} secs for carto".format(inputFile.fileName, timeToWait))
            time.sleep( timeToWait )
            cartoFinished = inputFile.isCartoFinished( params['db'] )   # run SQL query

        inputFile.logger.info( "Carto finished for {0}".format(inputFile.fileName))
        return True

    ##########################
    # Traitement des erreurs #
    ##########################s
    def __processJsonErrorFile(self, inputFile, errors):
        xmlPath = inputFile.fileName
        if len(errors) > 0:
            inputFile.logger.info("{0} failed".format(xmlPath))
        else:
            inputFile.logger.info("{0} succeeded".format(xmlPath))

        for error in errors:
            inputFile.logger.error("error : {0}".format(error))

    def __removeJsonErrorFile(self):
        if os.path.exists( self.jsonErrorFile ):
            os.remove( self.jsonErrorFile )

