#!/usr/bin/python3
# -*- coding: utf-8 -*-

import InputFiles
import InputFile
import LogManager
import EventFunctions
from Status import Status

import multiprocessing as mp
import os
import time

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
            'ignore'    : [EventFunctions.ignore    ],
            'compile'   : [EventFunctions.compile   ],
            'begin'     : [EventFunctions.begin     ],
            'start'     : [EventFunctions.start     ],
            'finishing' : [EventFunctions.finishing ],
            'finish'    : [EventFunctions.finish    ],
            'end'       : [EventFunctions.end       ],
            'xmlFilter' : [EventFunctions.xmlFilter ],
            'runCommand': [EventFunctions.runCommand]
        }

    def runEventhandler(self, eventName, *args, **kwargs):
        self.logger.debug("Running eventHandler {0}".format(eventName) )
        codeRetour = 0
        if eventName not in self.eventHandlers:
            return codeRetour
        for function in self.eventHandlers[eventName]:
            self.logger.debug("EventHandler {0} : run function {1}".format(eventName, function.__name__))
            retour = function(*args, **kwargs)
            if retour is not None:
                codeRetour = max(codeRetour, retour)
        return codeRetour

    def addEventHandler(self, eventName, function):
        self.logger.debug("Adding function {0} to eventHandler {1}".format(function.__name__, eventName))
        if eventName not in self.eventHandlers:
            self.eventHandlers[eventName] = []
        self.eventHandlers[eventName].append( function )

    def start(self):
        self.runEventhandler('begin', self)
        self.logger.info("Starting Fork-Me")

        self.logger.debug("Creating semaphore with {0} process max".format(self.processes))
        semaphore = mp.Semaphore( self.processes )

        for inputFile in self.inputFiles:
            childParam = {
                'semaphore' : semaphore,
                'inputFile' : inputFile,
                'time'      : self.time,
                'categorie' : self.categorie,
                'infra'     : self.infra,
                'maxStarts' : self.maxStarts,
                'logFolder' : self.parameters['log'],
                'jsonErrorFile' : self.jsonErrorFile,
                'binary'    : self.binary,
                'tiger'     : self.parameters['tiger'],
                'to'        : self.parameters['to'],
                'is_a'      : self.parameters['is_a']
            }
            process = mp.Process(target=self.__startChildProcessing,
                                 name=inputFile.fileName,
                                 kwargs=childParam,
                                 daemon=True)
            inputFile.process = process
            process.start()
            self.logger.debug("Process for file {0} started".format(inputFile.fileName))

        self.logger.info("Waiting for all childs to end")
        self.inputFiles.waitAllProcess()
        self.runEventhandler('end', self)


    ########################
    # Processing functions # Child processes
    ########################
    def __startChildProcessing(self, **params):
        inputFile = params['inputFile']
        semaphore = params['semaphore']
        logger = LogManager.initializeLogger(self.parameters['log'], inputFile.fileName)
        inputFile.logger = logger

        inputFile.UpdateXml(params['time'], params['categorie'], params['infra'])

        while self.__canRunInputFile(inputFile, params):
            self.runEventhandler('start', self, inputFile)

            semaphore.acquire( block=True, timeout=None)
            logger.debug("{0} dans le semaphore".format(os.getpid()))

            try:
                inputFile.status = Status.RUNNING
                self.__runChild(inputFile, params, logger)
            except Exception as e:
                logger.error("Exception in processing of " + inputFile.fileName)
                logger.exception(str(e))
                inputFile.status = Status.ERROR
            finally:
                self.logger.debug("{0} libere le semaphore".format(os.getpid()))
                semaphore.release()

            self.runEventhandler('finish', self, inputFile)

    def __runChild(self, inputFile, params, logger):
        cmdLineParams = {
            'logFile'   : os.path.join(params['logFolder'], self.binary + "_" + inputFile.fileName + '.log'),
            'xmlPath'   : inputFile.filePath,
            'to'    : os.path.join(params['to'], inputFile.fileName),
            'binary': params['binary'],
            'infra' : params['infra'],
            'tiger' : params['tiger'],
            'is_a'  : params['is_a'],
        }

        jsonErrorFile = params['jsonErrorFile']

        listErrors = []

        codeRetour = self.runEventhandler('runCommand', self, inputFile, **cmdLineParams)
        if codeRetour > 0:
            listErrors.append("{0} returned {1}".format(params['binary'], codeRetour))

        if os.path.exists(jsonErrorFile):
            if not inputFile.isProcessingSuccess(jsonErrorFile):
                listErrors.append("Error found in file {0}".format(jsonErrorFile))
        else:
            listErrors.append("Error file {0} must exist".format(jsonErrorFile))

        self.runEventhandler('finishing', self, inputFile, listErrors)

        if len(listErrors) > 0:
            inputFile.logger.info("File {0} ends with errors : \n{1}".format(inputFile.fileName,'\n'.join(listErrors)))
            inputFile.status = Status.ERROR
        else:
            inputFile.logger.info("File {0} ends success".format(inputFile.fileName))
            inputFile.status = Status.SUCCESS

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

        self.runEventhandler('xmlFilter', self, inputFile)
        if inputFile.status == Status.IGNORE:
            self.runEventhandler('ignore', self, inputFile)
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
