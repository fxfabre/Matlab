#!/usr/bin/python3
# -*- coding: utf-8 -*-


import logging
import os
import sys
import multiprocessing as mp
from datetime import datetime


def getLoggerName(fileName):
    fileName = fileName.split('.')[0]
    dateString = datetime.now().strftime("%Y%m%d_%H%M%S")
    serverName = os.getenv("COMPUTERNAME", 'COMPUTERNAME')
    return '_'.join([fileName, dateString, serverName, str(os.getpid())])

def getLoggerParams(logFolder, fileName):
    baseName = getLoggerName(fileName)
    # TODO : convert logFolder from relative path to absolute path
    params = {
        'filename'  : os.path.join(logFolder, baseName + '.xml'),
        'format'    : "%(asctime)s %(levelname)s %(message)s",
        'datefmt'   : "%Y%m%d_%H%M%S",
        'level'     : logging.DEBUG,
    }
    return params, baseName

def initializeLogger_MP(logFolder, fileName):
    loggerName  = getLoggerName(fileName)
    loggerPath  = os.path.join(logFolder, loggerName + '.xml')
    logFormat   = '%(asctime)s - %(levelname)6s - %(message)s' # - %(name)s -
    datefmt     = "%Y%m%d_%H%M%S"

    logger = mp.get_logger()
    logger.setLevel(logging.DEBUG)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(loggerPath)
    fh.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter(logFormat, datefmt=datefmt)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    print("logger {0} created".format(loggerName))
    logger.debug("logger {0} created".format(loggerName))
    return logger



def initializeLogger(logFolder, fileName):
    loggerName  = getLoggerName(fileName)
    loggerPath  = os.path.join(logFolder, loggerName + '.xml')
    logFormat   = '%(asctime)s - %(levelname)6s - %(message)s' # - %(name)s -
    datefmt     = "%Y%m%d_%H%M%S"

    logger = logging.getLogger(loggerName)
    logger.setLevel(logging.DEBUG)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(loggerPath)
    fh.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter(logFormat, datefmt=datefmt)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.debug("logger {0} created".format(loggerName))
    return logger

