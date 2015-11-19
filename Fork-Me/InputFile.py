#!/usr/bin/python3
# -*- coding: utf-8 -*-

from Status import Status
from Private import XPATH

import os
import time
import random
from lxml import etree

LOG_MESSAGE = {
    'debug'     : lambda logger, m: logger.debug(m),
    'info'      : lambda logger, m: logger.info(m),
    'warning'   : lambda logger, m: logger.warning(m),
    'error'     : lambda logger, m: logger.error(m),
    'exception' : lambda logger, m: logger.exception(m),
    'fatal'     : lambda logger, m: logger.fatal(m),
}

DEFAULT_VALUE = -9999999


class InputFile:

    def __init__(self, filePath):
        self.filePath = os.path.realpath( filePath )
        self.fileName = os.path.basename( filePath )
        self.nbStarts = 0

        self.dateLastCheckCarto = 0 # 01/01/1970
        self.timeSpanWaitCarto = -1 # Min time between 2 check carto

        try:
            self.xmlTree = etree.parse( self.filePath )
            self.__ok = True
        except Exception as e:
            print( e )
            self.xmlTree = None
            self.__ok = False

        self.status = Status.TO_START
        self.process = None
        self.logger = None

        # Set private fields for properties
        self.__niceness = DEFAULT_VALUE
        self.__date     = ''
        self.__folio    = DEFAULT_VALUE
        self.__categorie    = ''
        self.__description  = ''


    ############################
    # Carto specific functions
    ############################
    def initWaitForCartoProp(self):
        if self.timeSpanWaitCarto >= 0:
            return  # prop already initialized
        self.timeSpanWaitCarto = 0
        for xmlProp in self.xmlTree.xpath( XPATH['waitforcarto'] ):
            self.timeSpanWaitCarto = int(xmlProp.get('Value')) * 60

    def mustWaitForCarto(self):
        if self.timeSpanWaitCarto < 0:
            self.initWaitForCartoProp()
        return self.timeSpanWaitCarto > 0

    def timeWaitForCarto(self):
        """
        Doit renvoyer le temps restant avant le prochain check carto (en secondes),
        en prenant en compte la date à laquelle on a réalisé le dernier check carto
        """
        if not self.mustWaitForCarto():
            self.logMessage('error', "Should not wait for carto")
            return 0
        timeToWait = self.dateLastCheckCarto + self.timeSpanWaitCarto - time.time()
        return max(0, timeToWait)

    def isCartoFinished(self, dbConnexion):
        self.dateLastCheckCarto = time.time()
        cartos = dbConnexion.checkCarto(self.getDate(), self.getFolio(), self.getCategorie(), self.getDescription())
        if len(cartos) > 0:
            return True
        return False

    def isProcessingSuccess(self, jsonErrorFilePath):
        # TODO : Compléter la fonction
        status = random.random()
        if status > 0.5:
            return True
        return False

    #########################
    # XML Getters & setters #
    #########################
    @property
    def Niceness(self):
        if self.__niceness == DEFAULT_VALUE:
            if not self.xmlTree:
                return DEFAULT_VALUE
            for niceness in self.xmlTree.xpath( XPATH['niceness']):
                self.__niceness = int(niceness.text.strip())
        return self.__niceness

    @property
    def Date(self):
        if len( self.__date ) == 0:
            if not self.xmlTree:
                return ''
            for date in self.xmlTree.xpath( XPATH['date'] ):
                self.__date = date.text.strip()
        return self.__date

    @Date.setter
    def Date(self, date):
        if not self.xmlTree:
            return
        self.__date = date
        for xmlDate in self.xmlTree.xpath( XPATH['date'] ):
            xmlDate.text = date

    @property
    def Folio(self):
        if self.__folio == DEFAULT_VALUE:
            filename, _ = os.path.splitext( self.fileName )
            spitedName = filename.split('_')
            self.__folio = int(spitedName[1])
        return self.__folio

    @property
    def Categorie(self):
        if len(self.__categorie) == 0:
            if not self.xmlTree:
                return ''
            for categorie in self.xmlTree.xpath( XPATH['categorie'] ):
                self.__categorie = categorie.text.strip()
        return self.__categorie

    @Categorie.setter
    def Categorie(self, categorie):
        self.__categorie = categorie
        if not self.xmlTree:
            return
        for xmlCategorie in self.xmlTree.xpath( XPATH['categorie'] ):
            xmlCategorie.text = categorie

    @property
    def Description(self):
        if len(self.__description) == 0:
            if not self.xmlTree:
                return ''
            for desc in self.xmlTree.xpath( XPATH['description'] ):
                self.__description = desc.text.strip()
        return self.__description

    @Description.setter
    def Description(self, description):
        self.__description = description
        self.logMessage("Setting prop Description with value " + str(description))
        if not self.xmlTree:
            return
        for xmlDesc in self.xmlTree.xpath( XPATH['description'] ):
            xmlDesc.text = description


    #######################
    # Private functions
    #######################
    def UpdateXml(self, date='', categorie='', infra=''):
        self.logMessage('info', 'Updating XML file {0}'.format(self.fileName))

        if not self.xmlTree:
            self.logMessage('error', "Unable to read XML file {0}".format(self.fileName) )
            return
        if date and len(date) > 0:
            self.Date = date
        if categorie and len(categorie) > 0:
            self.Categorie = categorie

        if infra and len(infra) > 0:
            for xmlElt in self.xmlTree.xpath( XPATH['servercalculifcname'] ):
                xmlElt.text = "calcul" + infra
            for xmlElt in self.xmlTree.xpath( XPATH['servercalculname'] ):
                xmlElt.text = "calcul" + infra
            for xmlElt in self.xmlTree.xpath( XPATH['serverpositionname'] ):
                xmlElt.text = "position" + infra
            for xmlElt in self.xmlTree.xpath( XPATH['serverrepartitionname'] ):
                xmlElt.text = "repartition" + infra

        self.xmlTree.write( self.filePath )

    def logMessage(self, level, message):
        if not self.logger:
            print("{0:7} - {1}".format(level, message))
        else:
            LOG_MESSAGE[level](self.logger, message)


    ########################
    # Overloaded functions #
    ########################
    def __str__(self):
        return str("Input file : " + self.fileName)

    def __repr__(self):
        return str("Input file : " + self.fileName)

