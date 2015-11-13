#!/usr/bin/python3
# -*- coding: utf-8 -*-


import multiprocessing as mp

import Private

import logging
# import pyodbc


class DbConnexion:
    def __init__(self, connexionString):
#        self.logger = logging.getLogger(__name__)
        self.connexionString = None
        self.connexion = None
        self.cursor = None # TODO init new cursor
        self.mutex = mp.Semaphore(1)
        self.initConnexion(connexionString)

    def initConnexion(self, connexionString):
        connexionString = connexionString.strip()
        if len(connexionString) == 0:
            return
        self.connexionString = connexionString
        # TODO : init connexion

    def initConnexionFromFile(self, filePath, logger):
        connexionString = ''
        try:
            connexionParams = {}
            with open(filePath, 'r') as f:
                lines = f.readlines()
            for line in lines:
                content = line.split(':')
                key = content[0].strip()
                value = content[1].strip()
                if len(key) > 0:
                    connexionParams[key] = value
            connexionString = "oracle://{user}/{pwd}@{instance}".format(**connexionParams)
        except Exception as e:
            logger.error("Unable to parse file {0}".format(filePath))
            logger.exception( e )
            connexionString = ''
        self.initConnexion(connexionString)
        return connexionString

    def runQuerySelect(self, query):
        result = []
        self.mutex.acquire()
        try:
            result = self.cursor.execute( query )
        except Exception as e:
            self.logger.error( str(e) )
        self.mutex.release()
        return result

    def runQueryUpdate(self, query):
        self.mutex.acquire()
        try:
            self.cursor.execute( query )
            self.cursor.commit()
        except Exception as e:
            self.logger.error( str(e) )
        self.mutex.release()

    def closeConnexion(self):
        if self.connexion:
            self.connexion.close()
            self.connexion = None

    def checkCarto(self, date, folio, categorie, description):
        query = Private.QUERY_CARTO.format(
                date=date, folio=folio, categorie=categorie, description=description
            )
        return self.runQuerySelect( query )

    def __del__(self):
        self.closeConnexion()


def queryTimeFromDb( connexionString ):
    query = Private.QUERY_TIME
    db = DbConnexion( connexionString )
    time = db.runQuery( query )
    db.closeConnexion()
    return time


