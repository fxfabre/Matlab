#!/usr/bin/python3
# -*- coding: utf-8 -*-


def apply(launcher):
    launcher.addEventHandler('to_start', printToto)
    launcher.addEventHandler('runCommand', runCommand)
    launcher.addEventHandler('finishing', finishing)

def printToto(launcher):
    launcher.info("Launcher 'to_start' from module development")

def runCommand(launcher, inputFile, **cmdLineParams):
    cmdLineToRun = ""
    if cmdLineParams['tiger']:
        cmdLineToRun = "{binary} -grid {tiger} -in {xmlPath} -out {to} -service position{infra} > {logFile} 2>& 1".format(
            **cmdLineParams
        )
    else:
        cmdLineToRun = "{binary} {is_a} {xmlPath} {to} > {logFile} 2>& 1".format(**cmdLineParams)
    inputFile.logger.info("Running : \n" + cmdLineToRun)
    return 0

def finishing(launcher, inputFile, listErrors):
    inputFile.logger.info("Ingnoring {0} errors".format(inputFile.fileName))
    listErrors.clear()
    return 0

