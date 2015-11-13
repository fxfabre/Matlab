#!/usr/bin/python3
# -*- coding: utf-8 -*-

import InputFiles
import Constants
import LogManager
import PluginsManager
from DbConnexion import DbConnexion
from Launcher import Launcher
import os


def main():
    ### Parse ARGV
    params = Constants.getParameters()
    logger = LogManager.initializeLogger(params['log'], params['programName'])

    logger.info("{0} version {1}".format(params['programName'], params['programVersion']))
    logger.info("# {0}".format(params['id']))

    db = DbConnexion("")

    xmlFiles = InputFiles.getInputFiles(params['from'], params['ifs'], params['unlesses'])

    launcher = Launcher(xmlFiles, **params )
    PluginsManager.loadPlugins(launcher, params['environment'], params['add'], params['add_path'], logger)
    return launcher.start()


if __name__ == '__main__':
    exit( main() )

