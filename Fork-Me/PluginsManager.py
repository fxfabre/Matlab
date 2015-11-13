#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import os


import environment.production as production
import environment.test as test
import environment.development as development

availablePlugins = {
    'development'   : development,
    'test'          : test,
    'production'    : production
}

class PluginsManager:

    def __init__(self):
        pass


def loadPlugins(launcher, environment, pluginsToLoad, pluginsPaths, logger):
#    print("Loading plugins {0} and {1}".format(environment, ', '.join(pluginsToLoad)))

#    for path in pluginsPaths:
#        sys.path.append( os.path.realpath(path) )
#    sys.path.append(os.path.realpath('environment'))

    for plugin in pluginsToLoad:
        if plugin in availablePlugins:
            importPlugin(launcher, availablePlugins[plugin], logger)
        else:
            logger.error("Unable to load plugin " + str(plugin))

    importPlugin(launcher, availablePlugins[environment], logger)
#    print("__ plugin : " + str(environment))
#    __loadPlugins(launcher, "environment.{0}".format(environment), logger)
#    __loadPlugins(launcher, environment, logger)


def __loadPlugins(launcher, pluginName, logger):
    try:
        module = __import__(pluginName)
        print( dir(module) )
        print(type(module))
        print( module.__name__ )
        module.apply(launcher)
        logger.info("Import plugin {0} success".format(pluginName))
    except ImportError as e:
        logger.error("Unable to load plugin {0}".format(pluginName))
        logger.exception( e )

def importPlugin(launcher, plugin, logger):
    plugin.apply(launcher)
    logger.info("Import plugin {0} success".format(plugin.__name__))

