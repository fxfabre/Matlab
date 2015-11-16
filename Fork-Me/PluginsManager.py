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

    for plugin in pluginsToLoad:
        if plugin in availablePlugins:
            importPlugin(launcher, availablePlugins[plugin], logger)
        else:
            logger.error("Unable to load plugin " + str(plugin))

    importPlugin(launcher, availablePlugins[environment], logger)

def importPlugin(launcher, plugin, logger):
    plugin.apply(launcher)
    logger.info("Import plugin {0} success".format(plugin.__name__))

