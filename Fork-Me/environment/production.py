#!/usr/bin/python3
# -*- coding: utf-8 -*-


def apply(launcher):
    launcher.addEventHandler('to_start', lambda loc_launcher: printToto(loc_launcher))

def printToto(launcher):
    launcher.info("Launcher 'to_start' from module production")

