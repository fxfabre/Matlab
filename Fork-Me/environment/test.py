#!/usr/bin/python3
# -*- coding: utf-8 -*-


def apply(launcher):
    launcher.addEventHandler('to_start', printToto)
    print("Module test imported")

def printToto():
    print("Message from module test")


