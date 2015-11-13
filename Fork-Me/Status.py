#!/usr/bin/python3
# -*- coding: utf-8 -*-

from enum import Enum


class Status(Enum):

    IGNORE = -1

    TO_START = 1

    RUNNING = 2

    SUCCESS = 3

    ERROR = 4