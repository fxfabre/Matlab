#!/usr/bin/python3
# -*- coding: utf-8 -*-

import DbConnexion

import datetime
import logging
import re


def to_time( time ):
    logger = logging.getLogger(__name__)

    # time = yyyyMMdd
    if re.match(r"^\d{8}$", time):
        return datetime.datetime.strptime(time, '%Y%m%d').strftime('%d/%m/%Y')

    # time = oracle://login/mdp@server
    if re.match(r'^(\w+)://(\w+)/(\w+)@(\w+)$', time):
        results = DbConnexion.queryTimeFromDb(time)
        return to_time( results[0]['TIME'] )

    # return J-1
    # TODO : voir le return J-1 dans la fonction perl

    # time = yyyy/mm/dd
    return time

