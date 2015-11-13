#!/usr/bin/python3
# -*- coding: utf-8 -*-

import UtilityFunctions

import argparse
import sys
import os


PROGRAM_NAME = "Fork-Me"
PROGRAM_VERSION = "2.0.0"


def getParameters():
    __setDefaultEnvironmentVar()

    defaultParams = __getDefaultParameters()

    customParams = __parseArgs()
    defaultParams.update(customParams)

    return __updateParameters( defaultParams )

def __updateParameters( params ):
    # convert connexion string to date dd/mm/yyyy
    if not(params['time'] is None):
        print("Update time : " + str(params['time']))
        params['time'] = UtilityFunctions.to_time( params['time'] )

    # set program id
    if not params['id']:
        ids = ['ifs']
        ids.extend( params['ifs'] )
        ids.append('unlesses')
        ids.extend( params['unlesses'] )
        params['id'] = '_'.join( ids )

    # set json error file name
    params['jsonErrorFile'] = "restart_" + params['id']
    params['jsonErrorPath'] = os.path.join(params['from'], params['jsonErrorFile'])
    params['programName']   = PROGRAM_NAME
    params['programVersion']= PROGRAM_VERSION

    # check la cohérence de certains paramètres
    if params['tiger']:
        if not params['stress_test']:
            print( "__ERROR__ : option stress_test must be set")
            exit(1)
    else:
        if params['is_a'] != 'FULL_GENERIQUE':
            print("__ERROR__ : option is_a must be set to 'FULL_GENERIQUE'")
            exit(1)
    return params

def __parseArgs():
    parser = argparse.ArgumentParser(description='Fork-Me launcher')

    # program parameters
    parser.add_argument('--environment' , default='production'  , type=str,
                                choices=['development', 'test', 'production'],
                                help="Nom du fichier d'environnement à charger")
    parser.add_argument('--id'          , default=''            , type=str,
                                help="Valeur par défaut : fork_me_ifs_xxx_unlesses_xxx")
    parser.add_argument('--is_a'        , default=''            , type=str)
    parser.add_argument('--time'        ,
                                help="Date utilisée pour charger les données. yyyyMMdd ou oracle://user/mdp@base")
    parser.add_argument('--log'         , default='./Logs'      , type=str,
                                help="Dossier a utiliser pour enregistrer les logs")

    # plugins parameters
    parser.add_argument('--add'         , default=[]            , action='append',
                                help="nom / adresse d'un dossier contenant un plugin de check / de reprise")
    parser.add_argument('--add_path'    , default=[]            , action='append',
                                help="chemin où il faut chercher les plugins de check / de reprise")

    # input files
    parser.add_argument('--from'        , default='.'           , type=str,
                                help="dossier contenant les flux d'input")
    parser.add_argument('--if'          , default=[]            , action='append'   , dest='ifs',
                                help="RegEx insensible à la casse pour sélectionner les flux à lancer par leur nom")
    parser.add_argument('--unlesses'    , default=[]            , action='append',
                                help="RegEx insensible à la casse pour supprimer le traitement de flux XML")

    # program to launch
    parser.add_argument('--binary'      , default=''            , type=str,
                                help="Le nom du programme / de la commande à lancer : appbatches ou tigerClient")
    parser.add_argument('--processes'   , default=1             , type=int,
                                help="Nombre maximum de process que l'on peut lancer simultanément")
    parser.add_argument('--restarts'    , default=1             , type=int          , dest='maxStarts',
                                help="Nombre maximum de lancement d'un flux d'input")
    parser.add_argument('--tiger'       , default=''            , type=str)
    parser.add_argument('--to'          , default='.'           , type=str,
                                help="Chemin du fichier XML de résultat généré par le 'binary'")

    # parametres de l'infra
    parser.add_argument('--label'       , type=str              , help='Categorie')
    parser.add_argument('--stress_test', default=''             , help="nom de l'infra", type=str)

    args = vars( parser.parse_args() )
    return args

def __getDefaultParameters():
    parameters = {
        'ProgramName'   : 'fork_me'
    }
    return parameters

def __setDefaultEnvironmentVar():
    __addEnvValueToPath( 'CLIENT_TIGER_PATH')
    __addEnvValueToPath( 'CLIENT_MICO_PATH' )

def __addEnvValueToPath(envVarName):
    if envVarName is None:
        return

    envVarValue = os.getenv(envVarName, None)
    if envVarValue is None:
        return

    if os.path.exists( envVarValue ):
        sys.path.append( envVarValue )
