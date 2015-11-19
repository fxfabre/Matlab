#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import re
from InputFile import InputFile
from multiprocessing import Array # thread safe array

class InputFiles:
    def __init__(self, filesNames=[], folder='.'):
        files = map( lambda fileName: os.path.join(folder, fileName), filesNames)
        inputFiles = map( lambda file: InputFile(file), files)
#        self.inputFiles = Array(InputFile, list(inputFiles), lock=True)
        self.inputFiles = list( inputFiles )

    def orderBy(self):
        predicate = lambda inputFile: inputFile.Niceness
        self.files.sort(key=predicate)
        return self.files

    def waitAllProcess(self):
        for inputFile in self.inputFiles:
            if inputFile.process:
                inputFile.process.join()

    def countStatus(self, status):
        filtered = filter( lambda x: x.status == status, self.inputFiles)
        return len( list(filtered) )

    ########################
    # Overloaded functions #
    ########################
    def __str__(self):
        return str(self.inputFiles)

    def __repr__(self):
        return str(self.inputFiles)

    def __getitem__(self, key):
        return self.inputFiles[key]

    def __len__(self):
        return len( self.inputFiles )



def getInputFiles( folderFrom, ifs, unlesses ):
    files = filter(lambda f: f.endswith('.xml'), os.listdir(folderFrom))
    files = list( files )

#    print("\nAll files : \n{0}".format('\n'.join(map(lambda x: os.path.basename(x), files))) )

    if ifs:
        regexIf = list( map(lambda s: ".*{0}.*".format(s), ifs) )
        regexIf = '|'.join( regexIf)
#        print( "\nregexIf : " + regexIf )
        matched = []
        for file in files:
            res = re.match(regexIf, file)
            if res:
                matched.append(file)
        files = matched
#        print("\nAprès filter ifs  : \n{0}".format('\n'.join(map(lambda x: os.path.basename(x), files))) )

    if unlesses:
        patterns = list( map(lambda x: ".*" + x + ".*", unlesses) )
        pattern = '|'.join(patterns)
        regexUnless = re.compile(pattern)
#        print( "\nregex unless : " + pattern)
        matched = []
        for file in files:
            res = regexUnless.match(file)
            if res:
#                print(file + " match")
                pass
            else:
#                print(file + " ne match pas")
                matched.append(file)
        files = matched

#        print("\nAprès filter unless  : \n{0}".format('\n'.join(map(lambda x: os.path.basename(x), files))) )

    filesPath = map(lambda f: os.path.join(folderFrom, f), files)
    return InputFiles( list(filesPath) )



