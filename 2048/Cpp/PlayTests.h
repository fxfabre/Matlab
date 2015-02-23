#pragma once

#include "ExpectiMax.h"
#include "BoardDB.h"

unsigned int PlayTest(const HeuristicParameters& parameters, unsigned int rollbacks, BoardDB* boarddb = NULL);

void PlayTest_Tune(unsigned int rollbacks, unsigned int plays, unsigned int population_size, unsigned int tournament_size, unsigned int latency);
void PlayTest_Batch(unsigned int rollbacks, unsigned int plays, BoardDB* boarddb);
