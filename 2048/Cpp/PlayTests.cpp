#include "PlayTests.h"
#include "RandomSeed.h"
#include "BoardDB.h"
#include "Board.h"

#include <chrono>
#include <iostream>
#include <random>
#include <vector>


struct TuneElement {
	HeuristicParameters m_parameters;
	unsigned int m_score;
};

void PrintBoard(Board board) {
	for(unsigned int j = 0; j < 4; ++j) {
		std::cout << "[";
		for(unsigned int i = 0; i < 4; ++i) {
			unsigned int value = GetCell(board, i, j);
			std::cout.width(5);
			std::cout  << std::right << ((value == 0)? 0 : (1 << value));
		}
		std::cout << " ]" << std::endl;
	}
	std::cout << std::endl;
}

unsigned int Play2048(const HeuristicParameters& parameters, unsigned int rollbacks, BoardDB* boarddb)
{
	std::mt19937 rng(RandomSeed());
	Board board{0};

	auto t1 = std::chrono::high_resolution_clock::now();
	unsigned int moves = 0, score = 0;
	while (true)
	{
//		PrintBoard(board);

		/***********************************************
		 *	Insert random value in an empty location
		 ***********************************************/
		unsigned int locations[16], location_count = 0;
		// find all empty locations
		for(unsigned int location = 0; location < 16; ++location)
		{
			if(GetCell(board, location) == 0) {
				locations[location_count] = location;
				++location_count;
			}
		}
		// If no empty location (shouldn't happen) -> Exit
		if(location_count == 0)
		{
			std::cout << "Can't insert!" << std::endl;
			exit(1);
		}
		// Set random value in empty location
		uint new_position = locations[rng() % location_count];	// empty place, at random
		uint new_value    = (rng() % 10 == 0)? 2 : 1;			// 2 or 4
		board = SetCell(board, new_position, new_value);



		/***********************************************
		 *	Computer move left, up, down or right
		 ***********************************************/
		std::pair<enum_direction, uint> move = FindBestMove(board, parameters);
		if(move.first != DIRECTION_NONE)
		{
			BoardScore result = Collapse(board, move.first);
			if(result.m_board.m_data == board.m_data) {
				std::cout << "Invalid move!" << std::endl;
				exit(1);
			}
			board = result.m_board;
			score += result.m_score * 2;
			++moves;
		}
	}
//	PrintBoard(board);

	auto t2 = std::chrono::high_resolution_clock::now();
	unsigned int time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
	unsigned int time_per_move = time / moves;

	std::cout << "Game over - Move: " << moves << ", Time per move: " << time_per_move << ", Score: " << score << std::endl;

	return score;
}

void PlayTest_Batch(unsigned int rollbacks, unsigned int plays, BoardDB* boarddb)
{
	if(boarddb != NULL && rollbacks != 0)
	{
		std::cout << "PlayTest_Batch: Don't combine BoardDB with rollbacks!" << std::endl;
		exit(1);
	}

	HeuristicParameters parameters;
	GetDefaultHeuristicParameters(&parameters);

	// Array to save scores of each plays
	std::vector<unsigned int> scores(plays);

	// Launch the game
	for(unsigned int p = 0; p < plays; ++p) {
		std::cout << "Batch progress: " << 100 * p / plays << "%" << std::endl;
		scores[p] = Play2048(parameters, rollbacks, boarddb);
	}
	std::cout << "Finished ..." << std::endl;

	// Display scores
	std::cout << "scores = array([\n\t";
	for(unsigned int p = 0; p < plays; ++p){
		std::cout << scores[p];
		if(p != plays - 1) {
			if(p % 20 == 19)
				std::cout << ",\n\t";
			else
				std::cout << ", ";
		}
	}
	std::cout << "])" << std::endl;

	// Display average scores
	int total = 0;
	for (unsigned int p=0; p<plays; ++p) total += scores[p];
	std::cout << "Average score : " << (total / plays) << std::endl;
}







