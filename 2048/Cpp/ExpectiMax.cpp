#include "ExpectiMax.h"

#include "RandomSeed.h"

#include <iostream>
#include <random>

void GetDefaultHeuristicParameters(HeuristicParameters* parameters) {

	// normal tuning
	/*parameters->m_values[PARAM_STILLALIVE] = 19354;
	parameters->m_values[PARAM_FREECELL] = 94;
	parameters->m_values[PARAM_CENTEROFMASS1] = 41;
	parameters->m_values[PARAM_CENTEROFMASS2] = 309;
	parameters->m_values[PARAM_CENTEROFMASS3] = 383;
	parameters->m_values[PARAM_CENTEROFMASS4] = 143;*/

	// rollback tuning
	parameters->m_values[PARAM_STILLALIVE] = 7260;
	parameters->m_values[PARAM_FREECELL] = 125;
	parameters->m_values[PARAM_CENTEROFMASS1] = 41;
	parameters->m_values[PARAM_CENTEROFMASS2] = 351;
	parameters->m_values[PARAM_CENTEROFMASS3] = 328;
	parameters->m_values[PARAM_CENTEROFMASS4] = 151;
}

struct Transposition {
	Board m_board;
	unsigned int m_score;
	Transposition() : m_board{0}, m_score{0} {}
};
struct ExpectiMaxContext {
	HeuristicParameters m_parameters;
	std::minstd_rand m_rng;
	unsigned int m_weight_table[16];
	Transposition m_transposition_tables[SEARCH_DEPTH][TRANSPOSITION_TABLE_SIZE];
};

template<unsigned int moves_left>
unsigned int ExpectiMaxPlayer(Board board, ExpectiMaxContext* context);

template<unsigned int moves_left>
inline __attribute__((always_inline))
void TurnComputer(Board board, ExpectiMaxContext* context, unsigned int& score_sum, unsigned int& score_div, unsigned int location, unsigned int value) {
	assert(GetCell(board, location) == 0);
	Board newboard = SetCell(board, location, value);
	score_sum += ExpectiMaxPlayer<moves_left>(newboard, context);
	++score_div;
}

template<unsigned int moves_left>
unsigned int ExpectiMaxComputer(Board board, ExpectiMaxContext* context) {

	// normalize the board
	board = Normalize(board);

	// check the transposition table
	unsigned int bucket = Hash1(board) & (TRANSPOSITION_TABLE_SIZE - 1);
	Transposition &tt = context->m_transposition_tables[moves_left][bucket];
	if(tt.m_board.m_data == board.m_data) {
		return tt.m_score;
	}

	// get all free locations
	unsigned int locations[16], location_count = 0;
	for(unsigned int location = 0; location < 16; ++location) {
		if(GetCell(board, location) == 0) {
			locations[location_count] = location;
			++location_count;
		}
	}
	assert(location_count > 0);

	// should we do an exhaustive search?
	unsigned int score;
	if(location_count < 4)
	{
		// check 2-tiles and 4-tiles
		unsigned int score1_sum = 0, score1_div = 0;
		unsigned int score2_sum = 0, score2_div = 0;
		for(unsigned int k = 0; k < location_count; ++k)
		{
			unsigned int location = locations[k];
			TurnComputer<moves_left>(board, context, score1_sum, score1_div, location, 1);
			TurnComputer<moves_left>(board, context, score2_sum, score2_div, location, 2);
		}
		unsigned int score_sum = score1_sum * 9 + score2_sum, score_div = score1_div * 9 + score2_div;
		assert(score_div != 0);
		score = (score_sum + score_div / 2) / score_div;
	}
	else
	{
		// check only 2-tiles, ignore 4-tiles
		unsigned int score_sum = 0, score_div = 0;
		for(unsigned int k = 0; k < location_count; ++k)
		{
			unsigned int location = locations[k];
			TurnComputer<moves_left>(board, context, score_sum, score_div, location, 1); // (context->m_rng() % 10 == 0)? 2 : 1
		}
		assert(score_div != 0);
		score = (score_sum + score_div / 2) / score_div;
	}

	tt.m_board = board;
	tt.m_score = score;

	return score;
}

template<>
unsigned int ExpectiMaxComputer<0>(Board board, ExpectiMaxContext* context)
{
	// check the transposition table
	unsigned int bucket = Hash1(board) & (TRANSPOSITION_TABLE_SIZE - 1);
	Transposition &tt = context->m_transposition_tables[0][bucket];
	if(tt.m_board.m_data == board.m_data) {
		return tt.m_score;
	}

	unsigned int score = context->m_parameters.m_values[PARAM_STILLALIVE];
	unsigned int freecell = context->m_parameters.m_values[PARAM_FREECELL];
	int wx = 0, wy = 0;

	for(unsigned int j = 0; j < 4; ++j) {
		for(unsigned int i = 0; i < 4; ++i) {
			unsigned int value = GetCell(board, i, j);
			if(value == 0) {
				score += freecell;
				freecell >>= 1;
			} else {
				unsigned int weight = context->m_weight_table[value];
				wx += ((int) i * 2 - 3) * (int) weight;
				wy += ((int) j * 2 - 3) * (int) weight;
			}
		}
	}

	score += (abs(wx) + abs(wy)) >> 10;

	tt.m_board = board;
	tt.m_score = score;

	return score;
}

template<unsigned int moves_left>
unsigned int ExpectiMaxPlayer(Board board, ExpectiMaxContext* context) {
	unsigned int best_score = 0;
	/*for(unsigned int direction = 0; direction < 4; ++direction) {
		BoardScore result = Collapse(board, (enum_direction) direction);
		if(result.m_board.m_data != board.m_data) {
			unsigned int score = result.m_score * SCORE_MULTIPLIER + ExpectiMaxComputer<moves_left - 1>(result.m_board, context);
			if(score > best_score) {
				best_score = score;
			}
		}
	}*/
	{
		BoardScore result = CollapseLeft(board);
		if(result.m_board.m_data != board.m_data) {
			unsigned int score = result.m_score * SCORE_MULTIPLIER + ExpectiMaxComputer<moves_left - 1>(result.m_board, context);
			if(score > best_score)
				best_score = score;
		}
	}
	{
		BoardScore result = CollapseRight(board);
		if(result.m_board.m_data != board.m_data) {
			unsigned int score = result.m_score * SCORE_MULTIPLIER + ExpectiMaxComputer<moves_left - 1>(result.m_board, context);
			if(score > best_score)
				best_score = score;
		}
	}
	board = Transpose(board);
	{
		BoardScore result = CollapseLeft(board);
		if(result.m_board.m_data != board.m_data) {
			unsigned int score = result.m_score * SCORE_MULTIPLIER + ExpectiMaxComputer<moves_left - 1>(result.m_board, context);
			if(score > best_score)
				best_score = score;
		}
	}
	{
		BoardScore result = CollapseRight(board);
		if(result.m_board.m_data != board.m_data) {
			unsigned int score = result.m_score * SCORE_MULTIPLIER + ExpectiMaxComputer<moves_left - 1>(result.m_board, context);
			if(score > best_score)
				best_score = score;
		}
	}
	return best_score;
}

std::pair<enum_direction, unsigned int> FindBestMove(Board board, const HeuristicParameters& parameters)
{
	ExpectiMaxContext context;
	context.m_parameters = parameters;
	context.m_rng.seed(RandomSeed());
	for(unsigned int i = 0; i < 16; ++i) {
		unsigned int weight3 = i * parameters.m_values[PARAM_CENTEROFMASS4];
		unsigned int weight2 = i * (parameters.m_values[PARAM_CENTEROFMASS3] + weight3);
		unsigned int weight1 = i * (parameters.m_values[PARAM_CENTEROFMASS2] + weight2);
		unsigned int weight0 = i * (parameters.m_values[PARAM_CENTEROFMASS1] + weight1);
		context.m_weight_table[i] = weight0;
	}

	unsigned int best_score = 0;
	enum_direction best_move = DIRECTION_NONE;
	for(unsigned int direction = 0; direction < 4; ++direction)
	{
		BoardScore result = Collapse(board, (enum_direction) direction);
		if(result.m_board.m_data != board.m_data) {
			unsigned int score = result.m_score * SCORE_MULTIPLIER + ExpectiMaxComputer<SEARCH_DEPTH - 1>(result.m_board, &context);
			if(score > best_score || best_move == DIRECTION_NONE)
			{
				best_score = score;
				best_move = (enum_direction) direction;
			}
		}
	}

	return std::make_pair(best_move, best_score);
}
