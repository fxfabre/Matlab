/*
Copyright (c) 2014 Maarten Baert <maarten-baert@hotmail.com>

Permission to use, copy, modify, and/or distribute this software for any purpose
with or without fee is hereby granted, provided that the above copyright notice
and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
THIS SOFTWARE.
*/

#include "Board.h"
#include "PerfTests.h"
#include "PlayTests.h"
#include "Analyze.h"

#include <iostream>

int main() {

	//Test_CollapseRow();
	//Test_FlipBoard();
	//Test_CollapseBoard();
	//Test_NormalizeBoard();
	//Test_HashBoard();

	/*HeuristicParameters parameters;
	GetDefaultHeuristicParameters(&parameters);
	PlayTest(parameters);*/

	//PlayTest_Tune(10, 10000, 100, 10, 30);
	PlayTest_Batch(20, 5, NULL);

	/*BoardDB boarddb;
	boarddb.Load();
	PlayTest_Batch(0, 4000, &boarddb);
	boarddb.Save();*/

	/*BoardDB boarddb;
	boarddb.Load();
	Analyze_Test3(&boarddb);*/

	std::cout << "Done." << std::endl;
	return 0;
}
