#include <mpi.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 6144 bytes should be able to capture the largest of solvable boards like 26x26.
// Being unable to read line by line is hard if have to read multiple times and take care of cases
// when newlines are not aligned to buffer.
// Bigger sizes not really tractable and this program is not intended to solve arbitrarily big boards.
#define INPUT_BUF_LEN 6144

// These 2 allows for board and occupied mask to be condensed into 1 array.
// This is so that sending a board to slave only takes 1 communication.
#define BOARD(r) (r)
#define TAKEN(r) (r+R)

// Convenience macros to get row and column numbers from the given position
#define ROW(pos) (pos/C)
#define COL(pos) (pos%C)

#define MASTER_PID 0
#define WHITE 0
#define BLACK 1
#define MAX(a, b) (a>b? a: b)
#define MIN(a, b) (a<b? a: b)

int initBoard(char*, char*);
void deinitBoard();
void printBoard(int brd[const]);

int R = -1, C = -1, pid, numProcs, MAXDEPTH, MAXBOARDS, CORNERVALUE, EDGEVALUE, COLOR, TIMEOUT, *board;

/* Position labels in program, different from input
 * 210	[a3][b3][c3]
 * 543	[a2][b2][c2]
 * 876	[a1][b1][c1]
 */

 int main(int argc, char **argv) {
 	int res;

 	// Make sure the input files are given
 	if(argc<3){
		printf("Format: ./othellox-serial <board_file> <eval_params_file>\n");
		return 1;
	}

 	MPI_Init(&argc, &argv);
 	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
 	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

 	// Master process, compute 1 branch to get alpha beta bounds
 	// to help slaves with cut-off
 	if(pid==MASTER_PID) {
 		res = initBoard(argv[1], argv[2]);	// Initialize board
 		MPI_Bcast((void *)&res, 1, MPI_INT, MASTER_PID, MPI_COMM_WORLD);
 		if(res) printf("initBoard failed. Exiting master %d\n", pid);
 		else {
 			printf("Successfully initBoard. Resuming master %d\n", pid);
 		}
 		// printBoard(board);
 		deinitBoard();
 	} else {	// Slave just run original minimax with alpha-beta pruning.
 		// Wait for initialization result of master
 		MPI_Bcast((void *)&res, 1, MPI_INT, MASTER_PID, MPI_COMM_WORLD);
 		if(res) printf("initBoard failed. Exiting slave %d\n", pid);
 		else {
 			printf("Successfully initboard. Starting slave %d\n", pid);
 		}
 	}
 	MPI_Finalize();
 	return 0;
 }

 int initBoard(char *boardfile, char *paramfile) {
 	char buf[INPUT_BUF_LEN], *ptok, *cur, *start; int len, r, c, err;
 	MPI_File brdfp, paramsfp;
 	MPI_Status status;

 	// Open board file
 	if(err = MPI_File_open(MPI_COMM_SELF, boardfile, MPI_MODE_RDONLY, MPI_INFO_NULL, &brdfp)) {
 		printf("Unable to open board file\n");
 		return -1;
 	}

 	// Read in board file content
 	memset(buf, 0, INPUT_BUF_LEN*sizeof(char));
 	if(err = MPI_File_read(brdfp, buf, INPUT_BUF_LEN, MPI_CHAR, &status)) {
 		printf("Failed to read board file data\n");
 		MPI_File_close(&brdfp);
 		return -1;
 	}

 	// Read numColumns and numRows
 	start = buf+6; cur = strchr(start, '\n');	// 1st line is Size: C,R
 	for(ptok=cur; !isalnum(*ptok) && ptok>=start; ptok--) *ptok = 0;	// Erase non-alnum characters
 	ptok = strtok(start, ",");
 	while(ptok!=NULL) {
		if(C==-1) C = atoi(ptok);
		else if(R==-1) R = atoi(ptok);
		else break;
	}
	board = malloc(R*2*sizeof(int)); memset(board, 0, R*2*sizeof(int));

	// Read white positions
	start = cur+10; cur = strchr(start, '\n'); // 2nd line format is White: { <moves>,... }
	for(ptok=cur; !isalnum(*ptok) && ptok>=start; ptok--) *ptok = 0;	// Erase trailing non-alnum in line
	ptok = strtok(start, ",");
	while(ptok!=NULL) {
		c = C-1-(ptok[0]-'a'); r = R-atoi(ptok+1); ptok = strtok(NULL, ",");
		board[TAKEN(r)]|=(1<<c);	// White is bit value 0 => do nothing to actual board
	}

	// Read black positions
	start = cur+10; cur = strchr(start, '\n'); // 2nd line format is Black: { <moves>,... }
	for(ptok=cur; !isalnum(*ptok) && ptok>=start; ptok--) *ptok = 0;	// Erase trailing non-alnum in line
	ptok = strtok(start, ",");
	while(ptok!=NULL) {
		c = C-1-(ptok[0]-'a'); r = R-atoi(ptok+1); ptok = strtok(NULL, ",");
		board[TAKEN(r)]|=(1<<c); board[BOARD(r)]|=(1<<c);
	}	

 	MPI_File_close(&brdfp);	// Done with board file. Close it.

 	// Open params file for reading
 	if(err = MPI_File_open(MPI_COMM_SELF, paramfile, MPI_MODE_RDONLY, MPI_INFO_NULL, &paramsfp)) {
 		printf("Unable to open params file\n");
 		return -1;
 	}

 	// Read in the file content
 	memset(buf, 0, INPUT_BUF_LEN*sizeof(char));
 	if(err = MPI_File_read(paramsfp, buf, INPUT_BUF_LEN, MPI_CHAR, &status)) {
 		printf("Failed to read params file\n");
 		MPI_File_close(&paramsfp);
 		return -1;
 	}

 	// Read max depth
 	start = buf+10; cur = strchr(start, '\n');	// 1st line is MaxDepth: <maxdepth>
 	for(ptok=cur; !isalnum(*ptok) && ptok>=start; ptok--) *ptok = 0;
 	MAXDEPTH = atoi(start);

 	// Read max boards
 	start = cur+12; cur = strchr(start, '\n');	// 2nd line is MaxBoards: <maxboards>
 	for(ptok=cur; !isalnum(*ptok) && ptok>=start; ptok--) *ptok = 0;
 	MAXBOARDS = atoi(start);

 	// Read corner value
 	start = cur+14; cur = strchr(start, '\n');	// 3rd line is CornerValue: <cornervalue>
 	for(ptok=cur; !isalnum(*ptok) && ptok>=start; ptok--) *ptok = 0;
 	CORNERVALUE = atoi(start);

 	// Read edge value
 	start = cur+12; cur = strchr(start, '\n');	// 4th line is EdgeValue: <edgevalue>
 	for(ptok=cur; !isalnum(*ptok) && ptok>=start; ptok--) *ptok = 0;
 	EDGEVALUE = atoi(start);

 	// Read playing color
 	start = cur+7; cur = strchr(start, '\n');	// 5th line is Color: <White|Black>
 	for(ptok=cur; !isalnum(*ptok) && ptok>=start; ptok--) *ptok = 0;
 	COLOR = !strcmp(start, "WHITE")? WHITE: BLACK;

 	// Read timeout value
 	start = cur+9; cur = strchr(start, '\n');	// 6th line is Timeout: <timeout>
 	for(ptok=cur; !isalnum(*ptok) && ptok>=start; ptok--) *ptok = 0;
 	TIMEOUT = atoi(start);

 	printf("%d %d\n", COLOR, TIMEOUT);

 	MPI_File_close(&paramsfp);	// Done with params file. Close it.

 	return 0;
 }

 void deinitBoard() {
 	if(board!=NULL) free(board);
 	board = NULL;
 }

 void printBoard(int brd[const]) {
 	int r;
 	printf("Board:\n");
 	for(r=0; r<R; r++) printf("%d\n", brd[BOARD(r)]&brd[TAKEN(r)]);
 	printf("Board occupancy\n");
 	for(r=0; r<R; r++) printf("%d\n", brd[TAKEN(r)]);
 }