#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

// 6144 bytes should be able to capture the largest of solvable boards like 26x26.
// Being unable to read line by line is hard if have to read multiple times and take care of cases
// when newlines are not aligned to buffer.
// Bigger sizes not really tractable and this program is not intended to solve arbitrarily big boards.
#define INPUT_BUF_LEN 6144

// These 2 allows for board and occupied mask to be condensed into 1 array.
// This is so that sending a board to slave only takes 1 communication.
#define BOARD(r) (r)
#define TAKEN(r) ((r)+R)

// Convenience macros to get row and column numbers from the given position
#define ROW(pos) ((pos)/C)
#define COL(pos) ((pos)%C)

// Move encoding
#define DIRBITS 8
#define DOWNRIGHTMASK 1
#define DOWNLEFTMASK (DOWNRIGHTMASK<<1)
#define UPRIGHTMASK (DOWNLEFTMASK<<1)
#define UPLEFTMASK (UPRIGHTMASK<<1)
#define RIGHTMASK (UPLEFTMASK<<1)
#define LEFTMASK (RIGHTMASK<<1)
#define DOWNMASK (LEFTMASK<<1)
#define UPMASK (DOWNMASK<<1)
#define POS(move) ((move)>>DIRBITS)
#define DIRMASK(move) ((move)&((1<<DIRBITS)-1))
#define ENCODEMOVE(pos, dirs) (((pos)<<DIRBITS)|(dirs))

#define MASTER_PID (numProcs-1)
#define WHITE 0
#define BLACK 1
#define MINALPHA (-DBL_MAX)
#define MAXBETA DBL_MAX
#define STARTMOVEIDX(sid, nMoves, nSlaves) ((sid)*(nMoves)/(nSlaves))
#define MAX(a, b) ((a)>(b)? (a): (b))
#define MIN(a, b) ((a)<(b)? (a): (b))

/* Function declarations */
double elapsedTime(struct timeval, struct timeval);
int initBoard(char *boardfile, char *paramfile);
void deinitBoard();
void printBoard(int brd[const]);
void printOutput(const int bestMove);
void applyMove(int destbrd[], int srcbrd[const], const int move, const int color);
int isLegalMove(int brd[const], const int pos, const int color);
int getLegalMoves(int brd[const], const int color, int moves[]);
double evaluateBoard(int brd[const]);
double alphabeta(int brd[const], const int depth, const int color, const int passed, double alpha, double beta);
/* End function declarations */

int R = -1, C = -1, pid, numProcs, lowestDepth, pruned, tempPruned, numBoards, numMoves;
double bestAlpha, tempAlpha, *scores;
int MAXDEPTH, MAXBOARDS, CORNERVALUE, EDGEVALUE, COLOR, TIMEOUT;
int *board, bestMove, *legalMoves;
struct timeval starttime, endtime;

/* Position labels in program, different from input
 * 210	[a3][b3][c3]
 * 543	[a2][b2][c2]
 * 876	[a1][b1][c1]
 */

int main(int argc, char **argv) {
 	int res, i, *brdcpy;
 	double bestScore;

 	// Make sure the input files are given
 	if(argc<3){
		printf("Format: ./othellox-serial <board_file> <eval_params_file>\n");
		return 1;
	}

	gettimeofday(&starttime, NULL);

 	// Here, assume all operations are successful.
 	// Failure handling makes the code too complicated and deviates away from
 	// developing program logic.
 	res = initBoard(argv[1], argv[2]);	// Initialize board
 	legalMoves = malloc(R*C*sizeof(int));

 	// All compute valid first moves
 	numMoves = getLegalMoves(board, COLOR, legalMoves);

 	if(numMoves){
 		scores = malloc(numMoves*sizeof(double));
 		bestScore = MINALPHA; bestMove = -1;
 		brdcpy = malloc(R*2*sizeof(int));
 		for(i=0; i<numMoves; i++) {
 			applyMove(brdcpy, board, legalMoves[i], COLOR);
 			scores[i] = alphabeta(brdcpy, MAXDEPTH-1, !COLOR, 0, MINALPHA, MAXBETA);
 			if(scores[i]>bestScore){ bestScore = scores[i]; bestMove = legalMoves[i]; }
 		}
 		free(scores);
 	} else {	// No legal move
		bestMove = -1; numBoards = 1;
	}

	gettimeofday(&endtime, NULL);

	printOutput(bestMove);

 	// Cleanup
 	free(legalMoves);
 	deinitBoard();

 	return 0;
 }

// Returns a bitmap of valid directions. Value of 0 means no valid direction.
// Assumes C<32.
// Directions are: up, down, left, right, upleft, upright, downleft, downright
int isLegalMove(int brd[const], const int pos, const int color) {
	int r = ROW(pos), c = COL(pos), res = 0, mask = 1<<c, r2, c2, captureCount;

	// Position already taken, not legal move at all!
	if(brd[TAKEN(r)]&mask) return 0;

	// Check up
	for(r2=r-1, mask=1<<c, captureCount=0; r2>=0; r2--) {
		if(!(brd[TAKEN(r2)]&mask)) { captureCount = 0; break; }	// Not taken, cannot capture
		if(color==!!(brd[BOARD(r2)]&mask)) break;	// Captured 0 or more
		captureCount++;
	}
	if(r2>=0 && captureCount>0) res|=UPMASK;

	// Check down
	for(r2=r+1, mask=1<<c, captureCount=0; r2<R; r2++) {
		if(!(brd[TAKEN(r2)]&mask)) { captureCount = 0; break; }	// Not taken, cannot capture
		if(color==!!(brd[BOARD(r2)]&mask)) break;	// Captured 0 or more
		captureCount++;
	}
	if(r2<R && captureCount>0) res|=DOWNMASK;

	// Check left
	for(mask=1<<(c+1); mask<(1<<C); mask<<=1) {
		if(!(brd[TAKEN(r)]&mask)) { captureCount = 0; break; }	// Not taken, cannot capture
		if(color==!!(brd[BOARD(r)]&mask)) break;	// Captured 0 or more
		captureCount++;
	}
	if(mask<(1<<C) && captureCount>0) res|=LEFTMASK;

	// Check right
	for(mask=1<<(c-1); mask>0; mask>>=1) {
		if(!(brd[TAKEN(r)]&mask)) { captureCount = 0; break; }	// Not taken, cannot capture
		if(color==!!(brd[BOARD(r)]&mask)) break;	// Captured 0 or more
		captureCount++;
	}
	if(mask>0 && captureCount>0) res|=RIGHTMASK;

	// Check upleft
	for(r2=r-1, mask=1<<(c+1); r2>=0 && mask<(1<<C); r2--, mask<<=1) {
		if(!(brd[TAKEN(r2)]&mask)) { captureCount = 0; break; }	// Not taken, cannot capture
		if(color==!!(brd[BOARD(r2)]&mask)) break;	// Captured 0 or more
		captureCount++;
	}
	if(r2>=0 && mask<(1<<C) && captureCount>0) res|=UPLEFTMASK;

	// Check upright
	for(r2=r-1, mask=1<<(c-1); r2>=0 && mask>0; r2--, mask>>=1) {
		if(!(brd[TAKEN(r2)]&mask)) { captureCount = 0; break; }	// Not taken, cannot capture
		if(color==!!(brd[BOARD(r2)]&mask)) break;	// Captured 0 or more
		captureCount++;
	}
	if(r2>=0 && mask>0 && captureCount>0) res|=UPRIGHTMASK;

	// Check downleft
	for(r2=r+1, mask=1<<(c+1); r2<R && mask<(1<<C); r2++, mask<<=1) {
		if(!(brd[TAKEN(r2)]&mask)) { captureCount = 0; break; }	// Not taken, cannot capture
		if(color==!!(brd[BOARD(r2)]&mask)) break;	// Captured 0 or more
		captureCount++;
	}
	if(r2<R && mask<(1<<C) && captureCount>0) res|=DOWNLEFTMASK;

	// Check downright
	for(r2=r+1, mask=1<<(c-1); r2<R && mask>0; r2++, mask>>=1) {
		if(!(brd[TAKEN(r2)]&mask)) { captureCount = 0; break; }	// Not taken, cannot capture
		if(color==!!(brd[BOARD(r2)]&mask)) break;	// Captured 0 or more
		captureCount++;
	}
	if(r2<R && mask>0 && captureCount>0) res|=DOWNRIGHTMASK;
	return res;
}

// Returns number of legal moves found.
// The moves array will be filled with legal moves found.
// The last 8 bits of each move indicate the flippable directions for that move.
// Pre-condition: moves must be large enough to hold any possible number of moves if not null.
int getLegalMoves(int brd[const], const int color, int moves[]) {
	int pos, numMoves = 0, dir;
	for(pos=0; pos<R*C; pos++)
		if(dir = isLegalMove(brd, pos, color)) {
			if(moves) moves[numMoves++] = ENCODEMOVE(pos, dir);
			else numMoves++;
		}
	return numMoves;
}

// Generate new board by applying the given move to the given board.
// Assumes a valid board.
void applyMove(int destbrd[], int srcbrd[const], const int move, const int color) {
	if(POS(move)<0 || POS(move)>=R*C){ printf("Invalid position found\n"); return; }
	int pos = POS(move), dirs = DIRMASK(move), r = ROW(pos), c = COL(pos), mask = 1<<c, r2;

	// Copy srcbrd over so we can flip bits later
	memcpy((void*)destbrd, (void*)srcbrd, (R<<1)*sizeof(int));

	// If position already occupied, illegal move! Do nothing.
	if(destbrd[TAKEN(r)]&mask){ printf("Taking occupied cell\n"); return; }

	// Place the correct color disc on the specified position.
	destbrd[TAKEN(r)]|=mask;
	if(color==BLACK) destbrd[BOARD(r)]|=mask;

	// Scan all valid directions and flip discs. Assume dirs is properly set up.
	// So it is always possible to find back same color without going out-of-bounds or
	// encountering empty cell on VALID board.
	if(dirs&UPMASK)
		for(r2=r-1; color!=!!(destbrd[BOARD(r2)]&mask); r2--) destbrd[BOARD(r2)]^=mask;

	if(dirs&DOWNMASK)
		for(r2=r+1; color!=!!(destbrd[BOARD(r2)]&mask); r2++) destbrd[BOARD(r2)]^=mask;

	if(dirs&LEFTMASK)
		for(mask=1<<(c+1); color!=!!(destbrd[BOARD(r)]&mask); mask<<=1)
			destbrd[BOARD(r)]^=mask;

	if(dirs&RIGHTMASK)
		for(mask=1<<(c-1); color!=!!(destbrd[BOARD(r)]&mask); mask>>=1)
			destbrd[BOARD(r)]^=mask;

	if(dirs&UPLEFTMASK)
		for(r2=r-1, mask=1<<(c+1); color!=!!(destbrd[BOARD(r2)]&mask); r2--, mask<<=1)
			destbrd[BOARD(r2)]^=mask;

	if(dirs&UPRIGHTMASK)
		for(r2=r-1, mask=1<<(c-1); color!=!!(destbrd[BOARD(r2)]&mask); r2--, mask>>=1)
			destbrd[BOARD(r2)]^=mask;

	if(dirs&DOWNLEFTMASK)
		for(r2=r+1, mask=1<<(c+1); color!=!!(destbrd[BOARD(r2)]&mask); r2++, mask<<=1)
			destbrd[BOARD(r2)]^=mask;

	if(dirs&DOWNRIGHTMASK)
		for(r2=r+1, mask=1<<(c-1); color!=!!(destbrd[BOARD(r2)]&mask); r2++, mask>>=1)
			destbrd[BOARD(r2)]^=mask;
}

// Computes heuristic score for the given board.
// Scores are based on maximizing player - minimizing player for each heuristic.
// Assume values will fit into 32-bit signed integer.
double evaluateBoard(int brd[const]) {
	double score = 0; int totalDiscs = 0, numBlack = 0, numWhite = 0, r, mask = (1<<C)-1;

	// Compute difference in number of discs controlled
	for(r=0; r<R; r++) {
		totalDiscs+=__builtin_popcount(brd[TAKEN(r)]&mask);
		numBlack+=__builtin_popcount(brd[BOARD(r)]&brd[TAKEN(r)]&mask);
	}
	numWhite = totalDiscs-numBlack;
	if(totalDiscs) score = (COLOR==WHITE? -1: 1)*(numBlack-numWhite)/(double)totalDiscs;

	// Compute difference in number of legal moves
	numBlack = getLegalMoves(brd, BLACK, NULL);
	numWhite = getLegalMoves(brd, WHITE, NULL);
	if(numBlack+numWhite) score+=(COLOR==WHITE? -1: 1)*10.0*(numBlack-numWhite)/(numBlack+numWhite);

	// Compute difference in corner values
	mask = (1<<(C-1))|1;
	totalDiscs = __builtin_popcount(mask&brd[TAKEN(0)])+__builtin_popcount(mask&brd[TAKEN(R-1)]);
	numBlack = __builtin_popcount(mask&brd[TAKEN(0)]&brd[BOARD(0)])+__builtin_popcount(mask&brd[TAKEN(R-1)]&brd[BOARD(R-1)]);
	numWhite = totalDiscs-numBlack;
	if(totalDiscs) score+=(COLOR==WHITE? -1: 1)*CORNERVALUE*(numBlack-numWhite)/(double)totalDiscs;

	// Compute difference in edge values
	totalDiscs = 0; numBlack = 0;
	for(r=1; r<R-1; r++) {	// For vertical edges
		totalDiscs+=__builtin_popcount(mask&brd[TAKEN(r)]);
		numBlack+=__builtin_popcount(mask&brd[TAKEN(r)]&brd[BOARD(r)]);
	}
	mask^=((1<<C)-1);	// Horizontal edges
	totalDiscs+=__builtin_popcount(mask&brd[TAKEN(0)])+__builtin_popcount(mask&brd[TAKEN(R-1)]);
	numBlack+=__builtin_popcount(mask&brd[TAKEN(0)]&brd[BOARD(0)])+__builtin_popcount(mask&brd[TAKEN(R-1)]&brd[BOARD(R-1)]);
	numWhite = totalDiscs-numBlack;
	if(totalDiscs) score+=(COLOR==WHITE? -1: 1)*EDGEVALUE*(numBlack-numWhite)/(double)totalDiscs;

	return score;
}

// This is actually done on slaves
// TODO: Implement
double alphabeta(int brd[const], const int depth, const int color, const int passed, double alpha, double beta) {
	int *tempMoves, *brdcpy, nMoves, i, isMaxPlayer = color==COLOR;
	double res = isMaxPlayer? MINALPHA: MAXBETA;
	numBoards++; lowestDepth = MIN(lowestDepth, depth);

	printf("Process %d (%d, %d, %d)\n", pid, depth, color, passed);
	printf("%d\n", isMaxPlayer);


	if(!depth) return evaluateBoard(brd);	// Leaf node

	tempMoves = malloc(R*C*sizeof(int));
	brdcpy = malloc((R<<1)*sizeof(int));
	nMoves = getLegalMoves(brd, color, tempMoves);
	printf("Depth %d %d moves\n", depth, nMoves);

	if(!nMoves && passed) res = evaluateBoard(brd);	// End game
	else if(!nMoves) res = alphabeta(brd, depth-1, !color, 1, alpha, beta);
	else for(i=0; i<nMoves; i++) {
		printf("Depth %d i %d nMoves %d\n", depth, i, nMoves);
		printf("Applying move at %d\n", POS(tempMoves[i]));
		applyMove(brdcpy, brd, tempMoves[i], color);
		if(isMaxPlayer) {
			res = MAX(res, alphabeta(brdcpy, depth-1, !color, 0, alpha, beta));
			alpha = MAX(alpha, res);
		} else {
			res = MIN(res, alphabeta(brdcpy, depth-1, !color, 0, alpha, beta));
			beta = MIN(beta, res);
		}
		if(beta<=alpha) { pruned|=(i+1<nMoves); break; }
	}

	free(tempMoves); free(brdcpy);
	return res;
}

// Precondition: numBestMoves and bestMoves must be properly set up.
void printOutput(const int bestMove) {
	int r, i; char c;
	printf("Best moves: { ");
	if(bestMove<0) printf("na");
	else {
		r = R-ROW(POS(bestMove)); c = C-1-COL(POS(bestMove))+'a';
		printf("%c%d", c, r);
	}
	printf(" }\n");
	printf("Number of boards assessed: %llu\n", numBoards);
	printf("Depth of boards: %d\n", MAXDEPTH-lowestDepth);
	printf("Entire space: "); printf(pruned? "false\n": "true\n");
	printf("Elapsed time in seconds: %lf\n", elapsedTime(starttime, endtime));
}

// Returns elpased time in millseconds
double elapsedTime(struct timeval start, struct timeval end) {
	double timetaken = end.tv_sec-start.tv_sec;
	if(end.tv_usec<start.tv_usec) { timetaken-=1; end.tv_usec+=1000000; }
	return timetaken+(end.tv_usec-start.tv_usec)/1000000.0;
}

int initBoard(char *boardfile, char *paramfile) {
 	FILE *brdfile = fopen(boardfile, "r"), *evalfile = fopen(paramfile, "r");
 	char buf[1024], *ptok; int len, r, c;

 	// Read board size
	fgets(buf, 1024, brdfile); len = strlen(buf)-1;
	while(len>=0 && !isdigit(buf[len])) buf[len--] = 0;
	ptok = strtok(buf+6, ",");
	while(ptok!=NULL){
		if(C==-1) C = atoi(ptok);
		else if(R==-1) R = atoi(ptok);
		else break;
	}
	board = malloc(R*2*sizeof(int));
	memset(board, 0, R*2*sizeof(int));

	// Read and process white positions
	fgets(buf, 1024, brdfile); len = strlen(buf)-1;
	while(len>=0 && !isalnum(buf[len])) buf[len--] = 0;
	ptok = strtok(buf+9, ",");
	while(ptok!=NULL){
		c = C-1-(ptok[0]-'a'); r = R-atoi(ptok+1); ptok = strtok(NULL, ",");
		board[TAKEN(r)]|=(1<<c);
	}

	// Read and process black positions
	fgets(buf, 1024, brdfile); len = strlen(buf)-1;
	while(len>=0 && !isalnum(buf[len])) buf[len--] = 0;
	ptok = strtok(buf+9, ",");
	while(ptok!=NULL){
		c = C-1-(ptok[0]-'a'); r = R-atoi(ptok+1); ptok = strtok(NULL, ",");
		board[TAKEN(r)]|=(1<<c); board[BOARD(r)]|=(1<<c);
	}

	// Read color value
	fgets(buf, 1024, brdfile); len = strlen(buf)-1;
	while(len>=0 && !isalnum(buf[len])) buf[len--] = 0;
	COLOR = !strcmp(buf+7, "White")? WHITE: BLACK;

	// Read timeout value
	fgets(buf, 1024, brdfile); len = strlen(buf)-1;
	while(len>=0 && !isalnum(buf[len])) buf[len--] = 0;
	TIMEOUT = atoi(buf+9);
	fclose(brdfile);

	// Read max depth
	fgets(buf, 1024, evalfile); len = strlen(buf)-1;
	while(len>=0 && !isdigit(buf[len])) buf[len--] = 0;
	MAXDEPTH = atoi(buf+10);

	// Read max boards
	fgets(buf, 1024, evalfile); len = strlen(buf)-1;
	while(len>=0 && !isdigit(buf[len])) buf[len--] = 0;
	MAXBOARDS = atoi(buf+11);

	// Read corner value
	fgets(buf, 1024, evalfile); len = strlen(buf)-1;
	while(len>=0 && !isdigit(buf[len])) buf[len--] = 0;
	CORNERVALUE = atoi(buf+13);

	// Read edge value
	fgets(buf, 1024, evalfile); len = strlen(buf)-1;
	while(len>=0 && !isdigit(buf[len])) buf[len--] = 0;
	EDGEVALUE = atoi(buf+11);
	fclose(evalfile);
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