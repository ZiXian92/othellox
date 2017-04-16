#include <ctype.h>
#include <float.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
#define TIMEOUT_THRESH (TIMEOUT-1)
#define MAX(a, b) ((a)>(b)? (a): (b))
#define MIN(a, b) ((a)<(b)? (a): (b))

// MPI Stuff
#define NEW_ALPHA_TAG 0

/* Function declarations */
double elapsedTime(struct timespec, struct timespec);
int initBoard(char *boardfile, char *paramfile);
void deinitBoard();
void printBoard(int brd[const]);
void printOutput(const int bestMove);
void applyMove(int destbrd[], int srcbrd[const], const int move, const int color);
int isLegalMove(int brd[const], const int pos, const int color);
int getLegalMoves(int brd[const], const int color, int moves[]);
double evaluateBoard(int brd[const]);
double masteralphabeta(int brd[const], const int depth, const int color, const int passed);
double alphabeta(int brd[const], const int depth, const int color, const int passed, double alpha, double beta);
void masterProcess();
void slaveProcess(const int startMoveIdx, const int endMoveIdx);	// Processes legalMoves[startMoveIdx..(endMoveIdx-1)]
/* End function declarations */

int R = -1, C = -1, pid, numProcs, lowestDepth, pruned, tempPruned, numBoards, numMoves, totalBoards;
double bestAlpha, tempAlpha, *scores, timetaken;
int MAXDEPTH, MAXBOARDS, CORNERVALUE, EDGEVALUE, COLOR, TIMEOUT;
int *board, bestMove, *legalMoves, shouldStop;
struct timespec starttime, endtime;
MPI_Request alphaReq, stopSigReq, boardCountReq, scoresReq, alphabcastReq;
int alphaReqFlag, stopSigReqFlag, boardCountReqFlag, scoresReqFlag, alphabcastReqFlag;
MPI_Comm alphaChannel, stopChannel, alphabcastChannel;

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

	clock_gettime(CLOCK_MONOTONIC, &starttime);

 	MPI_Init(&argc, &argv);
 	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
 	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
 	MPI_Comm_dup(MPI_COMM_WORLD, &alphaChannel);
 	MPI_Comm_dup(MPI_COMM_WORLD, &stopChannel);
 	MPI_Comm_dup(MPI_COMM_WORLD, &alphabcastChannel);

 	// Here, assume all operations are successful.
 	// Failure handling makes the code too complicated and deviates away from
 	// developing program logic.
 	res = initBoard(argv[1], argv[2]);	// Initialize board
 	legalMoves = malloc(R*C*sizeof(int));

 	// All compute valid first moves
 	numMoves = getLegalMoves(board, COLOR, legalMoves);

 	// Master process, compute 1 branch to get alpha beta bounds
 	// to help slaves with cut-off
 	if(pid==MASTER_PID) {
 		masterProcess();
 	} else {	// Slave just run original minimax with alpha-beta pruning.
 		slaveProcess(STARTMOVEIDX(pid, numMoves, numProcs-1), STARTMOVEIDX(pid+1, numMoves, numProcs-1));
 	}

 	// Cleanup
 	free(legalMoves);
 	deinitBoard();

 	MPI_Finalize();
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

// Coordination work by master process
void masterProcess() {
	// Assume board is not too big that storing all positions is infeasible
	// Assume successful allocation
	int *recvcounts = malloc(numProcs*sizeof(int)), *displs = malloc(numProcs*sizeof(int)), i;
	double bestScore;

	numBoards = 0; lowestDepth = MAXDEPTH; pruned = 0; shouldStop = 0;

	if(numMoves) {
		scores = malloc(numMoves*sizeof(double));

		// Find initial alpha and beta values and broadcast to all slaves
		// in hope of faster pruning.
		bestAlpha = masteralphabeta(board, MAXDEPTH, COLOR, 0);
		MPI_Bcast(&bestAlpha, 1, MPI_DOUBLE, MASTER_PID, alphabcastChannel);


		// Leave a request open to gather best results for all legal moves
		// Compute how many scores to receive from each slave
		recvcounts[numProcs-1] = displs[numProcs-1] = 0;
		for(i=0; i<numProcs-1; i++) {
			displs[i] = STARTMOVEIDX(i, numMoves, numProcs-1);
			recvcounts[i] = STARTMOVEIDX(i+1, numMoves, numProcs-1)-displs[i];
		}
		MPI_Igatherv(NULL, 0, MPI_DOUBLE, scores, recvcounts, displs, MPI_DOUBLE, MASTER_PID, MPI_COMM_WORLD, &scoresReq);

		// TODO: Implement loop to receive and broadcast updated alpha values
		MPI_Test(&scoresReq, &scoresReqFlag, MPI_STATUS_IGNORE);
		clock_gettime(CLOCK_MONOTONIC, &endtime);
		while(!scoresReqFlag && (timetaken = elapsedTime(starttime, endtime))<TIMEOUT_THRESH) {
			// MPI_Irecv(&tempAlpha, 1, MPI_INT, MPI_ANY_SOURCE, NEW_ALPHA_TAG, MPI_COMM_WORLD, &alphaReq);
			// MPI_Iscan(&numBoards, &totalBoards, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD, &boardCountReq);
			// printf("Time taken: %lfs\n", timetaken);

			MPI_Test(&scoresReq, &scoresReqFlag, MPI_STATUS_IGNORE);
			clock_gettime(CLOCK_MONOTONIC, &endtime);
		}

		shouldStop = 1;

		// Time threshold exceeded, tell everyone to stop
		MPI_Ibcast(&shouldStop, 1, MPI_INT, MASTER_PID, stopChannel, &stopSigReq);
		MPI_Wait(&scoresReq, MPI_STATUS_IGNORE); // Wait for all to send scores
		
		// Here, all scores should be ready for processing.
		// Process scores to get best move(s)
		// for(i=0; i<numMoves; i++) printf(" %lf", scores[i]); printf("\n");
		bestScore = MINALPHA; bestMove = -1;
		for(i=0; i<numMoves; i++) if(bestScore<scores[i]) { bestScore = scores[i]; bestMove = legalMoves[i]; }

		// Gather other statistics
		MPI_Scan(&numBoards, &numBoards, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		MPI_Scan(&pruned, &pruned, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
		MPI_Scan(&lowestDepth, &lowestDepth, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

		// Cleanup variables in intermediate computation
		free(scores); free(recvcounts); free(displs);
	} else {	// No legal move
		bestMove = -1; numBoards = 1;
	}

	clock_gettime(CLOCK_MONOTONIC, &endtime);

	printOutput(bestMove);	// Print output
}

// Work done by each slave
void slaveProcess(const int startMoveIdx, const int endMoveIdx) {
	int i, *brdcpy;

	numBoards = 0; lowestDepth = MAXDEPTH; pruned = 0; shouldStop = 0;

	// Open a channel to receive stop signal.
	MPI_Ibcast(&shouldStop, 1, MPI_INT, MASTER_PID, stopChannel, &stopSigReq);
	MPI_Test(&stopSigReq, &stopSigReqFlag, MPI_STATUS_IGNORE);

	if(startMoveIdx>=endMoveIdx) {
		while(!stopSigReqFlag) {
			// MPI_Iscan(&numBoards, &totalBoards, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD, &boardCountReq);
			MPI_Test(&stopSigReq, &stopSigReqFlag, MPI_STATUS_IGNORE);
		}

		MPI_Igatherv(NULL, 0, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, MASTER_PID, MPI_COMM_WORLD, &scoresReq);
		MPI_Scan(&numBoards, &numBoards, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		MPI_Scan(&pruned, &pruned, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
		MPI_Scan(&lowestDepth, &lowestDepth, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
		return;
	}

	scores = malloc((endMoveIdx-startMoveIdx)*sizeof(double));	// Prepare holder for each move's score
	brdcpy = malloc((R<<1)*sizeof(int));
	memset(scores, 0, (endMoveIdx-startMoveIdx)*sizeof(double));

	// Receive initial alpha and beta from master
	MPI_Bcast(&bestAlpha, 1, MPI_DOUBLE, MASTER_PID, alphabcastChannel);

	for(i=startMoveIdx; i<endMoveIdx; i++) {
		// Check if master has issued a termination order
		MPI_Test(&stopSigReq, &stopSigReqFlag, MPI_STATUS_IGNORE);
		if(stopSigReqFlag) break;

		// Safe to continue search for now
		applyMove(brdcpy, board, legalMoves[i], COLOR);	// Apply given move to evaluate
		scores[i-startMoveIdx] = alphabeta(brdcpy, MAXDEPTH-1, !COLOR, 0, bestAlpha, MAXBETA);	// Evaluate subtree
	}

	// Send scores and then help compute search statistics
	MPI_Igatherv(scores, endMoveIdx-startMoveIdx, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, MASTER_PID, MPI_COMM_WORLD, &scoresReq);
	MPI_Scan(&numBoards, &numBoards, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Scan(&pruned, &pruned, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
	MPI_Scan(&lowestDepth, &lowestDepth, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

	free(scores);	// Cleanup
}

// Only does recursively for 1 branch of search tree
double masteralphabeta(int brd[const], const int depth, const int color, const int passed) {
	int shouldMaximise = color==COLOR, nextMove, *brdcpy, *tempMoves;
	double score = 0;
	lowestDepth = MIN(lowestDepth, depth); numBoards++;

	if(!depth) return evaluateBoard(brd);	// Leaf node: not specified to go deeper!

	tempMoves = malloc(R*C*sizeof(int));
	nextMove = getLegalMoves(brd, color, tempMoves);	// Get a feasible move. Recycling nextMove to count legal moves.
	if(nextMove) {
		// Grab 1st move since only evaluating 1 full path on master.
		nextMove = tempMoves[0]; brdcpy = malloc((R<<1)*sizeof(int));
		applyMove(brdcpy, brd, nextMove, color);
		score = masteralphabeta(brdcpy, depth-1, !color, 0);
		free(brdcpy); free(tempMoves);
	} else if(!passed) score = masteralphabeta(brd, depth-1, !color, 1);	// Skip
	else score = evaluateBoard(brd);	// Previous and current user cannot make a move. Endgame.

	return score;
}

// This is actually done on slaves
double alphabeta(int brd[const], const int depth, const int color, const int passed, double alpha, double beta) {
	int *tempMoves, *brdcpy, nMoves, i, isMaxPlayer = color==COLOR;
	double res = isMaxPlayer? MINALPHA: MAXBETA, score;
	numBoards++; lowestDepth = MIN(lowestDepth, depth);

	if(!depth) return evaluateBoard(brd);	// Leaf node

	tempMoves = malloc(R*C*sizeof(int));
	brdcpy = malloc((R<<1)*sizeof(int));
	nMoves = getLegalMoves(brd, color, tempMoves);

	if(!nMoves && passed) res = evaluateBoard(brd);	// End game
	else if(!nMoves) res = alphabeta(brd, depth-1, !color, 1, alpha, beta);
	else for(i=0; i<nMoves; i++) {
		// Check if master has issued a termination order
		MPI_Test(&stopSigReq, &stopSigReqFlag, MPI_STATUS_IGNORE);
		if(stopSigReqFlag) break;

		applyMove(brdcpy, brd, tempMoves[i], color);
		score = alphabeta(brdcpy, depth-1, !color, 0, alpha, beta);
		if(isMaxPlayer) {
			res = MAX(res, score);
			alpha = MAX(alpha, res);
		} else {
			res = MIN(res, score);
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
double elapsedTime(struct timespec start, struct timespec end) {
	double timetaken = end.tv_sec-start.tv_sec;
	if(end.tv_nsec<start.tv_nsec) { timetaken-=1; end.tv_nsec+=1000000000; }
	return timetaken+(end.tv_nsec-start.tv_nsec)/1000000000.0;
}

int initBoard(char *boardfile, char *paramfile) {
 	char buf[INPUT_BUF_LEN], *ptok, *cur, *start; int len, r, c, err;
 	MPI_File brdfp, paramsfp;
 	MPI_Status status;

 	// Open board file
 	if(err = MPI_File_open(MPI_COMM_WORLD, boardfile, MPI_MODE_RDONLY, MPI_INFO_NULL, &brdfp)) {
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

	// Read playing color
 	start = cur+8; cur = strchr(start, '\n');	// 3th line is Color: <White|Black>
 	for(ptok=cur; !isalnum(*ptok) && ptok>=start; ptok--) *ptok = 0;
 	COLOR = !strcmp(start, "White")? WHITE: BLACK;

 	// Read timeout value
 	start = cur+10; cur = strchr(start, '\n');	// 4th line is Timeout: <timeout>
 	for(ptok=cur; !isalnum(*ptok) && ptok>=start; ptok--) *ptok = 0;
 	TIMEOUT = atoi(start);

 	MPI_File_close(&brdfp);	// Done with board file. Close it.

 	// Open params file for reading
 	if(err = MPI_File_open(MPI_COMM_WORLD, paramfile, MPI_MODE_RDONLY, MPI_INFO_NULL, &paramsfp)) {
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