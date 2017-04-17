/* othellox.c The main Othellox program
 * Matric. No.: A0110781N
 * Name: Qua Zi Xian
 */

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

// Game-related constants
#define WHITE 0
#define BLACK 1
#define MINALPHA (-DBL_MAX)
#define MAXBETA DBL_MAX

// OpenMPI specific constants and macros
#define MASTER_PID (numProcs-1)
#define TIMEOUT_THRESH (TIMEOUT-1.5)
#define BOARD_UPDATE_THRESH (MAXBOARDS/100)
#define STARTMOVEIDX(sid, nMoves, nSlaves) ((sid)*(nMoves)/(nSlaves))

// Miscellaneous macro functions
#define MAX(a, b) ((a)>(b)? (a): (b))
#define MIN(a, b) ((a)<(b)? (a): (b))

// MPI Stuff
#define NEW_ALPHA_TAG 0

// Multilevel move queue
struct OrderedMoves {
	int *corners, *edges, *others, nCorners, nEdges, nOthers;
};

/* Function declarations */
int isCorner(const int move);
int isEdge(const int move);
void orderMoves(struct OrderedMoves *oMoves, int moves[const], const int len);
int getNextMove(struct OrderedMoves *oMoves, int *move);
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

// Eval aram constants
int MAXDEPTH, CORNERVALUE, EDGEVALUE, COLOR, TIMEOUT;

int R = -1, C = -1, pid, numProcs, lowestDepth, pruned, numMoves;
double bestAlpha, tempAlpha, *scores, timetaken;
unsigned long long MAXBOARDS, numBoards, totalBoards;
int *board, bestMove, *legalMoves, shouldStop, dummy;
struct OrderedMoves *orderedMoveList;
int **moveList, **boardcopies;

// Timing variables
struct timespec starttime, endtime;

// MPI communication stuff. Mainly for asynchronous communications.
MPI_Request alphaReq, stopSigReq, boardCountReq, scoresReq, alphabcastReq, boardCountReq;
int alphaReqFlag, stopSigReqFlag, boardCountReqFlag, scoresReqFlag, alphabcastReqFlag, boardCountReqFlag;
MPI_Comm alphaChannel, stopChannel, alphabcastChannel, boardCountChannel;


/* Position labels in program, different from input
 * 210	[a3][b3][c3]
 * 543	[a2][b2][c2]
 * 876	[a1][b1][c1]
 */

int main(int argc, char **argv) {
 	int res, i;
 	struct OrderedMoves om;

 	// Make sure the input files are given
 	if(argc<3){
		printf("Format: ./othellox-serial <board_file> <eval_params_file>\n");
		return 1;
	}

	clock_gettime(CLOCK_REALTIME, &starttime);

 	MPI_Init(&argc, &argv);
 	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
 	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
 	MPI_Comm_dup(MPI_COMM_WORLD, &alphaChannel);
 	MPI_Comm_dup(MPI_COMM_WORLD, &stopChannel);
 	MPI_Comm_dup(MPI_COMM_WORLD, &alphabcastChannel);
 	MPI_Comm_dup(MPI_COMM_WORLD, &boardCountChannel);

 	// Here, assume all operations are successful.
 	// Failure handling makes the code too complicated and deviates away from
 	// developing program logic.
 	res = initBoard(argv[1], argv[2]);	// Initialize board

 	// Create a list of move list, 1 for each recursion depth.
 	// This is because there can only be at most 1 node in each depth being evaluated
 	// in the call stack at any time.
 	moveList = malloc(2*sizeof(int*));
 	for(i=0; i<2; i++) moveList[i] = malloc(R*C*sizeof(int));

 	// Create a list of board copies, similar to list of move lists.
 	boardcopies = malloc((MAXDEPTH+1)*sizeof(int*));
 	for(i=0; i<=MAXDEPTH; i++) boardcopies[i] = malloc(2*R*sizeof(int));

 	// Create and initialize list of ordered move list, similar to list of move lists.
 	orderedMoveList = malloc((MAXDEPTH+1)*sizeof(struct OrderedMoves));
 	for(i=0; i<=MAXDEPTH; i++) {
 		orderedMoveList[i].corners = malloc(4*sizeof(int));
 		orderedMoveList[i].edges = malloc((R+R+C+C-8)*sizeof(int));
 		orderedMoveList[i].others = malloc((R-2)*(C-2)*sizeof(int));
 		orderedMoveList[i].nCorners = orderedMoveList[i].nEdges = orderedMoveList[i].nOthers = 0;
 	}

 	// All compute valid first moves
 	numMoves = getLegalMoves(board, COLOR, moveList[0]);

 	// Master process, compute 1 branch to get alpha beta bounds
 	// to help slaves with cut-off
 	if(pid==MASTER_PID) {
 		masterProcess();
 	} else {	// Slave just run original minimax with alpha-beta pruning.
 		slaveProcess(STARTMOVEIDX(pid, numMoves, numProcs-1), STARTMOVEIDX(pid+1, numMoves, numProcs-1));
 	}

 	// Cleanup
 	for(i=0; i<2; i++) free(moveList[i]);
 	for(i=0; i<=MAXDEPTH; i++) {
 		if(orderedMoveList[i].corners) free(orderedMoveList[i].corners);
 		if(orderedMoveList[i].edges) free(orderedMoveList[i].edges);
 		if(orderedMoveList[i].others) free(orderedMoveList[i].others);
 		free(boardcopies[i]);
 	}
 	free(moveList);	free(orderedMoveList); free(boardcopies);
 	deinitBoard();

 	MPI_Finalize();
 	return 0;
 }

// Checks if a move is a corner
int isCorner(const int move) {
	int r = ROW(POS(move)), c = COL(POS(move));
	return (r==0 || r==R-1) && (c==0 || c==C-1);
}

// If a move is a corner, it is implicitly an edge
int isEdge(const int move) {
	int r = ROW(POS(move)), c = COL(POS(move));
	return isCorner(move) || r==0 || r==R-1 || c==0 || c==C-1;
}

// Sort moves in terms of preferability
// Make sure you really don't need this oMoves object before calling this.
void orderMoves(struct OrderedMoves *oMoves, int moves[const], const int len) {
	int i;
	oMoves->nCorners = oMoves->nEdges = oMoves->nOthers = 0;
	for(i=0; i<len; i++) {
		if(isCorner(moves[i])) oMoves->corners[oMoves->nCorners++] = moves[i];
		else if(isEdge(moves[i])) oMoves->edges[oMoves->nEdges++] = moves[i];
		else oMoves->others[oMoves->nOthers++] = moves[i];
	}
}

// Generator function to get next move from the ordered move queue.
// Returns 1 if there is a move left and 0 otherwise.
// If 1 is returned, move is set to be the next move.
// If you need to use this move again later, store it in a local variable.
// Pre-condition: oMoves is initialized with orderMoves() call.
int getNextMove(struct OrderedMoves *oMoves, int *move) {
	if(oMoves->nCorners) {
		*move = oMoves->corners[--oMoves->nCorners];
		return 1;
	} else if(oMoves->nEdges) {
		*move = oMoves->edges[--oMoves->nEdges];
		return 1;
	} else if(oMoves->nOthers) {
		*move = oMoves->others[--oMoves->nOthers];
		return 1;
	} else return 0;
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
		// Get the max initial alpha from slaves, then broadcast it
		bestAlpha = MINALPHA;
		MPI_Scan(&bestAlpha, &bestAlpha, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		MPI_Bcast(&bestAlpha, 1, MPI_DOUBLE, MASTER_PID, alphabcastChannel);

		// Prepare to receive board count and updated alpha
		MPI_Iscan(&numBoards, &totalBoards, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, boardCountChannel, &boardCountReq);
		MPI_Irecv(&tempAlpha, 1, MPI_DOUBLE, MPI_ANY_SOURCE, NEW_ALPHA_TAG, alphaChannel, &alphaReq);

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
		MPI_Test(&boardCountReq, &boardCountReqFlag, MPI_STATUS_IGNORE);
		clock_gettime(CLOCK_REALTIME, &endtime);
		while(!scoresReqFlag && (!boardCountReqFlag || totalBoards<MAXBOARDS) && (timetaken = elapsedTime(starttime, endtime))<TIMEOUT_THRESH) {
			// Update and broadcast new globa alpha
			MPI_Test(&alphaReq, &alphaReqFlag, MPI_STATUS_IGNORE);
			if(alphaReqFlag) {
				if(tempAlpha>bestAlpha) {
					bestAlpha = tempAlpha;
					MPI_Ibcast(&bestAlpha, 1, MPI_DOUBLE, MASTER_PID, alphabcastChannel, &alphabcastReq);
				}
				MPI_Irecv(&tempAlpha, 1, MPI_DOUBLE, MPI_ANY_SOURCE, NEW_ALPHA_TAG, alphaChannel, &alphaReq);
			}

			// Check board counts
			if(boardCountReqFlag) MPI_Iscan(&numBoards, &totalBoards, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, boardCountChannel, &boardCountReq);
			MPI_Test(&boardCountReq, &boardCountReqFlag, MPI_STATUS_IGNORE);
			MPI_Test(&scoresReq, &scoresReqFlag, MPI_STATUS_IGNORE);
			clock_gettime(CLOCK_REALTIME, &endtime);
		}

		shouldStop = 1;

		// Time threshold exceeded, tell everyone to stop
		MPI_Ibcast(&shouldStop, 1, MPI_INT, MASTER_PID, stopChannel, &stopSigReq);
		MPI_Wait(&scoresReq, MPI_STATUS_IGNORE); // Wait for all to send scores
		
		// Here, all scores should be ready for processing.
		// Process scores to get best move(s)
		bestScore = MINALPHA; bestMove = -1; legalMoves = moveList[0];
		for(i=0; i<numMoves; i++) if(bestScore<scores[i]) { bestScore = scores[i]; bestMove = legalMoves[i]; }

		// Gather other statistics
		MPI_Scan(&numBoards, &totalBoards, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
		MPI_Scan(&pruned, &pruned, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
		MPI_Scan(&lowestDepth, &lowestDepth, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

		// Cleanup variables in intermediate computation
		free(scores); free(recvcounts); free(displs);
	} else {	// No legal move
		bestMove = -1; numBoards = 1;
	}

	clock_gettime(CLOCK_REALTIME, &endtime);

	printOutput(bestMove);	// Print output
}

// Work done by each slave
void slaveProcess(const int startMoveIdx, const int endMoveIdx) {
	int i, *brdcpy, move;

	numBoards = totalBoards = 0; lowestDepth = MAXDEPTH; pruned = 0; shouldStop = 0; dummy = 0;

	// Open a channel to receive stop signal.
	MPI_Ibcast(&shouldStop, 1, MPI_INT, MASTER_PID, stopChannel, &stopSigReq);
	MPI_Test(&stopSigReq, &stopSigReqFlag, MPI_STATUS_IGNORE);

	// Facilitate completion of total board count
	MPI_Iscan(&totalBoards, &totalBoards, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, boardCountChannel, &boardCountReq);
	MPI_Test(&boardCountReq, &boardCountReqFlag, MPI_STATUS_IGNORE);

	// Won't be doing any subtree searching...
	if(startMoveIdx>=endMoveIdx) {
		bestAlpha = MINALPHA;
		MPI_Scan(&bestAlpha, &bestAlpha, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

		// Not involved in actual work but still need to receive bcast.
		MPI_Bcast(&bestAlpha, 1, MPI_DOUBLE, MASTER_PID, alphabcastChannel);
		MPI_Ibcast(&tempAlpha, 1, MPI_DOUBLE, MASTER_PID, alphabcastChannel, &alphabcastReq);

		// Standby so that working slaves can end on time
		MPI_Igatherv(NULL, 0, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, MASTER_PID, MPI_COMM_WORLD, &scoresReq);

		while(!stopSigReqFlag) {
			// Help keep alpha bcast working
			MPI_Test(&alphabcastReq, &alphabcastReqFlag, MPI_STATUS_IGNORE);
			if(alphabcastReqFlag) MPI_Ibcast(&tempAlpha, 1, MPI_DOUBLE, MASTER_PID, alphabcastChannel, &alphabcastReq);

			// Keep helping master update board count
			MPI_Test(&boardCountReq, &boardCountReqFlag, MPI_STATUS_IGNORE);
			if(boardCountReqFlag) MPI_Iscan(&numBoards, &totalBoards, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, boardCountChannel, &boardCountReq);

			// Watch for stop signal
			MPI_Test(&stopSigReq, &stopSigReqFlag, MPI_STATUS_IGNORE);
		}

		// Help master with book-keeping
		MPI_Scan(&numBoards, &numBoards, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
		MPI_Scan(&pruned, &pruned, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
		MPI_Scan(&lowestDepth, &lowestDepth, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
		return;
	}

	legalMoves = moveList[0];
	scores = malloc((endMoveIdx-startMoveIdx)*sizeof(double));	// Prepare holder for each move's score
	brdcpy = boardcopies[MAXDEPTH];
	memset(scores, 0, (endMoveIdx-startMoveIdx)*sizeof(double));

	// Sort the moves by favourability
	orderMoves(&orderedMoveList[MAXDEPTH], legalMoves+startMoveIdx, endMoveIdx-startMoveIdx);
	getNextMove(&orderedMoveList[MAXDEPTH], &move);

	// Do master alphabeta to get a good alpha from the start
	applyMove(brdcpy, board, move, COLOR);
	bestAlpha = masteralphabeta(brdcpy, MAXDEPTH-1, !COLOR, 0);
	MPI_Scan(&bestAlpha, &bestAlpha, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	// Receive initial alpha and beta from master
	MPI_Bcast(&bestAlpha, 1, MPI_DOUBLE, MASTER_PID, alphabcastChannel);
	MPI_Ibcast(&tempAlpha, 1, MPI_DOUBLE, MASTER_PID, alphabcastChannel, &alphabcastReq);
	
	// Try the moves in descending preferability
	do {
		// Check if master has issued a termination order
		MPI_Test(&stopSigReq, &stopSigReqFlag, MPI_STATUS_IGNORE);
		if(stopSigReqFlag) break;

		// Update global alpha before proceeding
		MPI_Test(&alphabcastReq, &alphabcastReqFlag, MPI_STATUS_IGNORE);
		if(alphabcastReqFlag) {
			bestAlpha = MAX(bestAlpha, tempAlpha);
			MPI_Ibcast(&tempAlpha, 1, MPI_DOUBLE, MASTER_PID, alphabcastChannel, &alphabcastReq);
		}

		// Safe to continue search for now
		applyMove(brdcpy, board, move, COLOR);	// Apply given move to evaluate
		for(i=startMoveIdx; i<endMoveIdx; i++) if(legalMoves[i]==move) break;	// Find where to place the score
		scores[i-startMoveIdx] = alphabeta(brdcpy, MAXDEPTH-1, !COLOR, 0, bestAlpha, MAXBETA);	// Evaluate subtree

		// Update global alpha if possible
		if(scores[i-startMoveIdx]>bestAlpha) {
			bestAlpha = scores[i-startMoveIdx];
			MPI_Isend(&bestAlpha, 1, MPI_DOUBLE, MASTER_PID, NEW_ALPHA_TAG, alphaChannel, &alphaReq);
		}
	} while(getNextMove(&orderedMoveList[MAXDEPTH], &move));

	// Send scores and then help compute search statistics
	MPI_Igatherv(scores, endMoveIdx-startMoveIdx, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, MASTER_PID, MPI_COMM_WORLD, &scoresReq);
	MPI_Scan(&numBoards, &numBoards, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
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

	tempMoves = moveList[1];
	nextMove = getLegalMoves(brd, color, tempMoves);	// Get a feasible move. Recycling nextMove to count legal moves.
	orderMoves(&orderedMoveList[depth], tempMoves, nextMove);
	if(getNextMove(&orderedMoveList[depth], &nextMove)) {
		// Grab 1st move since only evaluating 1 full path on master.
		brdcpy = boardcopies[depth];
		applyMove(brdcpy, brd, nextMove, color);
		score = masteralphabeta(brdcpy, depth-1, !color, 0);
	} else if(!passed) score = masteralphabeta(brd, depth-1, !color, 1);	// Skip
	else score = evaluateBoard(brd);	// Previous and current user cannot make a move. Endgame.

	return score;
}

// This is actually done on slaves
double alphabeta(int brd[const], const int depth, const int color, const int passed, double alpha, double beta) {
	int *tempMoves, *brdcpy, nMoves, i, isMaxPlayer = color==COLOR, move;
	double res = isMaxPlayer? MINALPHA: MAXBETA, score;
	numBoards++; lowestDepth = MIN(lowestDepth, depth);

	// Update global board count
	if(numBoards/BOARD_UPDATE_THRESH > dummy) {
		MPI_Test(&boardCountReq, &boardCountReqFlag, MPI_STATUS_IGNORE);
		// Hope that Iscan won't take too long to complete
		if(boardCountReqFlag) {	// Ok to send gather
			totalBoards = numBoards;
			dummy = numBoards/BOARD_UPDATE_THRESH;
			MPI_Iscan(&totalBoards, &totalBoards, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, boardCountChannel, &boardCountReq);
		}
	}

	if(!depth) return evaluateBoard(brd);	// Leaf node

	tempMoves = moveList[1];
	brdcpy = boardcopies[depth];
	nMoves = getLegalMoves(brd, color, tempMoves);

	if(!nMoves && passed) res = evaluateBoard(brd);	// End game
	else if(!nMoves) res = alphabeta(brd, depth-1, !color, 1, alpha, beta);
	else {
		// Sort the moves first
		orderMoves(&orderedMoveList[depth], tempMoves, nMoves);
		while(getNextMove(&orderedMoveList[depth], &move)) {
			// Check if master has issued a termination order
			MPI_Test(&stopSigReq, &stopSigReqFlag, MPI_STATUS_IGNORE);
			if(stopSigReqFlag) break;

			// Check for updated alpha
			MPI_Test(&alphabcastReq, &alphabcastReqFlag, MPI_STATUS_IGNORE);
			if(alphabcastReqFlag) {
				bestAlpha = MAX(bestAlpha, tempAlpha);
				alpha = MAX(alpha, bestAlpha);
				MPI_Ibcast(&tempAlpha, 1, MPI_DOUBLE, MASTER_PID, alphabcastChannel, &alphabcastReq);
			}

			applyMove(brdcpy, brd, move, color);
			score = alphabeta(brdcpy, depth-1, !color, 0, alpha, beta);
			if(isMaxPlayer) {
				res = MAX(res, score);
				alpha = MAX(alpha, res);
			} else {
				res = MIN(res, score);
				beta = MIN(beta, res);
			}

			// Check for updated alpha
			// Might be quite a while between previous check and this check
			// Even if child call updates bestAlpha, this alpha is not updated
			MPI_Test(&alphabcastReq, &alphabcastReqFlag, MPI_STATUS_IGNORE);
			if(alphabcastReqFlag) {
				bestAlpha = MAX(bestAlpha, tempAlpha);
				MPI_Ibcast(&tempAlpha, 1, MPI_DOUBLE, MASTER_PID, alphabcastChannel, &alphabcastReq);
			}
			alpha = MAX(alpha, bestAlpha);
			if(beta<=alpha) { pruned|=(i+1<nMoves); break; }
		}
	}

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
	printf("Number of boards assessed: %llu\n", totalBoards);
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
 	start = buf+6; cur = strchr(start, '\n');	// 1st line is Size: R,C
 	for(ptok=cur; !isalnum(*ptok) && ptok>=start; ptok--) *ptok = 0;	// Erase non-alnum characters
 	ptok = strtok(start, ",");
 	while(ptok!=NULL) {
		if(R==-1) R = atoi(ptok);
		else if(C==-1) C = atoi(ptok);
		else break;
		ptok = strtok(NULL, ",");
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
	MAXBOARDS = atol(start);

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
