#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAX(a, b) (a>b? a: b)
#define MIN(a, b) (a<b? a: b)

/* Position labels in program, different from input
 * 210	[a3][b3][c3]
 * 543	[a2][b2][c2]
 * 876	[a1][b1][c1]
 */

struct Vector { int *arr, size, capacity; };

// Must call this before using Vector
void vectorInit(struct Vector *v){
	v->arr = malloc(2*sizeof(int));	// Assume all malloc successful here for simplicity
	v->capacity = 2; v->size = 0;
}

// Must call this when done with the Vector object
void vectorDestroy(struct Vector *v){
	if(v->arr!=NULL) free(v->arr);
	v->size = v->capacity = 0;
}

void vectorClear(struct Vector *v){ v->size = 0; }

void vectorPush(struct Vector *v, int elem){
	int i;
	if(v->size==v->capacity){	// Already full, expand the array
		int *temp = malloc(v->capacity*2*sizeof(int));	// Allocate double the size
		memcpy((void*)temp, (void*)v->arr, v->size*sizeof(int));	// Copy over to new array
		free(v->arr); v->capacity<<=1;
		v->arr = temp;
	}
	v->arr[v->size++] = elem;	// Append to array
}

struct Solution {
	int pruned, depth;
	unsigned long long numBoards;
	struct Vector moves;
};

int R = -1, C = -1, MAXDEPTH, MAXBOARDS, CORNERVALUE, EDGEVALUE, *board, *taken;

// Returns a bitmap of directions contributing to legality of move.
// Order of bits from most significant bit: [up, down, left, right, upleft, upright, downleft, downright]
int isLegalMove(int brd[const], int taken[const], int pos, const int color){
	int r = pos/C, c = pos%C, pos2, dirs = 0, temp;

	for(pos2=pos-C, temp=0; pos2>=0; pos2-=C){	// Scan upwards
		r = pos2/C; c = pos2%C;
		if(!(taken[r]&(1<<c))){ temp = 0; break; }	// Fail to trap any opposite color disk
		if(color!=!!(brd[r]&(1<<c))) temp++;
		else break;	// See same color, know how many opposite color disk(s) trapped
	}
	temp = pos2<0? 0: temp;	// Possible to reach edge without seeing same color
	if(temp) dirs|=128;

	for(pos2=pos+C, temp=0; pos2<R*C; pos2+=C){	// Scan downwards
		r = pos2/C; c = pos2%C;
		if(!(taken[r]&(1<<c))){ temp = 0; break; }	// Fail to trap any opposite color disk
		if(color!=!!(brd[r]&(1<<c))) temp++;
		else break;	// See same color, know how many opposite color disk(s) trapped
	}
	temp = pos2>=R*C? 0: temp;	// Possible to reach edge without seeing same color
	if(temp) dirs|=64;

	for(pos2=pos+1, temp=0; pos2%C>0; pos2++){	// Scan leftwards
		r = pos2/C; c = pos2%C;
		if(!(taken[r]&(1<<c))){ temp = 0; break; }	// Fail to trap any opposite color disk
		if(color!=!!(brd[r]&(1<<c))) temp++;
		else break;	// See same color, know how many opposite color disk(s) trapped
	}
	temp = pos2%C==0? 0: temp; // Possible to reach edge without seeing same color
	if(temp) dirs|=32;

	if(pos%C>0){
		for(pos2=pos-1, temp=0; pos2%C>=0; pos2--){	// Scan rightwards
			r = pos2/C; c = pos2%C;
			if(!(taken[r]&(1<<c))){ temp = 0; break; }	// Fail to trap any opposite color disk
			if(color!=!!(brd[r]&(1<<c))) temp++;
			else break;	// See same color, know how many opposite color disk(s) trapped
		}
		temp = pos2==-1 || pos2%C==C-1? 0: temp; // Possible to reach edge without seeing same color
		if(temp) dirs|=16;
	}

	for(pos2=pos+1-C, temp=0; pos2>0; pos2-=C-1){	// Scan upleft
		r = pos2/C; c = pos2%C;
		if(!(taken[r]&(1<<c))){ temp = 0; break; }	// Fail to trap any opposite color disk
		if(color!=!!(brd[r]&(1<<c))) temp++;
		else break;	// See same color, know how many opposite color disk(s) trapped
	}
	temp = pos2<=0? 0: temp; // Possible to reach edge without seeing same color
	if(temp) dirs|=8;

	for(pos2=pos-1-C, temp=0; pos2>=0; pos2-=C+1){	// Scan upright
		r = pos2/C; c = pos2%C;
		if(!(taken[r]&(1<<c))){ temp = 0; break; }	// Fail to trap any opposite color disk
		if(color!=!!(brd[r]&(1<<c))) temp++;
		else break;	// See same color, know how many opposite color disk(s) trapped
	}
	temp = pos2<0? 0: temp; // Possible to reach edge without seeing same color
	if(temp) dirs|=4;

	for(pos2=pos+1+C, temp=0; pos2<R*C; pos2+=C+1){	// Scan downleft
		r = pos2/C; c = pos2%C;
		if(!(taken[r]&(1<<c))){ temp = 0; break; }	// Fail to trap any opposite color disk
		if(color!=!!(brd[r]&(1<<c))) temp++;
		else break;	// See same color, know how many opposite color disk(s) trapped
	}
	temp = pos2>=R*C? 0: temp; // Possible to reach edge without seeing same color
	if(temp) dirs|=2;

	for(pos2=pos-1+C, temp=0; pos2<R*C; pos2+=C-1){	// Scan downright
		r = pos2/C; c = pos2%C;
		if(!(taken[r]&(1<<c))){ temp = 0; break; }	// Fail to trap any opposite color disk
		if(color!=!!(brd[r]&(1<<c))) temp++;
		else break;	// See same color, know how many opposite color disk(s) trapped
	}
	temp = pos2>=R*C? 0: temp; // Possible to reach edge without seeing same color
	if(temp) dirs|=1;

	return dirs;
}

// This should only be called with dirs properly set by isLegalMove.
// pos here should be the same as that used in isLegalMove.
// With dirs mask check, we can ignore taken table and just check for color difference in for-loop
void flipDiscs(int brd[const], int pos, int dirs){
	int pos2, r = pos/C, c = pos%C, mask = 1<<c, color = !!(brd[r]&mask);
	if(dirs&128)	// Upwards
		for(pos2=r-1; pos2>=0 && color!=!!(brd[pos2]&mask); pos2--) brd[pos2]^=mask;

	if(dirs&64)	// Downwards
		for(pos2=r+1; pos2<R && color!=!!(brd[pos2]&mask); pos2++) brd[pos2]^=mask;

	if(dirs&32)	// Leftwards
		for(mask=1<<(c+1); color!=!!(brd[r]&mask); mask<<=1) brd[r]^=mask;

	if(dirs&16)	// Rightwards
		for(mask=1<<(c-1); color!=!!(brd[r]&mask); mask>>=1) brd[r]^=mask;

	if(dirs&8)	// Upleft
		for(mask=1<<(c+1), pos2=r-1; pos2>=0 && color!=!!(brd[pos2]&mask); pos2--, mask<<=1)
			brd[pos2]^=mask;

	if(dirs&4)	// Upright
		for(mask=1<<(c-1), pos2=r-1; pos2>=0 && color!=!!(brd[pos2]&mask); pos2--, mask>>=1)
			brd[pos2]^=mask;

	if(dirs&2)	// Downleft
		for(mask=1<<(c+1), pos2=r+1; pos2<R && color!=!!(brd[pos2]&mask); pos2++, mask<<=1)
			brd[pos2]^=mask;

	if(dirs&1)	// Downright
		for(mask=1<<(c-1), pos2=r+1; pos2<R && color!=!!(brd[pos2]&mask); pos2++, mask>>=1)
			brd[pos2]^=mask;
}

int evaluateBoard(int brd[const], int taken[const]){
	int numBlack = 0, numWhite = 0, pos, r, c, mask, numCorners, temp, diff;
	double res = 0;

	// Disc counting
	for(r=0; r<R; r++){ numBlack+=__builtin_popcount(brd[r]); numWhite+=__builtin_popcount(taken[r]); }
	numWhite-=numBlack; res = fabs((double)(numWhite-numBlack)/(numWhite+numBlack));

	// Mobility counting
	numBlack = numWhite = 0;
	for(pos=0; pos<R*C; pos++){
		r = R-1-pos/C; c = C-1-(pos%C);
		if(!(taken[r]&(1<<c))){
			if(isLegalMove(brd, taken, pos, 0)) numWhite++;
			if(isLegalMove(brd, taken, pos, 1)) numBlack++;
		}
	}
	res+=10.0*fabs((double)(numWhite-numBlack)/(numWhite+numBlack));

	// Strategic pieces
	// Handle corners
	mask = 1|(1<<(C-1));	// Corner mask
	r = taken[0]&mask; c = taken[R-1]&mask;
	numCorners = __builtin_popcount(r)+__builtin_popcount(c);	// Number of corners occupied
	r&=brd[0]; c&=brd[R-1]; r = __builtin_popcount(r)+__builtin_popcount(c); c = numCorners-r;
	diff = (r-c)*CORNERVALUE; numCorners*=CORNERVALUE;

	// Handle vertical edges
	for(pos=1; pos<R-1; pos++){	// Reusing pos as row number
		r = taken[pos]&mask; temp = __builtin_popcount(r);
		r = __builtin_popcount(brd[pos]&r); c = temp-r;
		diff+=(r-c)*EDGEVALUE; numCorners+=temp*EDGEVALUE;
	}

	// Handle top and bottom edges
	mask = (1<<C)-1-mask;
	r = taken[0]&mask; c = taken[R-1]&mask;	// Reuse r and c to get occupied cells in top and bottom edges
	temp = __builtin_popcount(r)+__builtin_popcount(c);	// Total number of edge cells taken
	r&=brd[0]; c&=brd[R-1]; r = __builtin_popcount(r)+__builtin_popcount(c); c = temp-r;
	diff+=(r-c)*EDGEVALUE; numCorners+=temp*EDGEVALUE;

	res+=(double)abs(diff)/numCorners;

	return (int)round(res);
}

int alphabeta(int brd[const], int taken[const], int depth, int alpha, int beta, int shouldMaximise, int passed, struct Solution *sol){
	int res = shouldMaximise? 1<<31: ~(1<<31), res2, pos, r, c, *brdcpy, *takencpy, shouldPass = 1, dirs;
	struct Vector moves; vectorInit(&moves);

	if(!depth){	// Terminal node in search tree. No possible moves.
		if(sol!=NULL){
			vectorClear(&sol->moves); sol->depth = 0; sol->pruned|=0; sol->numBoards++;
		}
		return evaluateBoard(brd, taken);
	}

	// Set up copy of board and taken
	brdcpy = malloc(R*sizeof(int)); takencpy = malloc(R*sizeof(int));


	// Brute-force try all positions
	for(pos=0; pos<R*C; pos++){
		r = R-1-pos/C; c = C-1-(pos%C);
		if(!(taken[r]&(1<<c)) && (dirs = isLegalMove(brd, taken, pos, !!shouldMaximise))){
			// Make fresh copy of the boards
			memcpy((void*)brdcpy, (void*)brd, R*sizeof(int));
			memcpy((void*)takencpy, (void*)taken, R*sizeof(int));

			// Place the disk on board copy.
			brdcpy[r] = shouldMaximise? brd[r]|(1<<c): brd[r]; takencpy[r] = taken[r]|(1<<c);
			flipDiscs(brdcpy, pos, dirs);
			shouldPass = 0;
			res2 = alphabeta(brdcpy, takencpy, depth-1, alpha, beta, !shouldMaximise, 0, sol);

			// Update set of maximizing/minimizing moves
			if(shouldMaximise){
				if(res2>res) vectorClear(&moves);
				if(res2>=res) vectorPush(&moves, pos);
				res = MAX(res, res2);
			}
			else{
				if(res2<res) vectorClear(&moves);
				if(res2<=res) vectorPush(&moves, pos);
				res = MIN(res, res2);
			}
			if(sol!=NULL) sol->numBoards++;
			if(beta<=alpha){ sol->pruned = 1; break; }
		}
	}

	// Cannot find any legal move, either pass or endgame
	if(shouldPass) res = passed? evaluateBoard(brd, taken): alphabeta(brd, taken, depth-1, alpha, beta, !shouldMaximise, 1, sol);

	vectorDestroy(&sol->moves); sol->moves = moves;	// Replace sol->moves with own moves
	free(brdcpy); free(takencpy);
	return res;
}

void printSolution(struct Solution * const sol){
	int i, r; char c;
	printf("Best moves: { ");
	for(i=0; i<sol->moves.size; i++){
		c = C-1-(sol->moves.arr[i]%C)+'a'; r = R-sol->moves.arr[i]/C;
		printf(i? ",%c%d": "%c%d", c, r);
		// printf(i? ",%d":"%d", sol->moves.arr[i]);
	}
	printf(" }\n");
	printf("Number of boards assessed: %llu\n", sol->numBoards);
	printf("Depth of boards: %d\n", MAXDEPTH-sol->depth);
	printf("Entire space: "); printf(sol->pruned? "false\n": "true\n");
	printf("Elapsed time in seconds: 123\n");
}

int main(int argc, char **argv){
	if(argc<3){
		printf("Format: ./othellox-serial <board_file> <eval_params_file>\n");
		return 1;
	}

	struct Solution sol;
	sol.numBoards = sol.pruned = 0;
	FILE *brdfile = fopen(argv[1], "r"), *evalfile = fopen(argv[2], "r");
	int len, r, c;
	char buf[1024], *ptok;

	// Read board size
	fgets(buf, 1024, brdfile); len = strlen(buf)-1;
	while(len>=0 && !isdigit(buf[len])) buf[len--] = 0;
	ptok = strtok(buf+6, ",");
	while(ptok!=NULL){
		if(C==-1) C = atoi(ptok);
		else if(R==-1) R = atoi(ptok);
		else break;
	}
	board = malloc(R*sizeof(int)); taken = malloc(R*sizeof(int));

	// Read and process white positions
	fgets(buf, 1024, brdfile); len = strlen(buf)-1;
	while(len>=0 && !isalnum(buf[len])) buf[len--] = 0;
	ptok = strtok(buf+9, ",");
	while(ptok!=NULL){
		c = C-1-(ptok[0]-'a'); r = R-atoi(ptok+1); ptok = strtok(NULL, ",");
		taken[r]|=(1<<c);
	}

	// Read and process black positions
	fgets(buf, 1024, brdfile); len = strlen(buf)-1;
	while(len>=0 && !isalnum(buf[len])) buf[len--] = 0;
	ptok = strtok(buf+9, ",");
	while(ptok!=NULL){
		c = C-1-(ptok[0]-'a'); r = R-atoi(ptok+1); ptok = strtok(NULL, ",");
		taken[r]|=(1<<c); board[r]|=(1<<c);
	}
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

	// Main algorithm here
	sol.depth = MAXDEPTH; vectorInit(&sol.moves);
	alphabeta(board, taken, MAXDEPTH, 1<<31, ~(1<<31), 1, 0, &sol);
	printSolution(&sol);

	// Clean up allocated memory
	vectorDestroy(&sol.moves); free(board); free(taken);
	return 0;
}
