#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#include <immintrin.h>
#include <sys/time.h>
#include <cassert>
#include <cstdint>
#include <iostream>

float tdiff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec-start->tv_sec) + 1e-6*(end->tv_usec-start->tv_usec);
}

struct Planet {
    double x;
    double y;
    double vx;
    double vy;
};

struct Velo4 {
    double vx[4];
    double vy[4];
};

struct Cord4 {
    double x[4];
    double y[4];
};

struct Cord {
    double x;
    double y;
};

struct Velo {
    double vx;
    double vy;
};

struct Mass {
    double mass;
};

struct Mass4 {
    double mass[4];
};

unsigned long long seed = 100;

unsigned long long randomU64() {
  seed ^= (seed << 21);
  seed ^= (seed >> 35);
  seed ^= (seed << 4);
  return seed;
}

double randomDouble()
{
    unsigned long long next = randomU64();
    next >>= (64 - 26);
    unsigned long long next2 = randomU64();
    next2 >>= (64 - 26);
    return ((next << 27) + next2) / (double)(1LL << 53);
}

int nplanets;
int timesteps;
double dt;
double G;

int main(int argc, const char** argv) {
    if (argc < 2) {
        printf("Usage: %s <nplanets> <timesteps>\n", argv[0]);
        return 1;
    }
    nplanets = atoi(argv[1]);
    timesteps = atoi(argv[2]);
    dt = 0.001;
    G = 6.6743;

    int nplanets_chunks = nplanets/4;


    Cord* cords = (Cord*)aligned_alloc(32, sizeof(Cord) * nplanets);
    Cord* nextcords = (Cord*)aligned_alloc(32, sizeof(Cord) * nplanets);
    Velo* velo = (Velo*)aligned_alloc(32, sizeof(Velo) * nplanets);

    Cord* tmp = cords;

    printf("planets: %p \n", cords);
    printf("nextplanets: %p \n", nextcords);

    
    Mass* masses = (Mass*)aligned_alloc(32, sizeof(Mass) * nplanets);
    assert(nplanets % 4 == 0 );

    for (int i=0; i<nplanets; i++) {

        double mass = randomDouble() * 10 + 0.2;
        masses[i].mass= mass;
        cords[i].x = ( randomDouble() - 0.5 ) * 100 * pow(1 + nplanets, 0.4);
        cords[i].y = ( randomDouble() - 0.5 ) * 100 * pow(1 + nplanets, 0.4);
        velo[i].vx = randomDouble() * 5 - 2.5;
        velo[i].vy = randomDouble() * 5 - 2.5;
        
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);

    for (int i=0; i<timesteps; i++) {    
        #pragma omp parallel for
            for (int i=0; i<nplanets; i++) {
                #pragma omp simd
                for (int j=0; j<nplanets; j++) {
                   double dx = cords[j].x - cords[i].x;
                   double dy = cords[j].y - cords[i].y;
                   double distSqr = dx*dx + dy*dy + 0.0001;
                   double invDist = masses[i].mass * masses[j].mass / sqrt(distSqr);
                   double invDist3 = invDist * invDist * invDist;
                   velo[i].vx += dt * dx * invDist3;
                   velo[i].vy += dt * dy * invDist3;
                }
                nextcords[i].x = dt * velo[i].vx + cords[i].x;
                nextcords[i].y = dt * velo[i].vy + cords[i].y;
            }
        tmp = cords;
        cords = nextcords;
        nextcords = tmp;
    }

    gettimeofday(&end, NULL);
    printf("Total time to run simulation %0.6f seconds, final location %f %f\n", tdiff(&start, &end), cords[nplanets-1].x, cords[nplanets-1].y);
    free(cords);
    free(nextcords);
    return 0;    
}