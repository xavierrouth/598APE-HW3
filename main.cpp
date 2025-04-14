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

struct PlanetVec4 {
    double x[4];
    double y[4];
    double vx[4];
    double vy[4];
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

    PlanetVec4* planets = (PlanetVec4*)aligned_alloc(32, sizeof(Planet) * nplanets);
    PlanetVec4* nextplanets = (PlanetVec4*)aligned_alloc(32, sizeof(Planet) * nplanets);
    PlanetVec4* tmp = planets;

    printf("planets: %p \n", planets);
    printf("nextplanets: %p \n", nextplanets);

    
    double* masses = (double*)aligned_alloc(32, sizeof(double) * nplanets);
    assert(nplanets % 4 == 0 );

    int nplanets_chunks = nplanets/4;
    for (int i=0; i<nplanets_chunks; i++) {
        for (int j = 0; j<4; j++) {
            double mass = randomDouble() * 10 + 0.2;
            masses[i] = mass;
            planets[i].x[j] = ( randomDouble() - 0.5 ) * 100 * pow(1 + nplanets, 0.4);
            planets[i].y[j] = ( randomDouble() - 0.5 ) * 100 * pow(1 + nplanets, 0.4);
            planets[i].vx[j] = randomDouble() * 5 - 2.5;
            planets[i].vy[j] = randomDouble() * 5 - 2.5;
        }
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);

    for (int i=0; i<timesteps; i++) {
        // REPLACE with memcpy.
        for (int i=0; i<nplanets_chunks; i++) {
            for (int j=0; j<4; j++) {
                nextplanets[i].vx[j] = planets[i].vx[j];
                nextplanets[i].vy[j] = planets[i].vy[j];
                nextplanets[i].x[j] = planets[i].x[j];
                nextplanets[i].y[j] = planets[i].y[j];
            }
        }
    
        for (int i=0; i<nplanets_chunks; i++) {
            __m256d mi = _mm256_load_pd(&masses[i * 4]);

            for (int ii = 0; ii< 4; ii++) {
                double vx = planets[i].vx[ii];
                double vy = planets[i].vy[ii];
                double x = planets[i].x[ii];
                double y = planets[i].y[ii];
                for (int j=0; j<nplanets_chunks; j++) {
                    __m256d ix = _mm256_load_pd(planets[i].x);
                    __m256d jx = _mm256_load_pd(planets[j].x);
                    __m256d iy = _mm256_load_pd(planets[i].y);
                    __m256d jy = _mm256_load_pd(planets[j].y);
                    __m256d dx = _mm256_sub_pd(ix, jx);
                    __m256d dy = _mm256_sub_pd(iy, jy);
                    __m256d s1 = _mm256_fmadd_pd(dx, dx, _mm256_set1_pd(0.001));
                    __m256d s2 = _mm256_fmadd_pd(dy, dy, s1);
                    __m256d sqrt = _mm256_sqrt_pd(s2);
                    // masses[i * nplanets + j]
                    
                    __m256d mj = _mm256_load_pd(&masses[j * 4]);
                    __m256d mij = _mm256_mul_pd(mi, mj);

                    __m256d div = _mm256_div_pd(mij, sqrt);

                    __m256d dtv = _mm256_set1_pd(dt);
                    auto a = _mm256_mul_pd(div, div);
                    auto b = _mm256_mul_pd(a, div);
                    auto c = _mm256_mul_pd(b, dtv);
                    auto vdx = _mm256_mul_pd(b, dx);
                    auto vdy = _mm256_mul_pd(b, dy);

                    __m256d temp = _mm256_hadd_pd(vdx, vdy);
                    __m256d shuffled = _mm256_permute4x64_pd(temp, _MM_SHUFFLE(3, 1, 2, 0));
                    __m256d hadd = _mm256_hadd_pd(shuffled, shuffled);
                    __m128d result = _mm256_castpd256_pd128(hadd);
                    vx += _mm_cvtsd_f64(result);
                    vy += _mm_cvtsd_f64(_mm_unpackhi_pd(result, result));  
                }
                x += dt * vx;
                y += dt * vy;
    
                nextplanets[i].vx[ii] = vx;
                nextplanets[i].vy[ii] = vy;
                nextplanets[i].x[ii] = x;
                nextplanets[i].y[ii] = y;
            }
        }
        tmp = planets;
        planets = nextplanets;
        nextplanets = tmp;
    }

    gettimeofday(&end, NULL);
    printf("Total time to run simulation %0.6f seconds, final location %f %f\n", tdiff(&start, &end), planets[nplanets_chunks-1].x[3], planets[nplanets_chunks-1].y[3]);
    free(planets);
    free(nextplanets);
    return 0;    
}