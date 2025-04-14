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


    Cord4* cords = (Cord4*)aligned_alloc(32, sizeof(Cord4) * nplanets_chunks);
    Cord4* nextcords = (Cord4*)aligned_alloc(32, sizeof(Cord4) * nplanets_chunks);
    Velo4* velo = (Velo4*)aligned_alloc(32, sizeof(Velo4) * nplanets_chunks);

    Cord4* tmp = cords;

    printf("planets: %p \n", cords);
    printf("nextplanets: %p \n", nextcords);

    
    Mass4* masses = (Mass4*)aligned_alloc(32, sizeof(Mass4) * nplanets);
    assert(nplanets % 4 == 0 );

    for (int i=0; i<nplanets_chunks; i++) {
        for (int j = 0; j<4; j++) {
            double mass = randomDouble() * 10 + 0.2;
            masses[i].mass[j] = mass;
            cords[i].x[j] = ( randomDouble() - 0.5 ) * 100 * pow(1 + nplanets, 0.4);
            cords[i].y[j] = ( randomDouble() - 0.5 ) * 100 * pow(1 + nplanets, 0.4);
            velo[i].vx[j] = randomDouble() * 5 - 2.5;
            velo[i].vy[j] = randomDouble() * 5 - 2.5;
        }
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);

    for (int i=0; i<timesteps; i++) {    
        for (int i=0; i<nplanets_chunks; i++) {
            __m256d mi = _mm256_load_pd(masses[i].mass);

            for (int ii = 0; ii< 4; ii++) {
                double vx = velo[i].vx[ii];
                double vy = velo[i].vy[ii];
                double x = cords[i].x[ii];
                double y = cords[i].y[ii];
                for (int j=0; j<nplanets_chunks; j++) {
                    __m256d ix = _mm256_load_pd(cords[i].x);
                    if (i == 0 && ii == 0 && j == 1) {
                        double a = _mm256_cvtsd_f64(ix);
                        printf("ix: %f ", a);
                    }

                    __m256d jx = _mm256_load_pd(cords[j].x);
                    if (i == 0 && ii == 0 && j == 1)  {
                        double a = _mm256_cvtsd_f64(jx);
                        printf("jx: %f ", a);
                    }

                    __m256d iy = _mm256_load_pd(cords[i].y);
                    if (i == 0 && ii == 0 && j == 1)   {
                        double a = _mm256_cvtsd_f64(iy);
                        printf("iy: %f ", a);
                    }
                    __m256d jy = _mm256_load_pd(cords[j].y);
                    if (i == 0 && ii == 0 && j == 1)   {
                        double a = _mm256_cvtsd_f64(jy);
                        printf("jy: %f ", a);
                    }
                    __m256d dx = _mm256_sub_pd(jx, ix);
                    if (i == 0 && ii == 0 && j == 1)   {
                        double a = _mm256_cvtsd_f64(dx);
                        printf("dx: %f ", a);
                    }
                    __m256d dy = _mm256_sub_pd(jy, iy);
                    if (i == 0 && ii == 0 && j == 1)   {
                        double a = _mm256_cvtsd_f64(dy);
                        printf("dy: %f ", a);
                    }
                    __m256d s1 = _mm256_fmadd_pd(dx, dx, _mm256_set1_pd(0.001));
                    if (i == 0 && ii == 0 && j == 1)   {
                        double a = _mm256_cvtsd_f64(s1);
                        printf("s1: %f ", a);
                    }

                    __m256d s2 = _mm256_fmadd_pd(dy, dy, s1);
                    if (i == 0 && ii == 0 && j == 1)   {
                        double a = _mm256_cvtsd_f64(s2);
                        printf("s2: %f ", a);
                    }
                    __m256d sqrt = _mm256_sqrt_pd(s2);
                    if (i == 0 && ii == 0 && j == 1)   {
                        double a = _mm256_cvtsd_f64(sqrt);
                        printf("sqrt: %f ", a);
                    }

                    // masses[i * nplanets + j]
                    
                    __m256d mj = _mm256_load_pd(masses[j].mass);

                    __m256d mij = _mm256_mul_pd(mi, mj);

                    if (i == 0 && ii == 0 && j == 1)   {
                        double aa = _mm256_cvtsd_f64(mi);
                        printf("mi: %f ", aa);

                        double ab = _mm256_cvtsd_f64(mj);
                        printf("mj: %f ", ab);

                        double a = _mm256_cvtsd_f64(mij);
                        printf("mij: %f ", a);
                    }


                    __m256d div = _mm256_div_pd(mij, sqrt);
                    if (i == 0 && ii == 0 && j == 1)   {
                        double a = _mm256_cvtsd_f64(div);
                        printf("div: %f ", a);
                    }
                    __m256d dtv = _mm256_set1_pd(dt);
                    if (i == 0 && ii == 0 && j == 1)   {
                        double a = _mm256_cvtsd_f64(dtv);
                        printf("dtv: %f ", a);
                    }

                    auto a = _mm256_mul_pd(div, div);
                    auto b = _mm256_mul_pd(a, div);
                    auto c = _mm256_mul_pd(b, dtv);
                    auto vdx = _mm256_mul_pd(b, dx);
                    auto vdy = _mm256_mul_pd(b, dy);

                    if (i == 0 && ii == 0 && j == 1)   {
                        double a = _mm256_cvtsd_f64(vdx);
                        printf("vdx: %f ", a);
                    }
                    if (i == 0 && ii == 0 && j == 1)   {
                        double a = _mm256_cvtsd_f64(vdy);  
                        printf("vdy: %f ", a);
                    }

                    __m256d temp = _mm256_hadd_pd(vdx, vdy);
                    __m256d shuffled = _mm256_permute4x64_pd(temp, _MM_SHUFFLE(3, 1, 2, 0));
                    __m256d hadd = _mm256_hadd_pd(shuffled, shuffled);
                    __m128d result = _mm256_castpd256_pd128(hadd);


                    if (i == 0 && ii == 0 && j == 1)   {
                        double a = _mm_cvtsd_f64(result);
                        printf("dvx: %f ", a);
                    }
                    if (i == 0 && ii == 0 && j == 1)   {
                        double a = _mm_cvtsd_f64(_mm_unpackhi_pd(result, result));  
                        printf("dvy: %f ", a);
                    }

                    vx += _mm_cvtsd_f64(result);
                    vy += _mm_cvtsd_f64(_mm_unpackhi_pd(result, result));  
                }
                x += dt * vx;
                y += dt * vy;
    
                velo[i].vx[ii] = vx;
                velo[i].vy[ii] = vy;
                nextcords[i].x[ii] = x;
                nextcords[i].y[ii] = y;
            }
        }
        tmp = cords;
        cords = nextcords;
        nextcords = tmp;
    }

    gettimeofday(&end, NULL);
    printf("Total time to run simulation %0.6f seconds, final location %f %f\n", tdiff(&start, &end), cords[nplanets_chunks-1].x[3], cords[nplanets_chunks-1].y[3]);
    free(cords);
    free(nextcords);
    return 0;    
}