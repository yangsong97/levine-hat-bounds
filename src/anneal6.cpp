// Simulated annealing for max H_d (d = 4 or 5): lower bounds on the relaxation value.
// Labels (a,b), a in 1..2^d-1, b in {0,1}; label covers x iff <a,x>^b = 1.
// H_d(P) = E max_x #(retained labels covering x) / (2(2^d-1)), blocks retained w.p. 1/2.
// Usage: anneal d seconds seed [Kmax]
#include <bits/stdc++.h>
#include <immintrin.h>
using namespace std;
typedef uint64_t u64; typedef uint32_t u32;
static int par(int v) { return __builtin_popcount(v) & 1; }
static inline u32 hmin(__m128i w) { return (u32)_mm_cvtsi128_si32(_mm_minpos_epu16(w)) & 0xFFFF; }
template <int NP>
u64 evalM(int K, const uint16_t (*vec)[NP]) {
    constexpr int R = NP / 8;
    int r = K < 4 ? K : 4, outer = K - r, nin = 1 << r;
    alignas(16) uint16_t tmp[NP];
    __m128i T[16][R], P[64][R];
    for (int t = 0; t < nin; t++) {
        for (int x = 0; x < NP; x++) { int s = 0; for (int i = 0; i < r; i++) if (t >> i & 1) s += vec[outer + i][x]; tmp[x] = s; }
        for (int q = 0; q < R; q++) T[t][q] = _mm_load_si128((__m128i*)(tmp + 8 * q));
    }
    for (int b = 0; b < outer; b++) { for (int x = 0; x < NP; x++) tmp[x] = vec[b][x]; for (int q = 0; q < R; q++) P[b][q] = _mm_load_si128((__m128i*)(tmp + 8 * q)); }
    const u32 BASE = 1000;
    __m128i w[R]; for (int q = 0; q < R; q++) w[q] = _mm_set1_epi16(BASE);
    u64 acc = 0;
    auto inner = [&]() -> u32 { u32 s = 0; for (int t = 0; t < nin; t++) { __m128i m = _mm_sub_epi16(w[0], T[t][0]); for (int q = 1; q < R; q++) m = _mm_min_epu16(m, _mm_sub_epi16(w[q], T[t][q])); s += hmin(m); } return s; };
    acc += inner();
    u64 lim = 1ull << outer;
    for (u64 s = 1; s < lim; s++) {
        int b = __builtin_ctzll(s); u64 g = s ^ (s >> 1);
        if ((g >> b) & 1) for (int q = 0; q < R; q++) w[q] = _mm_sub_epi16(w[q], P[b][q]);
        else for (int q = 0; q < R; q++) w[q] = _mm_add_epi16(w[q], P[b][q]);
        acc += inner();
    }
    return (u64)BASE * (1ull << K) - acc;
}

int D, NLAB, NPT; uint16_t GV[126][64];
// exact value as a double (M / 2^K / NLAB); also returns M,K
template <int NP> double score(const vector<int>& asg, int Kmax, u64* Mout = nullptr, int* Kout = nullptr) {
    int mp[64]; for (int i = 0; i < Kmax; i++) mp[i] = -1; int K = 0;
    uint16_t vec[64][NP];
    for (int i = 0; i < NLAB; i++) { int g = asg[i]; if (mp[g] < 0) { mp[g] = K; memset(vec[K], 0, sizeof vec[K]); K++; } for (int x = 0; x < NP; x++) vec[mp[g]][x] += GV[i][x]; }
    u64 M = evalM<NP>(K, vec);
    if (Mout) { *Mout = M; *Kout = K; }
    return (double)M / (double)(1ull << K) / NLAB;
}
double sc(const vector<int>& a, int Kmax, u64* M = nullptr, int* K = nullptr) { return NPT == 16 ? score<16>(a, Kmax, M, K) : NPT == 32 ? score<32>(a, Kmax, M, K) : score<64>(a, Kmax, M, K); }

int main(int argc, char** argv) {
    D = atoi(argv[1]); double secs = atof(argv[2]); int seed = atoi(argv[3]); int Kmax = argc > 4 ? atoi(argv[4]) : 12;
    NLAB = 2 * ((1 << D) - 1); NPT = 1 << D;
    for (int i = 0; i < NLAB; i++) { int a = i / 2 + 1, b = i % 2; for (int x = 0; x < NPT; x++) GV[i][x] = par(a & x) ^ b; }
    vector<int> lift; if (getenv("LIFT")) { ifstream f(getenv("LIFT")); int v; while (f >> v) lift.push_back(v); if ((int)lift.size() != 2 * ((1 << (D - 1)) - 1)) { fprintf(stderr, "bad lift size\n"); return 1; } }
    int nth = max(1u, thread::hardware_concurrency());
    mutex mu; double best = 0; vector<int> bestA; u64 bestM = 0; int bestK = 0; long restarts = 0;
    auto t0 = chrono::steady_clock::now();
    auto elapsed = [&]() { return chrono::duration<double>(chrono::steady_clock::now() - t0).count(); };
    vector<thread> th;
    for (int t = 0; t < nth; t++) th.emplace_back([&, t] {
        mt19937_64 rng(seed * 1000 + t);
        while (elapsed() < secs) {
            // random start with k blocks
            int k0 = 4 + rng() % 6;
            vector<int> a(NLAB); for (auto& v : a) v = rng() % k0;
            if (!lift.empty() && (rng() & 1)) {   // lift of a lower-dimensional partition: a = top + y copies (y, b ^ flip)
                int lo = (int)lift.size(), top = (lo / 2) + 1, flip = rng() & 1, K0 = *max_element(lift.begin(), lift.end()) + 1;
                for (int i = 0; i < lo; i++) a[i] = lift[i];
                for (int y = 1; y < top; y++) for (int b = 0; b < 2; b++) a[2 * (top + y - 1) + b] = lift[2 * (y - 1) + (b ^ flip)];
                a[2 * (top - 1)] = K0 % Kmax; a[2 * (top - 1) + 1] = (K0 + (rng() & 1)) % Kmax;
                for (int q = 0; q < (int)(rng() % 6); q++) a[rng() % NLAB] = rng() % Kmax;
            }
            double cur = sc(a, Kmax), lb = cur; vector<int> la = a;
            double T = 0.004, Tend = 0.00002; long steps = 60000;
            for (long s = 0; s < steps && elapsed() < secs; s++) {
                double temp = T * pow(Tend / T, (double)s / steps);
                int i = rng() % NLAB, old = a[i], nb = rng() % Kmax;
                if (nb == old) continue;
                // occasionally swap two labels' blocks instead
                int j = -1, oldj = -1;
                if (rng() % 4 == 0) { j = rng() % NLAB; oldj = a[j]; if (oldj == old) { j = -1; } else { a[j] = old; nb = oldj; } }
                a[i] = nb;
                double v = sc(a, Kmax);
                if (v >= cur || exp((v - cur) / temp) > (double)(rng() % 1000000) / 1e6) {
                    cur = v; if (v > lb) { lb = v; la = a; }
                } else { a[i] = old; if (j >= 0) a[j] = oldj; }
            }
            // final greedy polish from the best state of this run
            a = la; cur = lb; bool imp = true;
            while (imp) { imp = false; for (int i = 0; i < NLAB; i++) for (int nb = 0; nb < Kmax; nb++) { int old = a[i]; if (nb == old) continue; a[i] = nb; double v = sc(a, Kmax); if (v > cur + 1e-15) { cur = v; imp = true; } else a[i] = old; } }
            lock_guard<mutex> g(mu); restarts++;
            if (cur > best + 1e-15) { best = cur; bestA = a; sc(a, Kmax, &bestM, &bestK); fprintf(stderr, "[%.0fs] thread %d restart %ld: new best %.9f (M=%llu, K=%d)\n", elapsed(), t, restarts, best, (unsigned long long)bestM, bestK); }
        }
    });
    for (auto& x : th) x.join();
    // RGS of best
    int mp[64]; memset(mp, -1, sizeof mp); int k = 0; for (auto& v : bestA) { if (mp[v] < 0) mp[v] = k++; v = mp[v]; }
    printf("d=%d restarts=%ld best H=%.9f = %llu/(%d*2^%d)\nasg:", D, restarts, best, (unsigned long long)bestM, NLAB, bestK);
    for (int v : bestA) printf(" %d", v); printf("\n");
}
