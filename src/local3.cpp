// Independent d=3 local engine for Levine's two-player hat game (frame objective).
// Labels i=0..13 <-> (a,b), a=i/2+1 in 1..7, b=i%2. Point x in 0..7.
// Label covers x iff parity(a&x) ^ b == 1.
// For every set partition P of the 14 labels (restricted growth strings, lex order),
//   N(P) = 2^14 * E max_x #{retained labels covering x}
// where each block is retained independently with prob 1/2.  F = N / (14*2^14).
// Outputs: Ntab.bin (uint32 per partition in lex RGS order), PM.bin (prefix maxima,
// levels 0..13 concatenated), top.bin (RGS of partitions with N >= 81664), stats on stdout.
#include <bits/stdc++.h>
#include <immintrin.h>
using namespace std;
typedef uint64_t u64; typedef uint32_t u32;

const int NL = 14;
int LV[NL][8];
vector<u32> childStart[NL];
vector<uint8_t> kval[NL];
u64 C[NL + 1];
u32* Ntab;

static inline u32 hmin(__m128i w) { return (u32)_mm_cvtsi128_si32(_mm_minpos_epu16(w)) & 0xFFFF; }

// Sum over all subsets S of blocks of max_x sum_{B in S} vec[B][x].
static u64 evalM(int K, const uint16_t (*vec)[8]) {
    int r = K < 4 ? K : 4, outer = K - r, nin = 1 << r;
    alignas(16) uint16_t tmp[8];
    __m128i T[16], P[NL];
    for (int t = 0; t < nin; t++) {
        for (int x = 0; x < 8; x++) { int s = 0; for (int i = 0; i < r; i++) if (t >> i & 1) s += vec[outer + i][x]; tmp[x] = s; }
        T[t] = _mm_load_si128((__m128i*)tmp);
    }
    for (int b = 0; b < outer; b++) { for (int x = 0; x < 8; x++) tmp[x] = vec[b][x]; P[b] = _mm_load_si128((__m128i*)tmp); }
    const u32 BASE = 1000;
    __m128i w = _mm_set1_epi16(BASE);
    u64 acc = 0;
    auto inner = [&](__m128i w) -> u32 { u32 s = 0; for (int t = 0; t < nin; t++) s += hmin(_mm_sub_epi16(w, T[t])); return s; };
    acc += inner(w);
    u32 lim = 1u << outer;
    for (u32 s = 1; s < lim; s++) {
        int b = __builtin_ctz(s); u32 g = s ^ (s >> 1);
        if ((g >> b) & 1) w = _mm_sub_epi16(w, P[b]); else w = _mm_add_epi16(w, P[b]);
        acc += inner(w);
    }
    return (u64)BASE * ((u64)1 << K) - acc;
}

struct Worker {
    uint16_t vec[NL][8];
    int rgs[NL];
    vector<u64> hist;
    vector<array<uint8_t, 18>> top;  // 4 bytes idx + 14 rgs
    Worker() : hist(7 * 16384 + 1, 0) { memset(vec, 0, sizeof vec); }
};

void dfs(Worker& W, int j, u32 idx, int k) {
    if (j == NL) {
        u64 M = evalM(k, W.vec);
        u32 N = (u32)(M << (NL - k));
        Ntab[idx] = N;
        W.hist[N]++;
        if (N >= 81664) {
            array<uint8_t, 18> rec; memcpy(rec.data(), &idx, 4);
            for (int i = 0; i < NL; i++) rec[4 + i] = W.rgs[i];
            W.top.push_back(rec);
        }
        return;
    }
    u32 cs = childStart[j][idx];
    for (int c = 0; c <= k; c++) {
        for (int x = 0; x < 8; x++) W.vec[c][x] += LV[j][x];
        W.rgs[j] = c;
        dfs(W, j + 1, cs + c, c == k ? k + 1 : k);
        for (int x = 0; x < 8; x++) W.vec[c][x] -= LV[j][x];
    }
}

struct Task { u32 idx; int k; int rgs[6]; };
vector<Task> tasks;
void gen(int j, u32 idx, int k, int* rgs) {
    if (j == 6) { Task t; t.idx = idx; t.k = k; memcpy(t.rgs, rgs, sizeof t.rgs); tasks.push_back(t); return; }
    u32 cs = childStart[j][idx];
    for (int c = 0; c <= k; c++) { rgs[j] = c; gen(j + 1, cs + c, c == k ? k + 1 : k, rgs); }
}

int main(int argc, char** argv) {
    string out = argc > 1 ? argv[1] : ".";
    for (int i = 0; i < NL; i++) { int a = i / 2 + 1, b = i % 2; for (int x = 0; x < 8; x++) LV[i][x] = (__builtin_popcount(a & x) & 1) ^ b; }
    // level structure
    kval[0] = {0}; C[0] = 1;
    for (int j = 0; j < NL; j++) {
        childStart[j].resize(C[j]);
        u64 s = 0;
        for (u64 i = 0; i < C[j]; i++) { childStart[j][i] = (u32)s; s += kval[j][i] + 1; }
        C[j + 1] = s;
        if (j + 1 < NL) {
            kval[j + 1].resize(s); u64 p = 0;
            for (u64 i = 0; i < C[j]; i++) { int k = kval[j][i]; for (int c = 0; c <= k; c++) kval[j + 1][p++] = (c < k ? k : k + 1); }
        }
    }
    printf("Bell(14) from level structure: %llu (expect 190899322)\n", (unsigned long long)C[NL]);
    Ntab = (u32*)malloc(sizeof(u32) * C[NL]);
    int rg[NL]; gen(0, 0, 0, rg);
    int nth = max(1u, thread::hardware_concurrency());
    vector<Worker*> ws(nth);
    atomic<int> next(0);
    auto t0 = chrono::steady_clock::now();
    vector<thread> th;
    for (int t = 0; t < nth; t++) {
        ws[t] = new Worker();
        th.emplace_back([&, t] {
            Worker& W = *ws[t];
            for (;;) {
                int i = next++;
                if (i >= (int)tasks.size()) break;
                Task& T = tasks[i];
                memset(W.vec, 0, sizeof W.vec);
                for (int j = 0; j < 6; j++) { W.rgs[j] = T.rgs[j]; for (int x = 0; x < 8; x++) W.vec[T.rgs[j]][x] += LV[j][x]; }
                dfs(W, 6, T.idx, T.k);
            }
        });
    }
    for (auto& x : th) x.join();
    double secs = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
    vector<u64> hist(7 * 16384 + 1, 0);
    vector<array<uint8_t, 18>> top;
    for (auto w : ws) { for (size_t i = 0; i < hist.size(); i++) hist[i] += w->hist[i]; top.insert(top.end(), w->top.begin(), w->top.end()); }
    u64 tot = 0; for (auto h : hist) tot += h;
    printf("enumerated %llu partitions in %.1fs with %d threads\n", (unsigned long long)tot, secs, nth);
    int mx = 0, mn = 1 << 30; for (int i = 0; i < (int)hist.size(); i++) if (hist[i]) { mx = max(mx, i); mn = min(mn, i); }
    printf("max N = %d (= %d/896 * 256? %s), min N = %d\n", mx, mx / 256, mx % 256 ? "no" : "yes", mn);
    printf("All N values >= 81152 (score >= 317/896), with counts:\n");
    u64 cum = 0;
    for (int i = mx; i >= 81152; i--) if (hist[i]) { cum += hist[i]; printf("  N=%d  score=%.4f/896  count=%llu  cumulative=%llu\n", i, i / 256.0, (unsigned long long)hist[i], (unsigned long long)cum); }
    // prefix maxima
    vector<vector<u32>> PM(NL);
    PM[NL - 1].resize(C[NL - 1]);
    for (u64 i = 0; i < C[NL - 1]; i++) { u32 cs = childStart[NL - 1][i], m = 0; for (int c = 0; c <= kval[NL - 1][i]; c++) m = max(m, Ntab[cs + c]); PM[NL - 1][i] = m; }
    for (int j = NL - 2; j >= 0; j--) {
        PM[j].resize(C[j]);
        for (u64 i = 0; i < C[j]; i++) { u32 cs = childStart[j][i], m = 0; for (int c = 0; c <= kval[j][i]; c++) m = max(m, PM[j + 1][cs + c]); PM[j][i] = m; }
    }
    printf("PM root = %u\n", PM[0][0]);
    FILE* f = fopen((out + "/Ntab.bin").c_str(), "wb"); fwrite(Ntab, 4, C[NL], f); fclose(f);
    f = fopen((out + "/PM.bin").c_str(), "wb"); for (int j = 0; j < NL; j++) fwrite(PM[j].data(), 4, C[j], f); fclose(f);
    sort(top.begin(), top.end(), [](auto& a, auto& b) { u32 x, y; memcpy(&x, a.data(), 4); memcpy(&y, b.data(), 4); return x < y; });
    f = fopen((out + "/top.bin").c_str(), "wb"); for (auto& r : top) fwrite(r.data(), 1, 18, f); fclose(f);
    f = fopen((out + "/hist.txt").c_str(), "w"); for (int i = 0; i < (int)hist.size(); i++) if (hist[i]) fprintf(f, "%d %llu\n", i, (unsigned long long)hist[i]); fclose(f);
    printf("wrote tables; top records: %zu\n", top.size());
    return 0;
}
