// d=5 branch-and-bound for Levine's two-player game (common-row objective H_5).
//
// Labels i=0..61: (a,b), a=i/2+1 in 1..31, b=i%2; label covers x in F2^5 iff <a,x>^b = 1.
// H_5(P) = (1/62) E max_x #retained labels covering x.
// Hyperplanes T_n = {a : <a,n> = 0}, n in 1..31: H_5(P) <= (1/31) sum_T H_4(P|T).
// Frames (3-dim subspaces, 155): H_4(P|T) <= (1/15) sum_{W in T} F_W.
//
// Claim checked for target tau = m/13440:   max_P H_5(P) <= max(tau, best exact leaf).
//  * Every P has a hyperplane with maximal H_4(P|T). If that maximum is <= tau we are done.
//    Otherwise, by symmetry (AGL(5,2) x swap, transitive on hyperplanes and inducing the full
//    AGL(4,2) x swap on each), it is T_16 (labels 0..29) and P|T_16 is one of the given orbit
//    reps, which must include every d=4 orbit with H_4 > tau (checked by the caller).
//  * Outside labels 30..61 are assigned in order. Each non-root hyperplane is bounded by
//    min(h0, frame bound, exact H_4 once complete); frames use the prefix-max table.
//  * Prune when h0 + sum of bounds <= 31 tau.  Symmetry: the 32-element kernel acting
//    regularly on outside labels lets us require type(label 30) = min type (see below).
#include <bits/stdc++.h>
#include <immintrin.h>
using namespace std;
typedef uint64_t u64; typedef uint32_t u32; typedef int64_t i64; typedef __int128 i128;

const int NL = 14;
u64 C[NL + 1], PMoff[NL + 1];
vector<u32> childStart[NL];
vector<uint8_t> kval[NL];
u32* Ntab; u32* PMall;
inline u32 PMat(int j, u32 idx) { return j == NL ? Ntab[idx] : PMall[PMoff[j] + idx]; }
static int par(int v) { return __builtin_popcount(v) & 1; }

void buildLevels() {
    kval[0] = {0}; C[0] = 1;
    for (int j = 0; j < NL; j++) {
        childStart[j].resize(C[j]); u64 s = 0;
        for (u64 i = 0; i < C[j]; i++) { childStart[j][i] = (u32)s; s += kval[j][i] + 1; }
        C[j + 1] = s;
        if (j + 1 < NL) { kval[j + 1].resize(s); u64 p = 0; for (u64 i = 0; i < C[j]; i++) { int k = kval[j][i]; for (int c = 0; c <= k; c++) kval[j + 1][p++] = (c < k ? k : k + 1); } }
    }
    PMoff[0] = 0; for (int j = 0; j < NL; j++) PMoff[j + 1] = PMoff[j] + C[j];
}

// ---------- exact evaluation ----------
static inline u32 hmin(__m128i w) { return (u32)_mm_cvtsi128_si32(_mm_minpos_epu16(w)) & 0xFFFF; }
template <int NP>  // NP = 16 or 32 points
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
    auto inner = [&]() -> u32 {
        u32 s = 0;
        for (int t = 0; t < nin; t++) {
            __m128i m = _mm_sub_epi16(w[0], T[t][0]);
            for (int q = 1; q < R; q++) m = _mm_min_epu16(m, _mm_sub_epi16(w[q], T[t][q]));
            s += hmin(m);
        }
        return s;
    };
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

// ---------- d=5 geometry ----------
const int NLAB = 62;
uint16_t GV5[NLAB][32];
struct NRF { int8_t cpos[NLAB]; int shared[6]; int hyp[3]; int nh = 0; };
vector<NRF> nrf;                 // 140 non-root frames
vector<array<int, 3>> rootFrameBasis;  // 15 frames inside T_16
vector<int> FofL[NLAB], HofL[NLAB];
int hypNormal[30], hypRF[30], hypNR[30][14], hypLab[30][30];
uint16_t HV[30][30][16];         // cover vectors of hyperplane labels over 16 representative points
int rootLab[30]; uint16_t RV[30][16];

void buildGeometry() {
    for (int i = 0; i < NLAB; i++) { int a = i / 2 + 1, b = i % 2; for (int x = 0; x < 32; x++) GV5[i][x] = par(a & x) ^ b; }
    // all 3-dim subspaces
    set<array<int, 7>> subs;
    for (int u = 1; u < 32; u++) for (int v = u + 1; v < 32; v++) for (int w = v + 1; w < 32; w++) {
        if ((u ^ v) == w) continue;
        array<int, 7> s = {u, v, w, u ^ v, u ^ w, v ^ w, u ^ v ^ w}; sort(s.begin(), s.end());
        if (s[0] == 0) continue;
        subs.insert(s);
    }
    assert(subs.size() == 155);
    vector<array<int, 7>> nrList;
    for (auto& s : subs) {
        bool inRoot = true; for (int a : s) if (a & 16) inRoot = false;
        if (inRoot) { // basis: pick 3 independent
            int u = s[0], v = s[1], w = -1; for (int a : s) if (a != u && a != v && a != (u ^ v)) { w = a; break; }
            rootFrameBasis.push_back({u, v, w});
        } else nrList.push_back(s);
    }
    assert(rootFrameBasis.size() == 15 && nrList.size() == 140);
    for (auto& s : nrList) {
        NRF F; memset(F.cpos, -1, sizeof F.cpos);
        vector<int> out; for (int a : s) if (a & 16) out.push_back(a);
        sort(out.begin(), out.end()); assert(out.size() == 4 && (out[0] ^ out[1] ^ out[2]) == out[3]);
        int w3 = out[0], w1 = out[0] ^ out[1], w2 = out[0] ^ out[2];
        for (int a : s) {
            int phi = -1;
            for (int c = 0; c < 8; c++) if ((((c & 1) ? w1 : 0) ^ ((c & 2) ? w2 : 0) ^ ((c & 4) ? w3 : 0)) == a) phi = c;
            assert(phi > 0);
            for (int b = 0; b < 2; b++) {
                int i = 2 * (a - 1) + b; F.cpos[i] = 2 * (phi - 1) + b;
                // cover-vector equivalence check
                for (int x = 0; x < 32; x++) { int px = par(w1 & x) | (par(w2 & x) << 1) | (par(w3 & x) << 2); assert(GV5[i][x] == (par((phi) & px) ^ b)); }
            }
        }
        int expect = 6;
        for (int i = 0; i < NLAB; i++) if (F.cpos[i] >= 0) {
            if (i < 30) { assert(F.cpos[i] < 6); F.shared[F.cpos[i]] = i; }
            else { assert(F.cpos[i] == expect); expect++; }
        }
        assert(expect == 14);
        nrf.push_back(F);
    }
    for (int f = 0; f < 140; f++) for (int i = 30; i < NLAB; i++) if (nrf[f].cpos[i] >= 0) FofL[i].push_back(f);
    // hyperplanes
    int h = 0;
    for (int n = 1; n < 32; n++) {
        if (n == 16) continue;
        hypNormal[h] = n;
        int jb = __builtin_ctz(n); int npt = 0; int pts[16];
        for (int x = 0; x < 32; x++) if (!(x >> jb & 1)) pts[npt++] = x;
        int nl = 0;
        for (int i = 0; i < NLAB; i++) { int a = i / 2 + 1; if (!par(a & n)) { hypLab[h][nl] = i; for (int p = 0; p < 16; p++) HV[h][nl][p] = GV5[i][pts[p]]; nl++; } }
        assert(nl == 30);
        for (int i = 30; i < NLAB; i++) { int a = i / 2 + 1; if (!par(a & n)) HofL[i].push_back(h); }
        int nnr = 0;
        for (int f = 0; f < 140; f++) {
            bool in = true; for (int i = 0; i < NLAB; i++) if (nrf[f].cpos[i] >= 0 && par((i / 2 + 1) & n)) in = false;
            if (in) { hypNR[h][nnr++] = f; nrf[f].hyp[nrf[f].nh++] = h; }
        }
        assert(nnr == 14);
        hypRF[h] = -1;
        for (int r = 0; r < 15; r++) { auto& B = rootFrameBasis[r]; if (!par(B[0] & n) && !par(B[1] & n) && !par(B[2] & n)) { assert(hypRF[h] < 0); hypRF[h] = r; } }
        assert(hypRF[h] >= 0);
        h++;
    }
    for (int f = 0; f < 140; f++) assert(nrf[f].nh == 3);
    for (int i = 30; i < NLAB; i++) assert(FofL[i].size() == 35 && HofL[i].size() == 15);
    for (int i = 0; i < 30; i++) { rootLab[i] = i; for (int x = 0; x < 16; x++) RV[i][x] = GV5[i][x]; }  // points x<16
}

u32 frameNbasis(const int* asg, const array<int, 3>& B) {
    int arr[14];
    for (int c = 1; c < 8; c++) {
        int a = ((c & 1) ? B[0] : 0) ^ ((c & 2) ? B[1] : 0) ^ ((c & 4) ? B[2] : 0);
        for (int b = 0; b < 2; b++) arr[2 * (c - 1) + b] = asg[2 * (a - 1) + b];
    }
    int mp[64]; memset(mp, -1, sizeof mp); int k = 0; u32 idx = 0;
    for (int p = 0; p < 14; p++) { if (mp[arr[p]] < 0) mp[arr[p]] = k++; idx = childStart[p][idx] + mp[arr[p]]; }
    return Ntab[idx];
}
// exact H4 of a set of 30 labels with given 16-point cover vectors; returns units U (or -1 if K>24)
const i64 UNIT_H = (i64)1 << 25;  // U = 1/(2^25 * 105)
i64 exactH4U(const int* asg, const int* labs, const uint16_t (*cv)[16]) {
    int mp[64]; memset(mp, -1, sizeof mp); int K = 0;
    uint16_t vec[30][16]; memset(vec, 0, sizeof vec);
    for (int t = 0; t < 30; t++) { int g = asg[labs[t]]; if (mp[g] < 0) mp[g] = K++; for (int x = 0; x < 16; x++) vec[mp[g]][x] += cv[t][x]; }
    if (K > 24) return -1;
    u64 M = evalM<16>(K, vec);
    return (i64)M * 7 * ((i64)1 << (24 - K));
}

// ---------- search ----------
i64 tauU, thr31; int tauM;
struct Root { string val; vector<int> rgs; };
vector<Root> roots;

struct Worker {
    int asg[NLAB], K, K0;
    struct FS { u32 idx; int8_t k, j; int8_t map[64]; } fr[140];
    u32 fub[140];
    i64 hFS[30], hEx[30], hUB[30], S, h0U; int hcnt[30];
    u32 rootFN[15];
    struct Und { u32 idx; int8_t k, j, newmap; u32 fub; } und[NLAB][35];
    struct HS { i64 hFS[30], hEx[30], hUB[30], S; int hcnt[30]; int K; } hsave[NLAB];
    u64 nodes = 0, leaves = 0, viol = 0, hypExact = 0;
    i128 bestNum = 0; int bestK = 0; vector<int> bestAsg;   // best H5 = bestNum / (62 * 2^bestK)
    u64 consistencyFail = 0;

    void recomputeS() {
        i64 s = h0U;
        for (int h = 0; h < 30; h++) { i64 ub = min(h0U, hFS[h] * 1024); if (hEx[h] >= 0) ub = min(ub, hEx[h]); hUB[h] = ub; s += ub; }
        S = s;
    }
    bool initRoot(const Root& R) {
        for (int i = 0; i < 30; i++) asg[i] = R.rgs[i];
        K0 = K = *max_element(R.rgs.begin(), R.rgs.end()) + 1;
        h0U = exactH4U(asg, rootLab, RV);
        for (int r = 0; r < 15; r++) rootFN[r] = frameNbasis(asg, rootFrameBasis[r]);
        for (int f = 0; f < 140; f++) {
            FS& S_ = fr[f]; memset(S_.map, -1, sizeof S_.map); S_.k = 0; S_.idx = 0;
            for (int p = 0; p < 6; p++) { int g = asg[nrf[f].shared[p]]; if (S_.map[g] < 0) S_.map[g] = S_.k++; S_.idx = childStart[p][S_.idx] + S_.map[g]; }
            S_.j = 6; fub[f] = PMat(6, S_.idx);
        }
        for (int h = 0; h < 30; h++) {
            hFS[h] = rootFN[hypRF[h]]; for (int t = 0; t < 14; t++) hFS[h] += fub[hypNR[h][t]];
            hcnt[h] = 0; hEx[h] = -1;
        }
        recomputeS();
        return S > thr31;
    }
    inline int type(int c) const { return c < K0 ? c : K0; }
    bool push(int i, int c) {
        nodes++;
        HS& H = hsave[i]; memcpy(H.hFS, hFS, sizeof hFS); memcpy(H.hEx, hEx, sizeof hEx); memcpy(H.hUB, hUB, sizeof hUB); memcpy(H.hcnt, hcnt, sizeof hcnt); H.S = S; H.K = K;
        int t = 0;
        for (int f : FofL[i]) {
            FS& F = fr[f]; Und& U = und[i][t++];
            U.idx = F.idx; U.k = F.k; U.j = F.j; U.newmap = 0; U.fub = fub[f];
            int lc = F.map[c]; if (lc < 0) { lc = F.k; F.map[c] = lc; F.k++; U.newmap = 1; }
            F.idx = childStart[F.j][F.idx] + lc; F.j++;
            u32 v = PMat(F.j, F.idx);
            i64 d = (i64)v - (i64)fub[f]; fub[f] = v;
            for (int q = 0; q < 3; q++) hFS[nrf[f].hyp[q]] += d;
        }
        asg[i] = c; if (c == K) K++;
        bool ok = true;
        for (int h : HofL[i]) {
            if (++hcnt[h] == 16) {
                hypExact++;
                hEx[h] = exactH4U(asg, hypLab[h], HV[h]);
                if (hEx[h] > h0U) ok = false;   // P's best hyperplane is not the root: covered elsewhere
            }
        }
        recomputeS();
        return ok && S > thr31;
    }
    void pop(int i) {
        int t = 0;
        for (int f : FofL[i]) { Und& U = und[i][t++]; FS& F = fr[f]; if (U.newmap) F.map[asg[i]] = -1; F.idx = U.idx; F.k = U.k; F.j = U.j; fub[f] = U.fub; }
        HS& H = hsave[i]; memcpy(hFS, H.hFS, sizeof hFS); memcpy(hEx, H.hEx, sizeof hEx); memcpy(hUB, H.hUB, sizeof hUB); memcpy(hcnt, H.hcnt, sizeof hcnt); S = H.S; K = H.K;
    }
    void leaf() {
        leaves++;
        uint16_t vec[64][32]; memset(vec, 0, sizeof vec);
        for (int i = 0; i < NLAB; i++) for (int x = 0; x < 32; x++) vec[asg[i]][x] += GV5[i][x];
        u64 M = evalM<32>(K, vec);
        // consistency: H5 <= S/31  <=>  M * 2^25 * 105 <= S * 2^(K+1)
        if ((i128)M * ((i128)UNIT_H * 105) > (i128)S * ((i128)1 << (K + 1))) consistencyFail++;
        if ((i128)M << bestK > bestNum << K || bestAsg.empty()) { bestNum = M; bestK = K; bestAsg.assign(asg, asg + NLAB); }
        if ((i128)M * 13440 > (i128)tauM * 62 * ((i128)1 << K)) viol++;
    }
    void dfs(int i) {
        if (i == NLAB) { leaf(); return; }
        int t30 = i > 30 ? type(asg[30]) : 0;
        for (int c = 0; c <= K; c++) {
            if (i > 30 && type(c) < t30) continue;   // kernel symmetry breaking
            if (push(i, c)) dfs(i + 1);
            pop(i);
        }
    }
};

struct Task { int root; vector<int> pre; };

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "usage: bb5 datadir tauM rootsfile outprefix [D] [maxTasks]\n"); return 1; }
    string dir = argv[1]; tauM = atoi(argv[2]); string rf = argv[3]; string outp = argv[4];
    int D = argc > 5 ? atoi(argv[5]) : 3; long maxTasks = argc > 6 ? atol(argv[6]) : -1;
    tauU = (i64)tauM << 18; thr31 = 31 * tauU;
    buildLevels(); buildGeometry();
    Ntab = (u32*)malloc(4 * C[NL]); PMall = (u32*)malloc(4 * PMoff[NL]);
    { FILE* f = fopen((dir + "/Ntab.bin").c_str(), "rb"); if (fread(Ntab, 4, C[NL], f) != C[NL]) return 2; fclose(f); }
    { FILE* f = fopen((dir + "/PM.bin").c_str(), "rb"); if (fread(PMall, 4, PMoff[NL], f) != PMoff[NL]) return 2; fclose(f); }
    // self-test: heuristic d=5 partition (Python: H5 = 1401/3968)
    {
        int w[62] = {0,1,0,1,2,3,0,4,1,4,1,4,0,4,3,1,0,1,0,1,1,2,0,4,1,4,1,4,0,4,5,6,0,1,0,1,2,3,0,4,1,4,1,4,0,4,3,1,0,1,0,1,1,2,0,4,1,4,1,4,0,4};
        uint16_t vec[64][32]; memset(vec, 0, sizeof vec); int K = 0; for (int i = 0; i < 62; i++) { K = max(K, w[i] + 1); for (int x = 0; x < 32; x++) vec[w[i]][x] += GV5[i][x]; }
        u64 M = evalM<32>(K, vec);
        printf("selftest: H5 = %llu/(62*2^%d) = %.9f (python 1401/3968 = %.9f)\n", (unsigned long long)M, K, M / (62.0 * (1 << K)), 1401.0 / 3968);
        double mn = 1; for (int h = 0; h < 30; h++) mn = min(mn, (double)exactH4U(w, hypLab[h], HV[h]) / (UNIT_H * 105.0));
        printf("  min hyperplane H4 = %.9f (python 0.342708333)\n", mn);
    }
    { FILE* f = fopen(rf.c_str(), "r"); char val[64]; int sz;
      while (fscanf(f, "%63s %d", val, &sz) == 2) { Root R; R.val = val; R.rgs.resize(30); for (int i = 0; i < 30; i++) if (fscanf(f, "%d", &R.rgs[i]) != 1) return 3; roots.push_back(R); }
      fclose(f); }
    printf("tau = %d/13440 = %.9f; roots = %zu\n", tauM, tauM / 13440.0, roots.size());
    vector<Task> tasks;
    {
        Worker* W = new Worker();
        for (int r = 0; r < (int)roots.size(); r++) {
            if (!W->initRoot(roots[r])) { printf("root %d pruned at start (h0=%.9f)\n", r, W->h0U / (UNIT_H * 105.0)); continue; }
            vector<int> pre;
            function<void(int)> gen = [&](int i) {
                if (i == 30 + D) { tasks.push_back({r, pre}); return; }
                int t30 = i > 30 ? W->type(W->asg[30]) : 0;
                for (int c = 0; c <= W->K; c++) {
                    if (i > 30 && W->type(c) < t30) continue;
                    if (W->push(i, c)) { pre.push_back(c); gen(i + 1); pre.pop_back(); }
                    W->pop(i);
                }
            };
            gen(30);
        }
        delete W;
    }
    printf("total tasks generated: %zu\n", tasks.size()); fflush(stdout);
    if (maxTasks > 0 && (long)tasks.size() > maxTasks) {  // sampling mode for timing estimates
        mt19937 rng(1); shuffle(tasks.begin(), tasks.end(), rng); tasks.resize(maxTasks);
        printf("SAMPLING %ld tasks only (not a proof run)\n", maxTasks);
    }
    printf("tasks: %zu\n", tasks.size()); fflush(stdout);
    int nth = max(1u, thread::hardware_concurrency());
    vector<Worker*> ws(nth); atomic<int> next(0), done(0);
    auto t0 = chrono::steady_clock::now();
    vector<thread> th;
    for (int t = 0; t < nth; t++) {
        ws[t] = new Worker();
        th.emplace_back([&, t] {
            Worker& W = *ws[t];
            for (;;) {
                int k = next++; if (k >= (int)tasks.size()) break;
                Task& T = tasks[k];
                W.initRoot(roots[T.root]);
                bool alive = true; int i = 30;
                for (; i < 30 + (int)T.pre.size(); i++) if (!W.push(i, T.pre[i - 30])) { alive = false; i++; break; }
                if (alive) W.dfs(i);
                for (int j = i - 1; j >= 30; j--) W.pop(j);
                int d = ++done;
                if (d % 100 == 0 || d == (int)tasks.size()) {
                    double s = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
                    u64 nn = 0; for (auto w : ws) if (w) nn += w->nodes;
                    fprintf(stderr, "[%.0fs] tasks %d/%zu nodes %.3g\n", s, d, tasks.size(), (double)nn);
                }
            }
        });
    }
    for (auto& x : th) x.join();
    double secs = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
    u64 nodes = 0, leaves = 0, viol = 0, he = 0, cf = 0; Worker* bw = nullptr;
    for (auto w : ws) { nodes += w->nodes; leaves += w->leaves; viol += w->viol; he += w->hypExact; cf += w->consistencyFail;
        if (w->leaves && (!bw || ((i128)w->bestNum << bw->bestK) > (bw->bestNum << w->bestK))) bw = w; }
    printf("done in %.1fs: nodes=%llu hyperplane-exacts=%llu leaves=%llu violations(H5>tau)=%llu consistencyFail=%llu\n", secs,
           (unsigned long long)nodes, (unsigned long long)he, (unsigned long long)leaves, (unsigned long long)viol, (unsigned long long)cf);
    if (bw) { printf("best leaf H5 = %.9f asg:", (double)bw->bestNum / (62.0 * (double)(1ull << bw->bestK))); for (int x : bw->bestAsg) printf(" %d", x); printf("\n"); }
    return 0;
}
