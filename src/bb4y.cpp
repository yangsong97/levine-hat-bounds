// Branch-and-bound for the d=4 common-row objective H_4 (independent implementation).
//
// Global labels i=0..29: (a,b), a=i/2+1 in 1..15, b=i%2. Points x in 0..15.
// Frames: directions n in 1..15; frame(n) = labels with <a,n>=0 (14 labels).
// Root frame n0=8 = labels 0..13 (identical to canonical AG(3,2) labelling).
//
// Claim checked for threshold L:   max_P H_4(P) <= max( T_L , best exact H_4 found ),
//   T_L = (4860-L)/13440 = B - L/(15*896).
// Method: every partition has a frame of maximum local score; by symmetry (AGL(4,2) x swap)
// move it to n0 and its restriction to a canonical orbit rep.  Since every frame scores
// <= the root, sum of frame scores <= 15*N_root, so if 15*(82944-N_root) >= 256*L the
// frame bound already gives <= T_L; otherwise the root rep is in the input list.
// For each root rep, assign the 16 outside labels 14..29 (restricted growth),
// bounding each non-root frame by the max over completions of its partial local partition
// (prefix-max table), clamped at N_root.  Prune when sum <= 256*(4860-L).  At complete
// partitions that survive, compute H_4 exactly by Gray code over all block subsets.
#include <bits/stdc++.h>
#include <immintrin.h>
using namespace std;
typedef uint64_t u64; typedef uint32_t u32; typedef int64_t i64;

const int NL = 14;
u64 C[NL + 1], PMoff[NL + 1];
vector<u32> childStart[NL];
vector<uint8_t> kval[NL];
u32* Ntab; u32* PMall;
inline u32 PMat(int j, u32 idx) { return j == NL ? Ntab[idx] : PMall[PMoff[j] + idx]; }

int GV[30][16];
int cpos[14][30];          // canonical position of global label in non-root frame f, or -1
int frameNormal[14];
vector<int> FL[30];        // non-root frames containing outside label i
int sharedPos[14][6];      // global label at canonical positions 0..5

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
static int par(int v) { return __builtin_popcount(v) & 1; }

void buildFrames() {
    for (int i = 0; i < 30; i++) { int a = i / 2 + 1, b = i % 2; for (int x = 0; x < 16; x++) GV[i][x] = par(a & x) ^ b; }
    int f = 0;
    for (int n = 1; n < 16; n++) {
        if (n == 8) continue;
        frameNormal[f] = n;
        vector<int> out;
        for (int a = 8; a < 16; a++) if (!par(a & n)) out.push_back(a);
        assert(out.size() == 4);
        int w3 = out[0], w1 = out[0] ^ out[1], w2 = out[0] ^ out[2];
        for (int i = 0; i < 30; i++) cpos[f][i] = -1;
        for (int i = 0; i < 30; i++) {
            int a = i / 2 + 1, b = i % 2;
            if (par(a & n)) continue;
            int phi = -1;
            for (int c = 0; c < 8; c++) if ((((c & 1) ? w1 : 0) ^ ((c & 2) ? w2 : 0) ^ ((c & 4) ? w3 : 0)) == a) phi = c;
            assert(phi > 0);
            cpos[f][i] = 2 * (phi - 1) + b;
        }
        // checks: shared labels -> 0..5, outside labels in global order -> 6..13
        int expect = 6; vector<int> sh;
        for (int i = 0; i < 30; i++) if (cpos[f][i] >= 0) {
            if (i < 14) { assert(cpos[f][i] < 6); sharedPos[f][cpos[f][i]] = i; sh.push_back(i); }
            else { assert(cpos[f][i] == expect); expect++; FL[i].push_back(f); }
        }
        assert(expect == 14 && sh.size() == 6);
        // the frame value must match: label's cover vector on 16 points equals canonical label's
        // cover vector pulled back by psi(x) = (<w1,x>,<w2,x>,<w3,x>)
        for (int i = 0; i < 30; i++) if (cpos[f][i] >= 0) {
            int ca = cpos[f][i] / 2 + 1, cb = cpos[f][i] % 2;
            for (int x = 0; x < 16; x++) { int px = par(w1 & x) | (par(w2 & x) << 1) | (par(w3 & x) << 2); assert(GV[i][x] == (par(ca & px) ^ cb)); }
        }
        f++;
    }
    for (int i = 14; i < 30; i++) assert(FL[i].size() == 7);
}

static inline u32 hmin(__m128i w) { return (u32)_mm_cvtsi128_si32(_mm_minpos_epu16(w)) & 0xFFFF; }
// sum over subsets of blocks of max_x count, 16 points
u64 evalM16(int K, const uint16_t (*vec)[16]) {
    int r = K < 4 ? K : 4, outer = K - r, nin = 1 << r;
    alignas(16) uint16_t tmp[16];
    __m128i TL[16], TH[16], PL[30], PH[30];
    for (int t = 0; t < nin; t++) {
        for (int x = 0; x < 16; x++) { int s = 0; for (int i = 0; i < r; i++) if (t >> i & 1) s += vec[outer + i][x]; tmp[x] = s; }
        TL[t] = _mm_load_si128((__m128i*)tmp); TH[t] = _mm_load_si128((__m128i*)(tmp + 8));
    }
    for (int b = 0; b < outer; b++) { for (int x = 0; x < 16; x++) tmp[x] = vec[b][x]; PL[b] = _mm_load_si128((__m128i*)tmp); PH[b] = _mm_load_si128((__m128i*)(tmp + 8)); }
    const u32 BASE = 1000;
    __m128i wl = _mm_set1_epi16(BASE), wh = _mm_set1_epi16(BASE);
    u64 acc = 0;
    auto inner = [&]() -> u32 { u32 s = 0; for (int t = 0; t < nin; t++) s += hmin(_mm_min_epu16(_mm_sub_epi16(wl, TL[t]), _mm_sub_epi16(wh, TH[t]))); return s; };
    acc += inner();
    u64 lim = 1ull << outer;
    for (u64 s = 1; s < lim; s++) {
        int b = __builtin_ctzll(s); u64 g = s ^ (s >> 1);
        if ((g >> b) & 1) { wl = _mm_sub_epi16(wl, PL[b]); wh = _mm_sub_epi16(wh, PH[b]); }
        else { wl = _mm_add_epi16(wl, PL[b]); wh = _mm_add_epi16(wh, PH[b]); }
        acc += inner();
    }
    return (u64)BASE * (1ull << K) - acc;
}
u64 exactM(const int* asg, int K) {
    uint16_t vec[30][16]; memset(vec, 0, sizeof vec);
    for (int i = 0; i < 30; i++) for (int x = 0; x < 16; x++) vec[asg[i]][x] += GV[i][x];
    return evalM16(K, vec);
}
u32 frameN_full(const int* asg, int f) {  // direct: canonical RGS of frame f, rank, lookup
    int arr[14]; for (int i = 0; i < 30; i++) if (cpos[f][i] >= 0) arr[cpos[f][i]] = asg[i];
    int mp[30]; memset(mp, -1, sizeof mp); int k = 0; u32 idx = 0;
    for (int p = 0; p < 14; p++) { if (mp[arr[p]] < 0) mp[arr[p]] = k++; idx = childStart[p][idx] + mp[arr[p]]; }
    return Ntab[idx];
}
u32 rootN(const int* asg) { u32 idx = 0; int mp[30]; memset(mp, -1, sizeof mp); int k = 0; for (int p = 0; p < 14; p++) { if (mp[asg[p]] < 0) mp[asg[p]] = k++; idx = childStart[p][idx] + mp[asg[p]]; } return Ntab[idx]; }

int L; i64 thr; bool SYM = true; FILE* highF = nullptr; mutex highMu; i64 highThrU = (i64)4771 << 18;   // record leaves with H4 > HIGHF/13440
struct Root { u32 N; vector<int> rgs; vector<array<uint8_t, 16>> hinv; vector<array<uint8_t, 16>> sigma; };
int DSYM = 5;
// lifts of the root pattern's stabilizer in AGL(3,2) x swap to symmetries of the d=4 problem fixing frame 8:
// root (a,b), a in 1..7: (A a, b ^ <a,c> ^ s);  outside a = 8+y: (8 + (A y ^ mu), b ^ <y,c> ^ c3 ^ s)
void buildRootSym(Root& R) {
    int K0 = *max_element(R.rgs.begin(), R.rgs.end()) + 1;
    for (int u1 = 1; u1 < 8; u1++) for (int u2 = 1; u2 < 8; u2++) for (int u3 = 1; u3 < 8; u3++) {
        if (u2 == u1 || u3 == u1 || u3 == u2 || u3 == (u1 ^ u2)) continue;
        int us[3] = {u1, u2, u3};
        auto A = [&](int y) { int r = 0; for (int k = 0; k < 3; k++) if (y >> k & 1) r ^= us[k]; return r; };
        for (int c = 0; c < 8; c++) for (int sw = 0; sw < 2; sw++) {
            int p[14]; for (int i = 0; i < 14; i++) { int a = i / 2 + 1, b = i % 2; p[i] = 2 * (A(a) - 1) + (b ^ par(a & c) ^ sw); }
            int sig[16]; memset(sig, -1, sizeof sig); bool st = true;
            for (int i = 0; i < 14 && st; i++) { int a = R.rgs[i], b = R.rgs[p[i]]; if (sig[a] < 0) sig[a] = b; else if (sig[a] != b) st = false; }
            if (!st) continue;
            for (int mu = 0; mu < 8; mu++) for (int c3 = 0; c3 < 2; c3++) {
                if (u1 == 1 && u2 == 2 && u3 == 4 && c == 0 && sw == 0 && mu == 0 && c3 == 0) continue;
                array<uint8_t, 16> hi, sg;
                for (int j = 14; j < 30; j++) { int y = (j / 2 + 1) - 8, b = j % 2; int a2 = 8 + (A(y) ^ mu); int b2 = b ^ par(y & c) ^ c3 ^ sw; hi[2 * (a2 - 1) + b2 - 14] = j; }
                for (int k = 0; k < 16; k++) sg[k] = k < K0 ? sig[k] : 0;
                R.hinv.push_back(hi); R.sigma.push_back(sg);
            }
        }
    }
}
vector<Root> roots;

struct Best { u64 M = 0; int K = 0; vector<int> asg; bool better(u64 m, int k) const { return (__int128)m << K > (__int128)M << k; } };

struct Worker {
    struct FS { u32 idx; int8_t k, j; int8_t map[30]; } fs[14];
    u32 ub[14];
    int asg[30], K, K0; i64 sum; u32 Nroot;
    struct Und { u32 idx; int8_t k, j, newmap; u32 ub; } und[30][7];
    int oldK[30];
    u64 nodes = 0, leaves = 0, viol = 0;
    Best best;
    vector<vector<int>> violList;
    map<int, u64> leafK;
    vector<pair<double, vector<int>>> highLeaves;

    bool initRoot(const Root& R) {
        curRoot = &R;
        for (int i = 0; i < 14; i++) asg[i] = R.rgs[i];
        K0 = K = *max_element(R.rgs.begin(), R.rgs.end()) + 1;
        Nroot = R.N;
        sum = Nroot;
        for (int f = 0; f < 14; f++) {
            FS& S = fs[f]; memset(S.map, -1, sizeof S.map); S.k = 0; S.idx = 0;
            for (int p = 0; p < 6; p++) { int g = asg[sharedPos[f][p]]; if (S.map[g] < 0) S.map[g] = S.k++; S.idx = childStart[p][S.idx] + S.map[g]; }
            S.j = 6;
            ub[f] = min(PMat(6, S.idx), Nroot); sum += ub[f];
        }
        return sum > thr;
    }
    bool push(int i, int c) {
        nodes++;
        i64 s = sum; bool ok = true; int t = 0;
        for (int f : FL[i]) {
            FS& S = fs[f]; Und& U = und[i][t++];
            U.idx = S.idx; U.k = S.k; U.j = S.j; U.ub = ub[f]; U.newmap = 0;
            int lc = S.map[c]; if (lc < 0) { lc = S.k; S.map[c] = lc; S.k++; U.newmap = 1; }
            S.idx = childStart[S.j][S.idx] + lc; S.j++;
            u32 v = PMat(S.j, S.idx);
            if (S.j == NL && v > Nroot) ok = false;
            if (v > Nroot) v = Nroot;
            s += (i64)v - (i64)ub[f]; ub[f] = v;
        }
        oldK[i] = K; asg[i] = c; if (c == K) K++;
        sum = s;
        return ok && s > thr;
    }
    void pop(int i) {
        int t = 0;
        for (int f : FL[i]) {
            Und& U = und[i][t++]; FS& S = fs[f];
            if (U.newmap) S.map[asg[i]] = -1;
            S.idx = U.idx; S.k = U.k; S.j = U.j;
            sum += (i64)U.ub - (i64)ub[f]; ub[f] = U.ub;
        }
        K = oldK[i];
    }
    vector<i64> envMax = vector<i64>(256 * 200, -1);
    void leaf() {
        leaves++; leafK[K]++;
        u64 M = exactM(asg, K);
        { i64 sl = 82944 - (i64)Nroot; for (int f = 0; f < 14; f++) sl += 82944 - (i64)Ntab[fs[f].idx];
          i64 hU = (i64)M * 7 * ((i64)1 << (24 - K));   // H4 in units 1/(2^25*105)
          if (sl >= 0 && sl < (i64)envMax.size() && hU > envMax[sl]) envMax[sl] = hU; }
        if (best.better(M, K)) { best.M = M; best.K = K; best.asg.assign(asg, asg + 30); }
        double h = (double)M / (30.0 * (double)(1ull << K));
        if ((i64)M * 7 * ((i64)1 << (24 - K)) > highThrU) { lock_guard<mutex> g(highMu); fprintf(highF, "%.9f", h); for (int q = 0; q < 30; q++) fprintf(highF, " %d", asg[q]); fputc('\n', highF); }
        if ((__int128)448 * M >= ((__int128)1 << K) * (4860 - L)) { viol++; if (violList.size() < 100) violList.push_back(vector<int>(asg, asg + 30)); }
    }
    inline int type(int c) const { return c < K0 ? c : K0; }
    const Root* curRoot = nullptr; u64 symPruned = 0;
    bool lexPrune(int i) {
        if (i - 14 > DSYM || !curRoot) return false;
        int nm[32];
        for (size_t e = 0; e < curRoot->hinv.size(); e++) {
            const auto& hi = curRoot->hinv[e]; const auto& sg = curRoot->sigma[e];
            for (int q = 0; q < 32; q++) nm[q] = -1;
            int nn = 0;
            for (int p = 14; p < i; p++) {
                int pre = hi[p - 14]; if (pre >= i) break;
                int blk = asg[pre], v;
                if (blk < K0) v = sg[blk]; else { if (nm[blk] < 0) nm[blk] = K0 + nn++; v = nm[blk]; }
                if (v < asg[p]) { symPruned++; return true; }
                if (v > asg[p]) break;
            }
        }
        return false;
    }
    void dfs(int i) {
        if (i == 30) { leaf(); return; }
        int t14 = i > 14 ? type(asg[14]) : 0;
        for (int c = 0; c <= K; c++) { if (SYM && i > 14 && type(c) < t14) continue; if (push(i, c) && !(SYM && lexPrune(i + 1))) dfs(i + 1); pop(i); }
    }
};

struct Task { int root; vector<int> pre; };

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "usage: bb4 datadir L rootsfile outprefix [D]\n"); return 1; }
    string dir = argv[1]; L = atoi(argv[2]); string rf = argv[3]; string outp = argv[4];
    int D = argc > 5 ? atoi(argv[5]) : 3;
    if (getenv("HIGHF")) highThrU = (i64)atoi(getenv("HIGHF")) << 18;
    highF = fopen((outp + "_high.txt").c_str(), "w");
    long tLo = argc > 7 ? atol(argv[6]) : 0, tHi = argc > 7 ? atol(argv[7]) : LONG_MAX;   // task range for chunked runs
    if (argc > 8) SYM = atoi(argv[8]) != 0;
    thr = (i64)256 * (4860 - L);
    buildLevels(); buildFrames();
    Ntab = (u32*)malloc(4 * C[NL]); PMall = (u32*)malloc(4 * PMoff[NL]);
    { FILE* f = fopen((dir + "/Ntab.bin").c_str(), "rb"); if (fread(Ntab, 4, C[NL], f) != C[NL]) return 2; fclose(f); }
    { FILE* f = fopen((dir + "/PM.bin").c_str(), "rb"); if (fread(PMall, 4, PMoff[NL], f) != PMoff[NL]) return 2; fclose(f); }
    // self-test on the published loss-28 witness: frame values via direct lookup
    {
        int w[30] = {0,1,2,0,2,1,2,3,2,4,2,5,2,5,3,6,4,6,5,6,5,6,2,6,2,6,1,6,0,6};
        printf("selftest witness: root(n=8) N/256=%.2f;", rootN(w) / 256.0);
        for (int f = 0; f < 14; f++) printf(" n%d:%.2f", frameNormal[f], frameN_full(w, f) / 256.0);
        u64 M = exactM(w, 7); printf("\n  exact H4 = %llu/(30*128) = %.7f\n", (unsigned long long)M, M / (30.0 * 128));
    }
    // roots: include every rep whose loss is < L/15 (others are covered by the frame bound)
    {
        FILE* f = fopen(rf.c_str(), "r"); u32 N; int sz;
        while (fscanf(f, "%u %d", &N, &sz) == 2) {
            Root R; R.N = N; R.rgs.resize(14); for (int i = 0; i < 14; i++) if (fscanf(f, "%d", &R.rgs[i]) != 1) return 3;
            if ((i64)15 * (82944 - (i64)N) < (i64)256 * L) { if (getenv("DSYM")) DSYM = atoi(getenv("DSYM")); if (SYM && DSYM > 0) buildRootSym(R); roots.push_back(R); }
        }
        fclose(f);
    }
    // list must contain every local level whose loss < L/15: check against the histogram
    {
        FILE* f = fopen((dir + "/hist.txt").c_str(), "r"); u32 N; unsigned long long cnt; set<u32> need, have;
        while (fscanf(f, "%u %llu", &N, &cnt) == 2) if ((i64)15 * (82944 - (i64)N) < (i64)256 * L) need.insert(N);
        fclose(f);
        for (auto& R : roots) have.insert(R.N);
        if (need != have) { fprintf(stderr, "root list does not cover all needed levels\n"); return 4; }
        printf("L=%d thr=%lld roots=%zu levels:", L, (long long)thr, roots.size()); for (u32 n : need) printf(" %u/896", n / 256); printf("\n");
    }
    // tasks
    vector<Task> tasks;
    {
        Worker W;
        for (int r = 0; r < (int)roots.size(); r++) {
            if (!W.initRoot(roots[r])) continue;
            vector<int> pre;
            function<void(int)> gen = [&](int i) {
                if (i == 14 + D) { tasks.push_back({r, pre}); return; }
                int t14 = i > 14 ? W.type(W.asg[14]) : 0;
                for (int c = 0; c <= W.K; c++) { if (SYM && i > 14 && W.type(c) < t14) continue; if (W.push(i, c) && !(SYM && W.lexPrune(i + 1))) { pre.push_back(c); gen(i + 1); pre.pop_back(); } W.pop(i); }
            };
            gen(14);
        }
    }
    printf("tasks: %zu (symmetry breaking %s)\n", tasks.size(), SYM ? "on" : "off");
    if (tHi > (long)tasks.size()) tHi = tasks.size();
    if (tLo > 0 || tHi < (long)tasks.size()) { tasks = vector<Task>(tasks.begin() + tLo, tasks.begin() + tHi); printf("CHUNK tasks [%ld,%ld)\n", tLo, tHi); }
    fflush(stdout);
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
                bool alive = true; int i = 14;
                for (; i < 14 + (int)T.pre.size(); i++) if (!W.push(i, T.pre[i - 14])) { alive = false; i++; break; }
                if (alive) W.dfs(i);
                for (int j = i - 1; j >= 14; j--) W.pop(j);
                int d = ++done;
                if (d % 200 == 0 || d == (int)tasks.size()) {
                    double s = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
                    fprintf(stderr, "[%.0fs] tasks %d/%zu\n", s, d, tasks.size());
                }
            }
        });
    }
    for (auto& x : th) x.join();
    double secs = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
    u64 nodes = 0, leaves = 0, viol = 0; Best best; map<int, u64> leafK; vector<pair<double, vector<int>>> high; vector<vector<int>> vl;
    for (auto w : ws) {
        nodes += w->nodes; leaves += w->leaves; viol += w->viol;
        if (w->leaves && best.better(w->best.M, w->best.K)) best = w->best;
        for (auto& [k, v] : w->leafK) leafK[k] += v;
        high.insert(high.end(), w->highLeaves.begin(), w->highLeaves.end());
        vl.insert(vl.end(), w->violList.begin(), w->violList.end());
    }
    { u64 sp = 0; for (auto w : ws) sp += w->symPruned; printf("lex-leader prunes: %llu\n", (unsigned long long)sp); }
    printf("done in %.1fs: nodes=%llu surviving leaves=%llu violations(H4>=T_L)=%llu\n", secs, (unsigned long long)nodes, (unsigned long long)leaves, (unsigned long long)viol);
    printf("T_L = (4860-%d)/13440 = %.9f\n", L, (4860.0 - L) / 13440);
    if (leaves) {
        printf("best exact H4 among survivors: M=%llu K=%d  H4=%.9f  asg:", (unsigned long long)best.M, best.K, best.M / (30.0 * (double)(1ull << best.K)));
        for (int x : best.asg) printf(" %d", x); printf("\n");
    }
    printf("leaf block counts:"); for (auto& [k, v] : leafK) printf(" K%d:%llu", k, (unsigned long long)v); printf("\n");
    {   // envelope: max exact H4 at each exact frame-loss value s = 256*R (s < 256*L)
        vector<i64> env(256 * 200, -1);
        for (auto w : ws) for (size_t i = 0; i < env.size(); i++) env[i] = max(env[i], w->envMax[i]);
        FILE* g = fopen((outp + "_env.txt").c_str(), "w");
        for (size_t i = 0; i < env.size(); i++) if (env[i] >= 0) fprintf(g, "%zu %lld\n", i, (long long)env[i]);
        fclose(g);
        const double UH = 3523215360.0;
        printf("max exact H4 (x13440) among partitions with frame loss >= R (R < L):");
        for (int R = 28; R < L; R += 2) { i64 m = -1; for (size_t i = 256 * R; i < (size_t)256 * L; i++) m = max(m, env[i]); printf(" R%d:%.2f", R, m * 13440.0 / UH); }
        printf("\n");
    }
    fclose(highF);
    FILE* f = fopen((outp + "_viol.txt").c_str(), "w"); for (auto& a : vl) { for (int x : a) fprintf(f, "%d ", x); fprintf(f, "\n"); } fclose(f);
    return 0;
}
