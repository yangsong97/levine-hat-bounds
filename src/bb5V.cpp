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

// Same sum as evalM<32>, with one 512-bit register per vector (checked against evalM<32> at startup).
static inline u32 hmin512(__m512i m) {
    __m256i m2 = _mm256_min_epu16(_mm512_castsi512_si256(m), _mm512_extracti64x4_epi64(m, 1));
    return hmin(_mm_min_epu16(_mm256_castsi256_si128(m2), _mm256_extracti128_si256(m2, 1)));
}
u64 evalM32z(int K, const uint16_t (*vec)[32]) {
    int r = K < 4 ? K : 4, outer = K - r, nin = 1 << r;
    __m512i T[16], P[64];
    T[0] = _mm512_setzero_si512();
    for (int t = 1; t < nin; t++) T[t] = _mm512_add_epi16(T[t & (t - 1)], _mm512_loadu_si512((const void*)vec[outer + __builtin_ctz(t)]));
    for (int b = 0; b < outer; b++) P[b] = _mm512_loadu_si512((const void*)vec[b]);
    const u32 BASE = 1000;
    __m512i w = _mm512_set1_epi16(BASE);
    u64 acc = 0;
    auto inner = [&]() -> u32 { u32 s = 0; for (int t = 0; t < nin; t++) s += hmin512(_mm512_sub_epi16(w, T[t])); return s; };
    acc += inner();
    u64 lim = 1ull << outer;
    for (u64 s = 1; s < lim; s++) {
        int b = __builtin_ctzll(s); u64 g = s ^ (s >> 1);
        if ((g >> b) & 1) w = _mm512_sub_epi16(w, P[b]); else w = _mm512_add_epi16(w, P[b]);
        acc += inner();
    }
    return (u64)BASE * (1ull << K) - acc;
}

// evalM32z that also records, for every set of retained blocks, the lowest-index point attaining the maximum.
// arg is indexed by the subset: bit i (i < r) = block outer + i, bit r + b = block b, where r = min(K, 4), outer = K - r.
u64 evalM32arg(int K, const uint16_t (*vec)[32], uint8_t* arg) {
    int r = K < 4 ? K : 4, outer = K - r, nin = 1 << r;
    __m512i T[16], P[64];
    T[0] = _mm512_setzero_si512();
    for (int t = 1; t < nin; t++) T[t] = _mm512_add_epi16(T[t & (t - 1)], _mm512_loadu_si512((const void*)vec[outer + __builtin_ctz(t)]));
    for (int b = 0; b < outer; b++) P[b] = _mm512_loadu_si512((const void*)vec[b]);
    const u32 BASE = 1000;
    __m512i w = _mm512_set1_epi16(BASE);
    u64 acc = 0;
    auto inner = [&](u64 G) {
        u32 s = 0; uint8_t* out = arg + (G << r);
        for (int t = 0; t < nin; t++) {
            __m512i v = _mm512_sub_epi16(w, T[t]); u32 m = hmin512(v); s += m;
            out[t] = (uint8_t)_tzcnt_u32(_mm512_cmpeq_epi16_mask(v, _mm512_set1_epi16((short)m)));
        }
        acc += s;
    };
    inner(0);
    u64 lim = 1ull << outer;
    for (u64 s = 1; s < lim; s++) {
        int b = __builtin_ctzll(s); u64 g = s ^ (s >> 1);
        if ((g >> b) & 1) w = _mm512_sub_epi16(w, P[b]); else w = _mm512_add_epi16(w, P[b]);
        inner(g);
    }
    return (u64)BASE * (1ull << K) - acc;
}
// N[j][x] = number of block subsets containing block j whose recorded maximum point is x (j < K); N[K][x] = number of all subsets with maximum point x.
void buildN32(int K, const uint8_t* arg, u32 (*N)[32]) {
    memset(N, 0, sizeof(u32) * 32 * (K + 1));
    int r = K < 4 ? K : 4, outer = K - r; int blk[32];
    for (int p = 0; p < K; p++) blk[p] = p < r ? outer + p : p - r;
    for (u64 idx = 0; idx < (1ull << K); idx++) {
        int x = arg[idx]; N[K][x]++;
        for (u64 m = idx; m; m &= m - 1) N[blk[__builtin_ctzll(m)]][x]++;
    }
}

// Pair evaluation for one hyperplane and one complementary pair of new labels (a,0), (a,1).
// vec: K local blocks over the 16 hyperplane points, permuted so that points 0..7 are the points covered by (a,0)
// (side 0) and points 8..15 those covered by (a,1) (side 1). For a set U of retained blocks let m0 / m1 be the
// largest count on side 0 / side 1. Returns B = sum_U max(m0, m1) and sets bit U of b0 iff m0 >= m1, of b1 iff m1 >= m0.
// Bit index of U: bit i (i < r) = block outer + i, bit r + b = block b, where r = min(K, 4), outer = K - r.
// If (a,0) joins a block that is retained in U the maximum becomes max(m0 + 1, m1) = max(m0, m1) + [m0 >= m1];
// if (a,1) joins a retained block it gains [m1 >= m0]; if both are retained it gains 1 (not 2 when m0 = m1).
const int PB_MAXK = 14;
u64 pairEval16(int K, const uint16_t (*vec)[16], u64* b0, u64* b1) {
    int r = K < 4 ? K : 4, outer = K - r, nin = 1 << r;
    __m128i T[16][2], P[64][2];
    T[0][0] = T[0][1] = _mm_setzero_si128();
    for (int t = 1; t < nin; t++) { int i = __builtin_ctz(t), u = t & (t - 1);
        T[t][0] = _mm_add_epi16(T[u][0], _mm_loadu_si128((const __m128i*)vec[outer + i]));
        T[t][1] = _mm_add_epi16(T[u][1], _mm_loadu_si128((const __m128i*)(vec[outer + i] + 8))); }
    for (int b = 0; b < outer; b++) { P[b][0] = _mm_loadu_si128((const __m128i*)vec[b]); P[b][1] = _mm_loadu_si128((const __m128i*)(vec[b] + 8)); }
    const u32 BASE = 1000;
    __m128i w0 = _mm_set1_epi16(BASE), w1 = w0;
    size_t nw = K <= 6 ? 1 : (size_t)1 << (K - 6);
    memset(b0, 0, nw * 8); memset(b1, 0, nw * 8);
    uint16_t* q0 = (uint16_t*)b0; uint16_t* q1 = (uint16_t*)b1;   // one 16-bit group per outer subset when r = 4 (little endian)
    u64 acc = 0;
    auto inner = [&](u64 G) {
        u32 s = 0, m0 = 0, m1 = 0;
        for (int t = 0; t < nin; t++) {
            u32 a = hmin(_mm_sub_epi16(w0, T[t][0])), b = hmin(_mm_sub_epi16(w1, T[t][1]));   // BASE - max on side 0 / side 1
            s += a < b ? a : b;
            m0 |= (u32)(a <= b) << t; m1 |= (u32)(b <= a) << t;
        }
        acc += s; q0[G] = (uint16_t)m0; q1[G] = (uint16_t)m1;
    };
    inner(0);
    u64 lim = 1ull << outer;
    for (u64 s = 1; s < lim; s++) {
        int b = __builtin_ctzll(s); u64 g = s ^ (s >> 1);
        if ((g >> b) & 1) { w0 = _mm_sub_epi16(w0, P[b][0]); w1 = _mm_sub_epi16(w1, P[b][1]); }
        else { w0 = _mm_add_epi16(w0, P[b][0]); w1 = _mm_add_epi16(w1, P[b][1]); }
        inner(g);
    }
    return (u64)BASE * (1ull << K) - acc;
}
static const u64 PB_MSK[6] = {0xAAAAAAAAAAAAAAAAull, 0xCCCCCCCCCCCCCCCCull, 0xF0F0F0F0F0F0F0F0ull, 0xFF00FF00FF00FF00ull, 0xFFFF0000FFFF0000ull, 0xFFFFFFFF00000000ull};
static inline int pbBit(int K, int j) { int r = K < 4 ? K : 4, outer = K - r; return j < outer ? r + j : j - outer; }   // index bit of block j
// D[j] = number of set bits whose subset contains block j (j < K); D[K] = number of set bits. With bits2, counts bits & bits2.
void pbCount(int K, const u64* bits, const u64* bits2, u32* D) {
    size_t nw = K <= 6 ? 1 : (size_t)1 << (K - 6);
    u64 tot = 0; u64 lowc[6] = {0, 0, 0, 0, 0, 0}; u32 hic[16]; memset(hic, 0, sizeof hic);
    for (size_t w = 0; w < nw; w++) {
        u64 v = bits2 ? (bits[w] & bits2[w]) : bits[w]; u32 pc = __builtin_popcountll(v); tot += pc;
        for (int p = 0; p < 6; p++) lowc[p] += __builtin_popcountll(v & PB_MSK[p]);
        for (size_t m = w; m; m &= m - 1) hic[__builtin_ctzll(m)] += pc;
    }
    D[K] = (u32)tot;
    for (int j = 0; j < K; j++) { int p = pbBit(K, j); D[j] = p < 6 ? (u32)lowc[p] : hic[p - 6]; }
}
// number of subsets containing both blocks j1 and j2 whose bit is set in bits & bits2
u32 pbCount2(int K, const u64* bits, const u64* bits2, int j1, int j2) {
    size_t nw = K <= 6 ? 1 : (size_t)1 << (K - 6); int p1 = pbBit(K, j1), p2 = pbBit(K, j2); u64 c = 0;
    u64 lowm = ~0ull; size_t him = 0;
    if (p1 < 6) lowm &= PB_MSK[p1]; else him |= (size_t)1 << (p1 - 6);
    if (p2 < 6) lowm &= PB_MSK[p2]; else him |= (size_t)1 << (p2 - 6);
    for (size_t w = 0; w < nw; w++) if ((w & him) == him) c += __builtin_popcountll(bits[w] & bits2[w] & lowm);
    return (u32)c;
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
uint16_t HVP[30][8][30][16];     // the same, points permuted for outside pair q of the hyperplane: side a.x = 1 first, then a.x = 0
int rootLab[30]; uint16_t RV[30][16];
u32 hypOfLab[NLAB];   // bitmask of the non-root hyperplanes containing each label
typedef unsigned __int128 u128;
vector<vector<u128>> lists; vector<i64> capVal; int nlev = 0;

int g4[30][14];          // hyperplane h: global label at d=4 position p (0..13: the root frame T∩T0)
int dpos[30][NLAB];      // hyperplane h: d=4 position of global label (or -1)
u128 maskFrom[31];       // nibbles p..29 (and the 8 pad bits) set
vector<u128> allKeys; vector<uint8_t> keyLev; vector<uint8_t> blkMin; vector<vector<uint8_t>> sparse;
const int BS = 64;
inline int rmq(size_t a, size_t b) {   // min level over [a,b)
    int m = 255;
    size_t ba = (a + BS - 1) / BS, bb = b / BS;
    if (ba >= bb) { for (size_t i = a; i < b; i++) m = min<int>(m, keyLev[i]); return m; }
    for (size_t i = a; i < ba * BS; i++) m = min<int>(m, keyLev[i]);
    for (size_t i = bb * BS; i < b; i++) m = min<int>(m, keyLev[i]);
    size_t len = bb - ba; int k = 63 - __builtin_clzll(len);
    m = min<int>(m, min(sparse[k][ba], sparse[k][bb - (1ull << k)]));
    return m;
}
inline int bestLevel(u128 key, int len) {
    u128 hi = key | maskFrom[len];
    size_t a = lower_bound(allKeys.begin(), allKeys.end(), key) - allKeys.begin();
    size_t b = upper_bound(allKeys.begin() + a, allKeys.end(), hi) - allKeys.begin();
    if (a >= b) return nlev;
    return rmq(a, b);
}


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
    for (int h = 0; h < 30; h++) {
        int n = hypNormal[h];
        vector<int> out; for (int a = 16; a < 32; a++) if (!par(a & n)) out.push_back(a);
        sort(out.begin(), out.end()); assert(out.size() == 8);
        int d2 = out[0] ^ out[1], d3 = out[0] ^ out[2], d5 = out[0] ^ out[4];
        for (int r = 0; r < 8; r++) assert(out[r] == (out[0] ^ ((r & 1) ? d2 : 0) ^ ((r & 2) ? d3 : 0) ^ ((r & 4) ? d5 : 0)));
        for (int i = 0; i < NLAB; i++) dpos[h][i] = -1;
        for (int a = 1; a < 32; a++) {
            if (par(a & n)) continue;
            int base = a & 16 ? a ^ out[0] : a, phi = -1;
            for (int c = 0; c < 8; c++) if ((((c & 1) ? d2 : 0) ^ ((c & 2) ? d3 : 0) ^ ((c & 4) ? d5 : 0)) == base) phi = c;
            assert(phi >= 0); if (a & 16) phi += 8;
            assert(phi > 0);
            for (int b = 0; b < 2; b++) { int i = 2 * (a - 1) + b, p = 2 * (phi - 1) + b; dpos[h][i] = p; if (p < 14) g4[h][p] = i; }
        }
        int expect = 14; for (int i = 30; i < NLAB; i++) if (dpos[h][i] >= 0) { assert(dpos[h][i] == expect); expect++; }
        assert(expect == 30);
    }
    for (int h = 0; h < 30; h++) {   // permuted cover vectors for the batch evaluation
        for (int i = 30; i < NLAB; i += 2) {
            if (dpos[h][i] < 0) continue;
            int q = (dpos[h][i] - 14) / 2; assert(q >= 0 && q < 8 && dpos[h][i] == 14 + 2 * q && dpos[h][i + 1] == 15 + 2 * q);
            int ti = -1; for (int t = 0; t < 30; t++) if (hypLab[h][t] == i) ti = t;
            assert(ti >= 0 && hypLab[h][ti + 1] == i + 1);
            int perm[16], n1 = 0, n0 = 8;
            for (int p = 0; p < 16; p++) { if (HV[h][ti][p]) perm[n1++] = p; else perm[n0++] = p; assert(HV[h][ti + 1][p] == 1 - HV[h][ti][p]); }
            assert(n1 == 8 && n0 == 16);
            for (int t = 0; t < 30; t++) for (int p = 0; p < 16; p++) HVP[h][q][t][p] = HV[h][t][perm[p]];
        }
    }
    for (int p = 0; p <= 30; p++) { u128 m = 0xFF; for (int q = p; q < 30; q++) m |= (u128)15 << (4 * (29 - q) + 8); maskFrom[p] = m; }
    for (int i = 30; i < NLAB; i++) assert(FofL[i].size() == 35 && HofL[i].size() == 15);
    for (int i = 0; i < NLAB; i++) { hypOfLab[i] = 0; for (int h = 0; h < 30; h++) if (dpos[h][i] >= 0) hypOfLab[i] |= 1u << h; }
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
// envelope g*(s): max exact H4 (units U) over d=4 partitions with exact frame loss*256 >= s, valid for s < gsz
vector<i64> gstar; i64 gsz = 0;
// level lists: lists[k] = sorted keys of all labelled d=4 partitions with H4 >= levelVal[k]; capVal[nlev] = floor
inline bool consistent(const vector<u128>& L, u128 key, int len) {
    u128 hi = key | maskFrom[len];
    auto it = lower_bound(L.begin(), L.end(), key);
    return it != L.end() && *it <= hi;
}
struct Root { string val; vector<int> rgs; vector<array<uint8_t, 32>> hinv; vector<array<uint8_t, 32>> sigma; };
int DSYM = 5;
void buildRootSym(Root& R) {
    int K0 = *max_element(R.rgs.begin(), R.rgs.end()) + 1;
    for (int u1 = 1; u1 < 16; u1++) for (int u2 = 1; u2 < 16; u2++) for (int u3 = 1; u3 < 16; u3++) for (int u4 = 1; u4 < 16; u4++) {
        int us[4] = {u1, u2, u3, u4}; int span = 1; bool ok = true;   // bitmask of span
        for (int u : us) { if (span >> u & 1) { ok = false; break; } int s2 = span; for (int v = 0; v < 16; v++) if (span >> v & 1) s2 |= 1 << (v ^ u); span = s2; }
        if (!ok) continue;
        auto A = [&](int y) { int r = 0; for (int k = 0; k < 4; k++) if (y >> k & 1) r ^= us[k]; return r; };
        for (int c = 0; c < 16; c++) for (int sw = 0; sw < 2; sw++) {
            int p[30]; for (int i = 0; i < 30; i++) { int a = i / 2 + 1, b = i % 2; p[i] = 2 * (A(a) - 1) + (b ^ par(a & c) ^ sw); }
            // is p a stabilizer of R (as a partition)?  sigma: block R[i] -> block R[p[i]]
            int sig[32]; memset(sig, -1, sizeof sig); bool st = true;
            for (int i = 0; i < 30 && st; i++) { int a = R.rgs[i], b = R.rgs[p[i]]; if (sig[a] < 0) sig[a] = b; else if (sig[a] != b) st = false; }
            if (!st) continue;
            { bool inj = true; int seen = 0; for (int k = 0; k < K0; k++) { if (seen >> sig[k] & 1) inj = false; seen |= 1 << sig[k]; } if (!inj) continue; }
            for (int mu = 0; mu < 16; mu++) for (int c4 = 0; c4 < 2; c4++) {
                bool ident = (u1 == 1 && u2 == 2 && u3 == 4 && u4 == 8 && c == 0 && sw == 0 && mu == 0 && c4 == 0);
                if (ident) continue;
                array<uint8_t, 32> hi; array<uint8_t, 32> sg;
                for (int j = 30; j < 62; j++) { int y = (j / 2 + 1) - 16, b = j % 2; int a2 = 16 + (A(y) ^ mu); int b2 = b ^ par(y & c) ^ c4 ^ sw; int img = 2 * (a2 - 1) + b2; hi[img - 30] = j; }
                for (int k = 0; k < 32; k++) sg[k] = k < K0 ? sig[k] : 0;
                R.hinv.push_back(hi); R.sigma.push_back(sg);
                if (getenv("SYMTEST") && (R.hinv.size() % 97 == 1)) {   // check: H5 and root partition invariant
                    static mt19937 rng(11);
                    int P[62], Q[62]; for (int i = 0; i < 30; i++) P[i] = R.rgs[i];
                    for (int j = 30; j < 62; j++) P[j] = rng() % (K0 + 3);
                    int perm[62]; for (int i = 0; i < 30; i++) perm[i] = p[i]; for (int q = 0; q < 32; q++) perm[hi[q]] = q + 30;
                    for (int i = 0; i < 62; i++) Q[perm[i]] = P[i];
                    auto h5 = [&](int* X) { uint16_t vec[64][32]; memset(vec, 0, sizeof vec); int K = 0; for (int i = 0; i < 62; i++) { K = max(K, X[i] + 1); for (int x = 0; x < 32; x++) vec[X[i]][x] += GV5[i][x]; } return (double)evalM<32>(K, vec) / (double)(1ull << K); };
                    bool rootSame = true; { int m1[64], m2[64]; memset(m1, -1, sizeof m1); memset(m2, -1, sizeof m2); for (int i = 0; i < 30; i++) { if (m1[P[i]] < 0) m1[P[i]] = Q[i]; else if (m1[P[i]] != Q[i]) rootSame = false; if (m2[Q[i]] < 0) m2[Q[i]] = P[i]; else if (m2[Q[i]] != P[i]) rootSame = false; } }
                    static long tests = 0, fails = 0; tests++; if (fabs(h5(P) - h5(Q)) > 1e-9 || !rootSame) fails++;
                    if (tests % 50 == 0) fprintf(stderr, "symtest %ld fails %ld\n", tests, fails);
                }
            }
        }
    }
}
vector<Root> roots;

struct Worker {
    int asg[NLAB], K, K0;
    struct FS { u32 idx; int8_t k, j; int8_t map[64]; } fr[140];
    u32 fub[140];
    i64 hFS[30], hEx[30], hUB[30], S, h0U; int hcnt[30];
    i64 hPE[30], G = 0, Seff = 0, grpV = 0; u32 grpMask = 0;   // partial-exact caps; coupling gain of the complete hyperplanes; best bound at this node
    bool usePE = true, useGrp = true, useDir = true, forceAll = false;
    u64 peEvals = 0, grpEvals = 0, dirEvals = 0, pePrune = 0, grpPrune = 0, dirPrune = 0;
    u32 rootFN[15];
    struct Und { u32 idx; int8_t k, j, newmap; u32 fub; } und[NLAB][35];
    struct HS { i64 hFS[30], hEx[30], hUB[30], hPE[30], S, G, grpV; u32 grpMask; int hcnt[30]; int K; } hsave[NLAB];
    u64 nodes = 0, leaves = 0, viol = 0, hypExact = 0;
    i128 bestNum = 0; int bestK = 0; vector<int> bestAsg;   // best H5 = bestNum / (62 * 2^bestK)
    u64 consistencyFail = 0;
    int8_t hmap[30][64]; int hk[30], hpos[30], lvl[30]; u128 hkey[30];
    struct HU { int8_t newmap; int hk, hpos, lvl; u128 key; } hund[NLAB][15];
    u64 levelDrops = 0; vector<u64> rootNodes = vector<u64>(512, 0); int curRootIdx = 0;
    u64 dPush[NLAB] = {0}, dAlive[NLAB] = {0}, offHist[64] = {0}, onCnt = 0, offCnt = 0, dOff[NLAB][32];
    FILE* leafOut = nullptr;
    void initLevels() {
        for (int h = 0; h < 30; h++) {
            memset(hmap[h], -1, 64); hk[h] = 0; u128 key = 0;
            for (int p = 0; p < 14; p++) { int g = asg[g4[h][p]]; if (hmap[h][g] < 0) hmap[h][g] = hk[h]++; key |= (u128)min<int>(hmap[h][g], 15) << (4 * (29 - p) + 8); }
            hkey[h] = key; hpos[h] = 14; lvl[h] = 0;
            lvl[h] = hk[h] > 16 ? nlev : bestLevel(hkey[h], 14);
        }
    }

    void recomputeS() {
        i64 s = h0U;
        for (int h = 0; h < 30; h++) {
            i64 ub = min(h0U, hFS[h] * 1024);
            i64 sl = 1244160 - hFS[h];
            if (sl < gsz) ub = min(ub, gstar[sl]);
            if (nlev) ub = min(ub, capVal[lvl[h]]);
            ub = min(ub, hPE[h]);
            if (hEx[h] >= 0) ub = min(ub, hEx[h]);
            hUB[h] = ub; s += ub;
        }
        S = s;
    }
    // Partial-exact cap: 30 H4(P|T) <= E max_x #(retained placed labels of T covering x) + 3/4 per unplaced
    // complementary pair + 1/2 per unplaced single label (for any completion). Labels <= i are placed.
    i64 peCap(int h, int i) {
        peEvals++;
        int mp[64]; memset(mp, -1, sizeof mp); int Kh = 0, np = 0; uint16_t vec[30][16];
        for (int t = 0; t < 30; t++) { int lab = hypLab[h][t]; if (lab > i) break;
            int g = asg[lab]; if (mp[g] < 0) { mp[g] = Kh; memset(vec[Kh], 0, sizeof vec[Kh]); Kh++; }
            for (int x = 0; x < 16; x++) vec[mp[g]][x] += HV[h][t][x]; np++; }
        if (Kh > 20) return INT64_MAX / 64;
        u64 M = evalM<16>(Kh, vec);
        int un = 30 - np, p1 = (i >= 30 && i % 2 == 0 && dpos[h][i] >= 0) ? 1 : 0, p2 = (un - p1) / 2;
        return (i64)M * 7 * ((i64)1 << (24 - Kh)) + (i64)p2 * 88080384LL + (i64)p1 * 58720256LL;
    }
    // Coupled bound for a set A of non-root hyperplanes plus the root. Since sum_T c_T = 15 count,
    //   sum_{T in A+root} 30 H4(P|T)  may be replaced by  E max_x sum_{T in A+root} c_T(x)
    //   <= E max_x sum_{T in A+root} (placed part of c_T)(x) + sum_{T in A} (3/4 per unplaced pair + 1/2 per unplaced single).
    // Returned in units U (same units as the sum of the caps); valid for every completion. -1 if skipped.
    i64 coupledA(u32 A, int i, uint8_t* arg = nullptr, int8_t* mpOut = nullptr, int* KcOut = nullptr) {
        int mp[64]; memset(mp, -1, sizeof mp); int Kc = 0; alignas(64) static thread_local uint16_t vec[64][32];
        for (int lab = 0; lab <= i; lab++) {
            int w = (lab < 30 ? 1 : 0) + __builtin_popcount(A & hypOfLab[lab]);
            if (!w) continue;
            int g = asg[lab]; if (mp[g] < 0) { if (Kc == maxKc) return -1; mp[g] = Kc; memset(vec[Kc], 0, sizeof vec[Kc]); Kc++; }
            uint16_t* d = vec[mp[g]]; const uint16_t* c = GV5[lab];
            for (int x = 0; x < 32; x++) d[x] += w * c[x];
        }
        grpEvals++;
        u64 M = arg ? evalM32arg(Kc, vec, arg) : evalM32z(Kc, vec);
        if (mpOut) { for (int g = 0; g < 64; g++) mpOut[g] = (int8_t)mp[g]; *KcOut = Kc; }
        i64 v = (i64)M * 7 * ((i64)1 << (24 - Kc));
        for (int h = 0; h < 30; h++) if (A >> h & 1) {
            int un = 16 - hcnt[h], p1 = (i >= 30 && i % 2 == 0 && dpos[h][i] >= 0) ? 1 : 0, p2 = (un - p1) / 2;
            v += (i64)p2 * 88080384LL + (i64)p1 * 58720256LL;
        }
        return v;
    }
    // Lower bound on v(A + h) - v(A) from A's recorded maximum points: the sum over A + h at the point where A's own sum
    // is largest is at most the maximum. Exact integers in units U; a candidate with bound >= its cap cannot lower the total.
    bool useFilter = true; u64 filtered = 0, nBuilds = 0;
    uint8_t argBuf[2][1 << 14]; int8_t mpBuf[2][64]; int kcBuf[2]; u32 Ntab32[15][32]; u32 cov5[NLAB];
    i64 lbMarg(int h, int i, const int8_t* mpA, int Kc) {
        __m512i a1lo = _mm512_setzero_si512(), a1hi = a1lo, a2lo = a1lo, a2hi = a1lo;
        for (int t = 0; t < 30; t++) { int lab = hypLab[h][t]; if (lab > i) break;
            int g = mpA[asg[lab]]; u32 m = cov5[lab];
            if (g >= 0) { a1lo = _mm512_mask_add_epi32(a1lo, (__mmask16)m, a1lo, _mm512_loadu_si512((const void*)Ntab32[g])); a1hi = _mm512_mask_add_epi32(a1hi, (__mmask16)(m >> 16), a1hi, _mm512_loadu_si512((const void*)(Ntab32[g] + 16))); }
            else { a2lo = _mm512_mask_add_epi32(a2lo, (__mmask16)m, a2lo, _mm512_loadu_si512((const void*)Ntab32[Kc])); a2hi = _mm512_mask_add_epi32(a2hi, (__mmask16)(m >> 16), a2hi, _mm512_loadu_si512((const void*)(Ntab32[Kc] + 16))); }
        }
        i64 a1 = (i64)(u32)_mm512_reduce_add_epi32(_mm512_add_epi32(a1lo, a1hi)), a2 = (i64)(u32)_mm512_reduce_add_epi32(_mm512_add_epi32(a2lo, a2hi));
        int un = 16 - hcnt[h], p1 = (i >= 30 && i % 2 == 0 && dpos[h][i] >= 0) ? 1 : 0, p2 = (un - p1) / 2;
        return a1 * 7 * ((i64)1 << (24 - Kc)) + a2 * 7 * ((i64)1 << (23 - Kc)) + (i64)p2 * 88080384LL + (i64)p1 * 58720256LL;
    }
    inline i64 grpBound() const { if (!grpMask) return S; i64 s = grpV; for (int h = 0; h < 30; h++) if (!(grpMask >> h & 1)) s += hUB[h]; return s; }
    // Direct bound: 62 H5 <= E max_x #(retained placed labels covering x) + 3/4 per unplaced pair + 1/2 per single,
    // returned in the units of S (sum of the 31 hyperplane values). Returns INT64_MAX if skipped.
    i64 directS(int i) {
        if (K > 16) return INT64_MAX;
        dirEvals++;
        static thread_local uint16_t vec[64][32]; memset(vec, 0, sizeof(vec[0]) * K);
        for (int lab = 0; lab <= i; lab++) for (int x = 0; x < 32; x++) vec[asg[lab]][x] += GV5[lab][x];
        u64 M = evalM32z(K, vec);
        int p1 = (i % 2 == 0) ? 1 : 0, p2 = (61 - i - p1) / 2;
        return (i64)M * 105 * ((i64)1 << (24 - K)) + (i64)p2 * 1321205760LL + (i64)p1 * 880803840LL;
    }
    // Pair tables. For an even label e = (a,0) and a hyperplane h containing a, one evaluation at the node where the
    // labels < e are placed gives the partial-exact cap of h in every child (label e placed) and every grandchild
    // (labels e and e+1 placed): the values peCap(h, e) and peCap(h, e+1) would return there. owner identifies that node.
    struct PairEntry {
        u64 owner; bool tried, valid, tieOk; int Kh; int8_t mp[64];
        u64 B; u32 D0[PB_MAXK + 1], D1[PB_MAXK + 1], TC[PB_MAXK + 1]; u32 Tie[PB_MAXK][PB_MAXK];
        i64 tail1, tail2;
        u64 b0[1 << (PB_MAXK - 6)], b1[1 << (PB_MAXK - 6)];
    };
    PairEntry* pb = nullptr;   // [16 pairs][15 hyperplanes]
    u64 ser[NLAB], serCtr = 0;  // serial number of the current node at each depth (depth d = labels <= d placed)
    inline PairEntry& pbAt(int e, int t) { return pb[((e - 30) >> 1) * 15 + t]; }
    bool useBatch = true, batchChk = false; u64 batchEvals = 0, prePrune = 0, batchMismatch = 0, tieBuilds = 0, tableHits = 0;
    u64 tsc[8] = {0};
    // entry for label i (either label of the pair) and hyperplane t of that label, reset if it belongs to another node
    inline PairEntry& pbOwn(int i, int t) {
        int e = i & ~1; PairEntry& E = pbAt(e, t); u64 own = ser[e - 1];
        if (E.owner != own) { E.owner = own; E.tried = false; E.valid = false; E.tieOk = false; }
        return E;
    }
    void fillPair(int e, int t) {   // reads only the labels < e
        PairEntry& E = pbAt(e, t); E.tried = true; E.valid = false; E.tieOk = false;
        int h = HofL[e][t]; int q = (dpos[h][e] - 14) / 2;
        int mp[64]; memset(mp, -1, sizeof mp); int Kh = 0, np = 0; uint16_t vec[30][16];
        for (int u = 0; u < 30; u++) { int lab = hypLab[h][u]; if (lab >= e) break;
            int g = asg[lab]; if (mp[g] < 0) { if (Kh == PB_MAXK) return; mp[g] = Kh; memset(vec[Kh], 0, sizeof vec[Kh]); Kh++; }
            for (int x = 0; x < 16; x++) vec[mp[g]][x] += HVP[h][q][u][x]; np++; }
        E.B = pairEval16(Kh, vec, E.b0, E.b1); batchEvals++;
        pbCount(Kh, E.b0, nullptr, E.D0); pbCount(Kh, E.b1, nullptr, E.D1); pbCount(Kh, E.b0, E.b1, E.TC);
        E.Kh = Kh; for (int g = 0; g < 64; g++) E.mp[g] = (int8_t)mp[g];
        int un1 = 30 - (np + 1), un2 = 30 - (np + 2);
        E.tail1 = (i64)((un1 - 1) / 2) * 88080384LL + 58720256LL;   // after (a,0): its partner is single, the rest are pairs
        E.tail2 = (i64)(un2 / 2) * 88080384LL;                       // after both labels
        E.valid = true;
    }
    // cap of the hyperplane in the child that puts label e into block c
    inline i64 pe1(const PairEntry& E, int c) const {
        int g = E.mp[c], K = E.Kh;
        return g >= 0 ? (i64)(E.B + E.D0[g]) * 7 * ((i64)1 << (24 - K)) + E.tail1
                      : (i64)(2 * E.B + E.D0[K]) * 7 * ((i64)1 << (23 - K)) + E.tail1;
    }
    // cap in the grandchild that puts label e into block c and label e+1 into block c2
    i64 pe2(PairEntry& E, int c, int c2) {
        int g = E.mp[c], g2 = E.mp[c2], K = E.Kh; u64 M; int Kk;
        if (g >= 0 && g2 >= 0) {
            if (!E.tieOk) { for (int a = 0; a < K; a++) { E.Tie[a][a] = E.TC[a]; for (int b = a + 1; b < K; b++) E.Tie[a][b] = E.Tie[b][a] = pbCount2(K, E.b0, E.b1, a, b); } E.tieOk = true; tieBuilds++; }
            M = E.B + E.D0[g] + E.D1[g2] - E.Tie[g][g2]; Kk = K;
        }
        else if (g >= 0) { M = 2 * E.B + 2 * (u64)E.D0[g] + E.D1[K] - E.TC[g]; Kk = K + 1; }           // (a,1) in a block with no label here
        else if (c2 == c) { M = 2 * E.B + ((u64)1 << K); Kk = K + 1; }                                 // both in the same new block
        else if (g2 >= 0) { M = 2 * E.B + E.D0[K] + 2 * (u64)E.D1[g2] - E.TC[g2]; Kk = K + 1; }       // (a,0) in a block with no label here
        else { M = 4 * E.B + 2 * (u64)E.D0[K] + 2 * (u64)E.D1[K] - E.TC[K]; Kk = K + 2; }             // two different new blocks
        return (i64)M * 7 * ((i64)1 << (24 - Kk)) + E.tail2;
    }
    // Self-test of the pair tables: pe1 / pe2 against direct evaluation (evalM<16>) of every placement of the two labels.
    long pairSelfTest(long& ncmp) {
        mt19937 rng(20261110); long bad = 0; ncmp = 0;
        PairEntry* Ep = (PairEntry*)malloc(sizeof(PairEntry)); PairEntry& E = *Ep;
        for (int it = 0; it < 600; it++) {
            int K = 1 + rng() % 10; uint16_t vec[16][16]; memset(vec, 0, sizeof vec);
            int hi = 1 + rng() % 4;
            for (int k = 0; k < K; k++) for (int x = 0; x < 16; x++) vec[k][x] = rng() % (hi + 1);
            E.B = pairEval16(K, vec, E.b0, E.b1);
            pbCount(K, E.b0, nullptr, E.D0); pbCount(K, E.b1, nullptr, E.D1); pbCount(K, E.b0, E.b1, E.TC);
            E.Kh = K; E.tieOk = false; E.tail1 = E.tail2 = 0; for (int g = 0; g < 64; g++) E.mp[g] = g < K ? (int8_t)g : (int8_t)-1;
            // reference: blocks 0..K-1 exist here, K and K+1 are blocks with no label of this hyperplane
            auto ref = [&](int g, int g2) -> i64 {
                uint16_t v[18][16]; memset(v, 0, sizeof v); memcpy(v, vec, sizeof(uint16_t) * 16 * K);
                int used = K;
                for (int x = 0; x < 8; x++) v[g][x] += 1;
                used = max(used, g + 1);
                if (g2 >= 0) { for (int x = 8; x < 16; x++) v[g2][x] += 1; used = max(used, g2 + 1); }
                return (i64)evalM<16>(used, v) * 7 * ((i64)1 << (24 - used));
            };
            for (int g = 0; g <= K; g++) {
                ncmp++; if (pe1(E, g) != ref(g, -1)) bad++;
                for (int g2 = 0; g2 <= K + (g == K ? 1 : 0); g2++) { ncmp++; if (pe2(E, g, g2) != ref(g, g2)) bad++; }
            }
        }
        free(Ep);
        return bad;
    }
    bool initRoot(const Root& R) {
        curRoot = &R;
        if (!pb) { pb = (PairEntry*)malloc(sizeof(PairEntry) * 16 * 15); for (int q = 0; q < 16 * 15; q++) { pb[q].owner = 0; pb[q].tried = pb[q].valid = pb[q].tieOk = false; } }
        ser[29] = ++serCtr;
        for (int l = 0; l < NLAB; l++) { u32 m = 0; for (int x = 0; x < 32; x++) if (GV5[l][x]) m |= 1u << x; cov5[l] = m; }
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
            hcnt[h] = 0; hEx[h] = -1; hPE[h] = INT64_MAX / 64;
        }
        G = 0; grpMask = 0; grpV = 0;
        initLevels();
        recomputeS();
        if (usePE && nlev) { for (int h = 0; h < 30; h++) if (lvl[h] == nlev) hPE[h] = min(hPE[h], peCap(h, 29)); recomputeS(); }
        Seff = S;
        return S > thr31;
    }
    int dirFrom = 53; int dumpLab = -1, dumpLeft = 0; u64 dumpEvery = 1; int greedyUn = 16, greedyMode = 0, maxKc = 14; u64 greedyAdds = 0; int chkLeft = 0; u64 chkEvery = 1;
    inline int type(int c) const { return c < K0 ? c : K0; }
    const Root* curRoot = nullptr; u64 symPruned = 0;
    // lex-leader test: positions 30..i-1 assigned; prune if some symmetry maps P to a smaller RGS
    bool lexPrune(int i) {
        if (i - 30 > DSYM || !curRoot) return false;
        int nm[64]; 
        for (size_t e = 0; e < curRoot->hinv.size(); e++) {
            const auto& hi = curRoot->hinv[e]; const auto& sg = curRoot->sigma[e];
            for (int q = 0; q < 64; q++) nm[q] = -1;
            int nn = 0;
            for (int p = 30; p < i; p++) {
                int pre = hi[p - 30]; if (pre >= i) break;
                int blk = asg[pre], v;
                if (blk < K0) v = sg[blk]; else { if (nm[blk] < 0) nm[blk] = K0 + nn++; v = nm[blk]; }
                if (v < asg[p]) { symPruned++; return true; }
                if (v > asg[p]) break;
            }
        }
        return false;
    }
    bool push(int i, int c) {
        nodes++; dPush[i]++; ser[i] = ++serCtr;
        u64 t_0 = __rdtsc();
        HS& H = hsave[i]; memcpy(H.hFS, hFS, sizeof hFS); memcpy(H.hEx, hEx, sizeof hEx); memcpy(H.hUB, hUB, sizeof hUB); memcpy(H.hPE, hPE, sizeof hPE); memcpy(H.hcnt, hcnt, sizeof hcnt); H.S = S; H.G = G; H.grpV = grpV; H.grpMask = grpMask; H.K = K;
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
        { int t2 = 0;
          for (int h : HofL[i]) {
            HU& V = hund[i][t2++]; V.hk = hk[h]; V.hpos = hpos[h]; V.lvl = lvl[h]; V.key = hkey[h]; V.newmap = 0;
            int p = hpos[h];
            int id = hmap[h][c]; if (id < 0) { id = hk[h]++; hmap[h][c] = id; V.newmap = 1; }
            if (id > 15) lvl[h] = nlev; else hkey[h] |= (u128)id << (4 * (29 - p) + 8);
            hpos[h] = p + 1;
            if (lvl[h] < nlev) { int nl = bestLevel(hkey[h], hpos[h]); if (nl > lvl[h]) levelDrops++; lvl[h] = nl; }
          } }
        bool ok = true, completed = false;
        for (int h : HofL[i]) {
            if (++hcnt[h] == 16) {
                completed = true;
                hypExact++;
                hEx[h] = exactH4U(asg, hypLab[h], HV[h]);
                if (lvl[h] == nlev) { offCnt++; double v = hEx[h] / 262144.0; int b = (int)floor((4764.0 - v) / 4.0); offHist[max(0, min(63, b))]++; } else onCnt++;
                if (hEx[h] > h0U) ok = false;   // P's best hyperplane is not the root: covered elsewhere
            }
        }
        recomputeS();
        bool res = ok && S > thr31;
        u64 t_1 = __rdtsc(); tsc[0] += t_1 - t_0;
        if ((res || forceAll) && usePE && nlev) {   // partial-exact caps for the off-list hyperplanes touched by this label
            bool ch = false;
            for (int t3 = 0; t3 < 15; t3++) { int h = HofL[i][t3]; if (lvl[h] == nlev && hcnt[h] < 16) {
                i64 pe; bool got = false;
                if (useBatch) {
                    PairEntry& E = pbOwn(i, t3);
                    if (!E.tried) fillPair(i & ~1, t3);
                    if (E.valid) { pe = (i & 1) ? pe2(E, asg[i - 1], c) : pe1(E, c); got = true; tableHits++; if (batchChk && pe != peCap(h, i)) batchMismatch++; }
                }
                if (!got) pe = peCap(h, i);
                if (pe < hPE[h]) { hPE[h] = pe; ch = true; } } }
            if (ch) { recomputeS(); if (res && !(S > thr31)) { res = false; pePrune++; } }
        }
        Seff = min(S, grpBound());
        if (res && !(Seff > thr31)) { res = false; grpPrune++; }
        u64 t_2 = __rdtsc(); tsc[1] += t_2 - t_1;
        if ((res || forceAll) && useGrp && ok) {
            // Couple the root with a set A of hyperplanes: the complete ones, those whose cap is the partial-exact one,
            // the set inherited from the parent node, and greedily any further hyperplane that lowers the bound.
            u32 A = grpMask;
            for (int h = 0; h < 30; h++) if (hcnt[h] == 16 || (lvl[h] == nlev && hPE[h] <= hUB[h])) A |= 1u << h;
            if (A) {
                int cur = 0; bool nOk = false;
                i64 v = useFilter ? coupledA(A, i, argBuf[0], mpBuf[0], &kcBuf[0]) : coupledA(A, i);
                if (v >= 0) {
                    i64 rest = 0; for (int h = 0; h < 30; h++) if (!(A >> h & 1)) rest += hUB[h];
                    i64 sBest = v + rest; u32 Abest = A; i64 vBest = v;
                    if (greedyUn > 0 && (sBest > thr31 || forceAll)) {
                        for (int pass = 0; pass < 2 && (sBest > thr31 || forceAll); pass++) {
                            const int* cand; int nc; int allh[30];
                            if (pass == 0) { cand = HofL[i].data(); nc = (int)HofL[i].size(); }
                            else { if (greedyMode == 1 && i > 30 + 3) break; for (int h = 0; h < 30; h++) allh[h] = h; cand = allh; nc = 30; }
                            for (int q = 0; q < nc && (sBest > thr31 || forceAll); q++) {
                                int h = cand[q]; if ((Abest >> h & 1) || 16 - hcnt[h] > greedyUn) continue;
                                if (useFilter) {
                                    if (!nOk) { buildN32(kcBuf[cur], argBuf[cur], Ntab32); nOk = true; nBuilds++; }
                                    if (lbMarg(h, i, mpBuf[cur], kcBuf[cur]) >= hUB[h]) { filtered++; continue; }
                                }
                                u32 A2 = Abest | (1u << h);
                                i64 v2 = useFilter ? coupledA(A2, i, argBuf[cur ^ 1], mpBuf[cur ^ 1], &kcBuf[cur ^ 1]) : coupledA(A2, i); if (v2 < 0) continue;
                                i64 s2 = v2 + (sBest - vBest) - hUB[h];
                                if (s2 < sBest) { sBest = s2; Abest = A2; vBest = v2; greedyAdds++; cur ^= 1; nOk = false; }
                            }
                        }
                    }
                    if (sBest < grpBound()) { grpMask = Abest; grpV = vBest; }
                }
                Seff = min(S, grpBound());
                if (res && !(Seff > thr31)) { res = false; grpPrune++; }
            }
        }
        tsc[2] += __rdtsc() - t_2;
        if ((res || forceAll) && useDir && i >= dirFrom && i < 61) {
            i64 sd = directS(i); if (sd < Seff) Seff = sd;
            if (res && !(Seff > thr31)) { res = false; dirPrune++; }
        }
        if (res && i == dumpLab && dumpLeft > 0 && (nodes % dumpEvery) == 0) { dumpLeft--;
            vector<tuple<double,int,int,int>> v; for (int h = 0; h < 30; h++) v.push_back({hUB[h] / 262144.0, lvl[h] == nlev ? 0 : 1, 16 - hcnt[h], h});
            sort(v.rbegin(), v.rend());
            printf("DUMP label %d slack/31=%.2f h0=%.2f grpGain=%.1f |A|=%d K=%d:", i, (double)(Seff - thr31) / 31.0 / 262144.0, h0U / 262144.0, (S - min(S, grpBound())) / 262144.0, __builtin_popcount(grpMask), K);
            for (auto& [c, on, un, h] : v) printf(" %.1f%s%d", c, on ? "L" : "o", un); printf("\n"); }
        if (res && chkLeft > 0 && (nodes % chkEvery) == 0) {   // raw data for the independent Python check of the three bounds
            chkLeft--;
            u32 A = grpMask ? grpMask : (u32)((nodes * 2654435761u) & 0x3FFFFFFF);
            printf("CHK %d |", i); for (int q = 0; q <= i; q++) printf(" %d", asg[q]);
            printf(" | %u %lld | %lld |", A, (long long)coupledA(A, i), (long long)(K <= 16 ? directS(i) : -1));
            for (int h = 0; h < 30; h++) printf(" %d:%lld", hypNormal[h], (long long)peCap(h, i));
            printf("\n");
        }
        if (res) { dAlive[i]++; int no = 0; for (int h = 0; h < 30; h++) if (lvl[h] == nlev) no++; dOff[i][no]++; }
        return res;
    }
    void pop(int i) {
        { int t2 = 0; for (int h : HofL[i]) { HU& V = hund[i][t2++]; if (V.newmap) hmap[h][asg[i]] = -1; hk[h] = V.hk; hpos[h] = V.hpos; lvl[h] = V.lvl; hkey[h] = V.key; } }
        int t = 0;
        for (int f : FofL[i]) { Und& U = und[i][t++]; FS& F = fr[f]; if (U.newmap) F.map[asg[i]] = -1; F.idx = U.idx; F.k = U.k; F.j = U.j; fub[f] = U.fub; }
        HS& H = hsave[i]; memcpy(hFS, H.hFS, sizeof hFS); memcpy(hEx, H.hEx, sizeof hEx); memcpy(hUB, H.hUB, sizeof hUB); memcpy(hPE, H.hPE, sizeof hPE); memcpy(hcnt, H.hcnt, sizeof hcnt); S = H.S; G = H.G; grpV = H.grpV; grpMask = H.grpMask; K = H.K;
    }
    void leaf() {
        leaves++;
        uint16_t vec[64][32]; memset(vec, 0, sizeof vec);
        for (int i = 0; i < NLAB; i++) for (int x = 0; x < 32; x++) vec[asg[i]][x] += GV5[i][x];
        u64 M = evalM<32>(K, vec);
        // consistency: H5 <= S/31  <=>  M * 2^25 * 105 <= S * 2^(K+1)
        if ((i128)M * ((i128)UNIT_H * 105) > (i128)S * ((i128)1 << (K + 1))) consistencyFail++;
        if ((i128)M * ((i128)UNIT_H * 105) > (i128)min(S, grpBound()) * ((i128)1 << (K + 1))) consistencyFail++;
        if ((i128)M << bestK > bestNum << K || bestAsg.empty()) { bestNum = M; bestK = K; bestAsg.assign(asg, asg + NLAB); }
        if ((i128)M * 13440 > (i128)tauM * 62 * ((i128)1 << K)) viol++;
        if (leafOut) { fprintf(leafOut, "%.6f %.4f", (double)M / (62.0 * (double)(1ull << K)) * 13440.0, (double)S / 31.0 / 262144.0); for (int i = 0; i < NLAB; i++) fprintf(leafOut, " %d", asg[i]); fprintf(leafOut, "\n"); }
    }
    void dfs(int i) {
        if (i == NLAB) { leaf(); return; }
        int t30 = i > 30 ? type(asg[30]) : 0;
        bool pre = useBatch && usePE && nlev && !forceAll;
        if (pre && !(i & 1)) {   // first label of a pair: tables for the hyperplanes that are already off the lists
            u64 t_0 = __rdtsc();
            for (int t = 0; t < 15; t++) { int h = HofL[i][t]; if (lvl[h] == nlev) { PairEntry& E = pbOwn(i, t); if (!E.tried) fillPair(i, t); } }
            tsc[3] += __rdtsc() - t_0;
        }
        i64 gB = grpBound();
        for (int c = 0; c <= K; c++) {
            if (i > 30 && type(c) < t30) continue;   // kernel symmetry breaking
            if (pre && !batchChk) {
                // Bound on the child before it is built: no cap can rise, and the tables give the child's partial-exact
                // caps. A child that fails this test would be pruned by the same numbers after it is built.
                i64 dS = 0, dG = 0;
                for (int t = 0; t < 15; t++) { PairEntry& E = pbAt(i & ~1, t); if (E.owner != ser[(i & ~1) - 1] || !E.valid) continue;
                    int h = HofL[i][t]; i64 pe = (i & 1) ? pe2(E, asg[i - 1], c) : pe1(E, c);
                    if (pe < hUB[h]) { dS += hUB[h] - pe; if (!(grpMask >> h & 1)) dG += hUB[h] - pe; } }
                if (!(min(S - dS, gB - dG) > thr31)) { nodes++; dPush[i]++; prePrune++; continue; }
            }
            if (push(i, c) && !lexPrune(i + 1)) dfs(i + 1);
            pop(i);
        }
    }
};

struct Task { int root; vector<int> pre; };

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "usage: bb5 datadir tauM rootsfile outprefix [D] [maxTasks]\n"); return 1; }
    string dir = argv[1]; tauM = string(argv[2]) == "path" ? 4788 : atoi(argv[2]); string rf = argv[3]; string outp = argv[4];
    int D = argc > 5 ? atoi(argv[5]) : 3; long maxTasks = argc > 6 ? atol(argv[6]) : -1;
    if (argc > 8) {   // envelope file from bb4env and the loss threshold L of that run
        int Lenv = atoi(argv[8]); gsz = 256LL * Lenv;
        vector<i64> env(gsz, -1);
        FILE* g = fopen(argv[7], "r"); long long a, b; while (fscanf(g, "%lld %lld", &a, &b) == 2) if (a < gsz) env[a] = max(env[a], (i64)b); fclose(g);
        gstar.assign(gsz, 0);
        i64 run = (i64)(4860 - Lenv) << 18;   // partitions with frame loss >= Lenv: H4 <= (4860-Lenv)/13440
        for (i64 i = gsz - 1; i >= 0; i--) { run = max(run, env[i]); gstar[i] = run; }
        printf("envelope from %s, L=%d: g*(0)=%.2f g*(40)=%.2f g*(42)=%.2f g*(50)=%.2f g*(60)=%.2f g*(70)=%.2f (x13440)\n", argv[7], Lenv,
               gstar[0] / 262144.0, gstar[min<i64>(gsz - 1, 40 * 256LL)] / 262144.0, gstar[min<i64>(gsz - 1, 42 * 256LL)] / 262144.0, gstar[min<i64>(gsz - 1, 50 * 256LL)] / 262144.0,
               gstar[min<i64>(gsz - 1, 60 * 256LL)] / 262144.0, gstar[min<i64>(gsz - 1, 70 * 256LL)] / 262144.0);
    }
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
    {   // self-test: AVX-512 evaluator against the SSE one on random weighted vectors
        mt19937 rng(77); long bad = 0;
        for (int t = 0; t < 300; t++) {
            int K = 1 + rng() % 12; alignas(64) static uint16_t v[64][32];
            for (int k = 0; k < K; k++) for (int x = 0; x < 32; x++) v[k][x] = rng() % 40;
            u64 ref = evalM<32>(K, v);
            if (ref != evalM32z(K, v)) bad++;
            static uint8_t arg[1 << 12];
            if (ref != evalM32arg(K, v, arg)) bad++;
            // recorded maximum points: lowest-index maximiser of each subset (checked directly on 64 subsets)
            int r = K < 4 ? K : 4, outer = K - r;
            for (int q = 0; q < 64; q++) { u32 idx = rng() % (1u << K); int sum[32] = {0};
                for (int p = 0; p < K; p++) if (idx >> p & 1) { int blk = p < r ? outer + p : p - r; for (int x = 0; x < 32; x++) sum[x] += v[blk][x]; }
                int bx = 0; for (int x = 1; x < 32; x++) if (sum[x] > sum[bx]) bx = x;
                if (arg[idx] != bx) bad++; }
        }
        printf("evaluator selftest (AVX-512 vs SSE) mismatches: %ld\n", bad);
        if (bad) return 9;
        { Worker* W = new Worker(); long ncmp = 0; long pbad = W->pairSelfTest(ncmp); delete W;
          printf("pair-table selftest mismatches: %ld (%ld placements compared with direct evaluation)\n", pbad, ncmp);
          if (pbad) return 9; }
    }
    if (argc > 10) {
        FILE* f = fopen(argv[9], "rb"); int F = atoi(argv[10]);
        map<i64, vector<u128>, greater<i64>> byVal; uint8_t rep[30]; uint64_t n; long tot = 0;
        while (fread(rep, 1, 30, f) == 30) {
            if (fread(&n, 8, 1, f) != 1) return 5;
            vector<u128> k(n); if (fread(k.data(), 16, n, f) != n) return 5;
            int r[30]; for (int i = 0; i < 30; i++) r[i] = rep[i];
            i64 v = exactH4U(r, rootLab, RV);
            if (v <= ((i64)F << 18)) { fprintf(stderr, "list contains a value at or below the floor\n"); return 6; }
            auto& dst = byVal[v]; dst.insert(dst.end(), k.begin(), k.end()); tot += n;
        }
        fclose(f);
        { int lv = 0; size_t tot2 = 0; for (auto& [v, k] : byVal) tot2 += k.size();
          vector<pair<u128, uint8_t>> kv; kv.reserve(tot2);
          for (auto& [v, k] : byVal) { for (auto& x : k) kv.push_back({x, (uint8_t)lv}); capVal.push_back(v); lv++; k.clear(); k.shrink_to_fit(); }
          sort(kv.begin(), kv.end());
          allKeys.resize(kv.size()); keyLev.resize(kv.size());
          for (size_t i = 0; i < kv.size(); i++) { allKeys[i] = kv[i].first; keyLev[i] = kv[i].second; if (i && allKeys[i] == allKeys[i - 1]) { fprintf(stderr, "duplicate key\n"); return 8; } }
          size_t nb = kv.size() / BS; blkMin.assign(nb, 255);
          for (size_t i = 0; i < nb * BS; i++) blkMin[i / BS] = min(blkMin[i / BS], keyLev[i]);
          sparse.push_back(blkMin);
          for (int k = 1; (1ull << k) <= nb; k++) { vector<uint8_t> r(nb - (1ull << k) + 1); for (size_t i = 0; i < r.size(); i++) r[i] = min(sparse[k - 1][i], sparse[k - 1][i + (1ull << (k - 1))]); sparse.push_back(r); }
          nlev = lv; }
        capVal.push_back((i64)F << 18);
        lists.resize(nlev);
        printf("level lists: %ld labelled partitions;", tot);
        for (int k = 0; k <= nlev; k++) printf(" %s%.2f", k == nlev ? "floor " : "", capVal[k] / 262144.0);
        printf("\n");
        // self-test: d=4 map of every hyperplane preserves H4 on a few random partitions
        mt19937 rng(5); long bad = 0;
        for (int t = 0; t < 20; t++) {
            int p[62]; int k = 3 + rng() % 6; for (int i = 0; i < 62; i++) p[i] = rng() % k;
            for (int h = 0; h < 30; h++) {
                int q[30]; for (int i = 0; i < 62; i++) if (dpos[h][i] >= 0) q[dpos[h][i]] = p[i];
                if (exactH4U(q, rootLab, RV) != exactH4U(p, hypLab[h], HV[h])) bad++;
            }
        }
        printf("hyperplane-map selftest mismatches: %ld\n", bad);
        if (bad) return 7;
    }
    if (string(argv[2]) == "path") {   // soundness test: replay full partitions, check every bound along the path
        FILE* f = fopen(rf.c_str(), "r"); int nP = 0; long bad = 0, bad2 = 0, tighter = 0, checks = 0; u64 chk = 0; double minRatio = 1e9;
        Worker* W = new Worker();
        while (true) {
            int p[62]; bool ok = true; for (int i = 0; i < 62; i++) if (fscanf(f, "%d", &p[i]) != 1) { ok = false; break; }
            if (!ok) break;
            nP++;
            i64 fin[30]; for (int h = 0; h < 30; h++) fin[h] = exactH4U(p, hypLab[h], HV[h]);
            uint16_t vec[64][32]; memset(vec, 0, sizeof vec); int K = 0;
            for (int i = 0; i < 62; i++) { K = max(K, p[i] + 1); for (int x = 0; x < 32; x++) vec[p[i]][x] += GV5[i][x]; }
            u64 M = evalM<32>(K, vec);
            Root R; R.rgs.assign(p, p + 30);
            W->forceAll = true; W->dirFrom = getenv("DIRFROM") ? atoi(getenv("DIRFROM")) : 30;
            if (getenv("GREEDYUN")) W->greedyUn = atoi(getenv("GREEDYUN"));
            W->initRoot(R);
            { static mt19937 grng(getenv("PATHSEED") ? atoi(getenv("PATHSEED")) : 1); for (int i = 30; i < 62; i++) W->asg[i] = grng() % 9; }   // garbage in the unplaced slots: bounds must not read them
            for (int h = 0; h < 30; h++) if (fin[h] > W->h0U) { fprintf(stderr, "input %d: root is not a max hyperplane\n", nP); }
            for (int i = 30; i <= 62; i++) {
                checks++;
                for (int h = 0; h < 30; h++) if (fin[h] >= 0 && W->hUB[h] < fin[h]) bad++;
                if ((i128)W->S * ((i128)1 << K) < (i128)M * ((i128)1 << 24) * 105) bad++;
                if ((i128)W->Seff * ((i128)1 << K) < (i128)M * ((i128)1 << 24) * 105) bad2++;
                if (W->Seff < W->S) tighter++;
                { u64 hsh = (u64)W->Seff * 1000003ULL + (u64)W->S; for (int h = 0; h < 30; h++) hsh = hsh * 1099511628211ULL + (u64)W->hUB[h]; chk = chk * 6364136223846793005ULL + hsh; }
                { double sl = (double)W->Seff * (double)(1ull << K) / ((double)M * (double)(1ull << 24) * 105.0); if (sl < minRatio) minRatio = sl; }
                if (i < 62) W->push(i, p[i]);
            }
            for (int i = 61; i >= 30; i--) W->pop(i);
        }
        printf("path check: %d partitions, %ld node checks, %ld bound violations, %ld violations of the tightened bound (tighter at %ld nodes); min bound/truth ratio %.6f; checksum %016llx\n", nP, checks, bad, bad2, tighter, minRatio, (unsigned long long)chk);
        return 0;
    }
    { FILE* f = fopen(rf.c_str(), "r"); char val[64]; int sz;
      while (fscanf(f, "%63s %d", val, &sz) == 2) { Root R; R.val = val; R.rgs.resize(30); for (int i = 0; i < 30; i++) if (fscanf(f, "%d", &R.rgs[i]) != 1) return 3; roots.push_back(R); }
      fclose(f);
      if (getenv("DSYM")) DSYM = atoi(getenv("DSYM"));
      size_t tot = 0; for (auto& R : roots) { if (DSYM > 0) buildRootSym(R); tot += R.hinv.size(); }
      printf("lex-leader depth %d, symmetry elements per root (total %zu):", DSYM, tot); for (auto& R : roots) printf(" %zu", R.hinv.size() + 1); printf("\n"); }
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
                    if (W->push(i, c) && !W->lexPrune(i + 1)) { pre.push_back(c); gen(i + 1); pre.pop_back(); }
                    W->pop(i);
                }
            };
            gen(30);
        }
        delete W;
    }
    printf("total tasks generated: %zu\n", tasks.size()); fflush(stdout);
    if (getenv("TLO") && getenv("THI")) {
        long lo = atol(getenv("TLO")), hi = min<long>(atol(getenv("THI")), tasks.size());
        tasks = lo < hi ? vector<Task>(tasks.begin() + lo, tasks.begin() + hi) : vector<Task>();
        printf("CHUNK tasks [%ld,%ld)\n", lo, hi); fflush(stdout);
    }
    if (maxTasks > 0 && (long)tasks.size() > maxTasks) {  // sampling mode for timing estimates
        mt19937 rng(1); shuffle(tasks.begin(), tasks.end(), rng); tasks.resize(maxTasks);
        printf("SAMPLING %ld tasks only (not a proof run)\n", maxTasks);
    }
    printf("tasks: %zu\n", tasks.size()); fflush(stdout);
    { Worker* W0 = new Worker();
      if (getenv("NOPE")) W0->usePE = false; if (getenv("NOGRP")) W0->useGrp = false; if (getenv("NODIR")) W0->useDir = false;
      if (getenv("NOBATCH")) W0->useBatch = false; if (getenv("BATCHCHK")) W0->batchChk = true; if (getenv("NOFILTER")) W0->useFilter = false;
      if (getenv("DIRFROM")) W0->dirFrom = atoi(getenv("DIRFROM")); if (getenv("GREEDYUN")) W0->greedyUn = atoi(getenv("GREEDYUN"));
      if (getenv("GREEDYMODE")) W0->greedyMode = atoi(getenv("GREEDYMODE")); if (getenv("MAXKC")) W0->maxKc = atoi(getenv("MAXKC"));
      printf("bounds: partial-exact=%d coupled=%d greedyUn=%d greedyMode=%d maxKc=%d direct=%d from label %d batch=%d%s filter=%d\n", W0->usePE, W0->useGrp, W0->greedyUn, W0->greedyMode, W0->maxKc, W0->useDir, W0->dirFrom, W0->useBatch, W0->batchChk ? " (check mode)" : "", W0->useFilter); fflush(stdout); delete W0; }
    int nth = max(1u, thread::hardware_concurrency());
    vector<Worker*> ws(nth); atomic<int> next(0), done(0);
    auto t0 = chrono::steady_clock::now();
    vector<thread> th;
    for (int t = 0; t < nth; t++) {
        ws[t] = new Worker(); memset(ws[t]->dOff, 0, sizeof ws[t]->dOff);
        if (getenv("NOPE")) ws[t]->usePE = false; if (getenv("NOGRP")) ws[t]->useGrp = false; if (getenv("NODIR")) ws[t]->useDir = false;
        if (getenv("NOBATCH")) ws[t]->useBatch = false; if (getenv("BATCHCHK")) ws[t]->batchChk = true; if (getenv("NOFILTER")) ws[t]->useFilter = false;
        if (getenv("DIRFROM")) ws[t]->dirFrom = atoi(getenv("DIRFROM"));
        if (getenv("GREEDYUN")) ws[t]->greedyUn = atoi(getenv("GREEDYUN"));
        if (getenv("GREEDYMODE")) ws[t]->greedyMode = atoi(getenv("GREEDYMODE"));
        if (getenv("MAXKC")) ws[t]->maxKc = atoi(getenv("MAXKC"));
        if (getenv("CHKDUMP") && t == 0) { ws[t]->chkLeft = atoi(getenv("CHKDUMP")); ws[t]->chkEvery = getenv("CHKEVERY") ? atol(getenv("CHKEVERY")) : 1; }
        if (getenv("DUMPLAB") && t == 0) { ws[t]->dumpLab = atoi(getenv("DUMPLAB")); ws[t]->dumpLeft = 12; ws[t]->dumpEvery = getenv("DUMPEVERY") ? atol(getenv("DUMPEVERY")) : 1; }
        if (getenv("LEAFOUT")) ws[t]->leafOut = fopen((string(getenv("LEAFOUT")) + "." + to_string(t)).c_str(), "w");
        th.emplace_back([&, t] {
            Worker& W = *ws[t];
            for (;;) {
                int k = next++; if (k >= (int)tasks.size()) break;
                Task& T = tasks[k];
                W.initRoot(roots[T.root]); u64 n0 = W.nodes;
                bool alive = true; int i = 30;
                for (; i < 30 + (int)T.pre.size(); i++) if (!W.push(i, T.pre[i - 30])) { alive = false; i++; break; }
                if (alive) W.dfs(i);
                for (int j = i - 1; j >= 30; j--) W.pop(j);
                W.rootNodes[T.root] += W.nodes - n0;
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
    { u64 ld = 0, sp = 0; for (auto w : ws) { ld += w->levelDrops; sp += w->symPruned; } printf("level drops: %llu, lex-leader prunes: %llu\n", (unsigned long long)ld, (unsigned long long)sp);
      vector<pair<u64, int>> rn; for (int r = 0; r < (int)roots.size(); r++) { u64 t = 0; for (auto w : ws) t += w->rootNodes[r]; rn.push_back({t, r}); }
      sort(rn.rbegin(), rn.rend()); printf("heaviest roots (nodes, root, value, |sym|):"); for (int q = 0; q < min<int>(6, rn.size()); q++) printf(" (%.3g, %d, %s, %zu)", (double)rn[q].first, rn[q].second, roots[rn[q].second].val.c_str(), roots[rn[q].second].hinv.size() + 1); printf("\n"); }
    printf("done in %.1fs: nodes=%llu hyperplane-exacts=%llu leaves=%llu violations(H5>tau)=%llu consistencyFail=%llu\n", secs,
           (unsigned long long)nodes, (unsigned long long)he, (unsigned long long)leaves, (unsigned long long)viol, (unsigned long long)cf);
    { u64 a[6] = {0,0,0,0,0,0}; for (auto w : ws) { a[0] += w->peEvals; a[1] += w->pePrune; a[2] += w->grpEvals; a[3] += w->grpPrune; a[4] += w->dirEvals; a[5] += w->dirPrune; }
      { u64 ga = 0; for (auto w : ws) ga += w->greedyAdds; printf("greedy adds %llu\n", (unsigned long long)ga); }
      {
        { u64 a[4] = {0,0,0,0}; for (auto w : ws) for (int q = 0; q < 4; q++) a[q] += w->tsc[q]; double tot = a[0] + a[1] + a[2] + a[3];
          printf("time split (tsc): frames+lists+exact %.1f%%, partial-exact in push %.1f%%, coupled %.1f%%, batch at parent %.1f%%\n", 100 * a[0] / tot, 100 * a[1] / tot, 100 * a[2] / tot, 100 * a[3] / tot); }
        { u64 fl = 0, nb = 0; for (auto w : ws) { fl += w->filtered; nb += w->nBuilds; } printf("coupled filter: candidates skipped %llu, tables built %llu\n", (unsigned long long)fl, (unsigned long long)nb); }
        { u64 be = 0, pp = 0, bm = 0, tb = 0, th_ = 0; for (auto w : ws) { be += w->batchEvals; pp += w->prePrune; bm += w->batchMismatch; tb += w->tieBuilds; th_ += w->tableHits; }
          printf("pair tables: evaluations %llu, tie tables %llu, caps read %llu, children pruned before push %llu, mismatches against peCap %llu\n", (unsigned long long)be, (unsigned long long)tb, (unsigned long long)th_, (unsigned long long)pp, (unsigned long long)bm); }
      }
      printf("new bounds: partial-exact evals %llu prunes %llu; coupled evals %llu prunes %llu; direct evals %llu prunes %llu\n", (unsigned long long)a[0], (unsigned long long)a[1], (unsigned long long)a[2], (unsigned long long)a[3], (unsigned long long)a[4], (unsigned long long)a[5]); }
    { printf("depth profile (label: pushes alive | alive by #off-list hyperplanes 0,1-2,3-5,6-10,11+):\n");
      for (int i = 30; i < NLAB; i++) { u64 a = 0, b = 0; u64 o[5] = {0,0,0,0,0}; for (auto w : ws) { a += w->dPush[i]; b += w->dAlive[i]; for (int q = 0; q < 31; q++) o[q == 0 ? 0 : q <= 2 ? 1 : q <= 5 ? 2 : q <= 10 ? 3 : 4] += w->dOff[i][q]; }
        printf("  %d: %llu %llu | %llu %llu %llu %llu %llu\n", i, (unsigned long long)a, (unsigned long long)b, (unsigned long long)o[0], (unsigned long long)o[1], (unsigned long long)o[2], (unsigned long long)o[3], (unsigned long long)o[4]); }
      u64 on = 0, off = 0, hh[64] = {0}; for (auto w : ws) { on += w->onCnt; off += w->offCnt; for (int q = 0; q < 64; q++) hh[q] += w->offHist[q]; }
      printf("hyperplanes completed: on-list %llu, off-list %llu; off-list exact H4 histogram (4764-4k .. ):", (unsigned long long)on, (unsigned long long)off);
      for (int q = 0; q < 64; q++) if (hh[q]) printf(" [%d]=%llu", 4764 - 4 * q, (unsigned long long)hh[q]); printf("\n");
      for (auto w : ws) if (w->leafOut) fclose(w->leafOut); }
    if (bw) { printf("best leaf H5 = %.9f asg:", (double)bw->bestNum / (62.0 * (double)(1ull << bw->bestK))); for (int x : bw->bestAsg) printf(" %d", x); printf("\n"); }
    return 0;
}
