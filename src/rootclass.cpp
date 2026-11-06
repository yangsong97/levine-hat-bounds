// Classify top local patterns (N >= given minimum) into orbits under the frame group
// G = AGL(3,2) x complement swap (order 2688), acting on labels (a,b) of AG(3,2):
//   (a,b) -> (A a, b ^ <a,c> ^ s).
// Also checks that N is constant on every orbit (invariance check of the table).
// Output: one line per orbit: N orbitSize rgs[0..13]
#include <bits/stdc++.h>
using namespace std;
typedef uint64_t u64; typedef uint32_t u32;
const int NL = 14;
u64 C[NL + 1];
vector<u32> childStart[NL];
vector<uint8_t> kval[NL];

void build() {
    kval[0] = {0}; C[0] = 1;
    for (int j = 0; j < NL; j++) {
        childStart[j].resize(C[j]); u64 s = 0;
        for (u64 i = 0; i < C[j]; i++) { childStart[j][i] = (u32)s; s += kval[j][i] + 1; }
        C[j + 1] = s;
        if (j + 1 < NL) { kval[j + 1].resize(s); u64 p = 0; for (u64 i = 0; i < C[j]; i++) { int k = kval[j][i]; for (int c = 0; c <= k; c++) kval[j + 1][p++] = (c < k ? k : k + 1); } }
    }
}
u32 rankRGS(const int* r) { u32 idx = 0; for (int j = 0; j < NL; j++) idx = childStart[j][idx] + r[j]; return idx; }

int main(int argc, char** argv) {
    string dir = argv[1]; u32 Nmin = atoi(argv[2]);
    build();
    vector<u32> Ntab(C[NL]);
    { FILE* f = fopen((dir + "/Ntab.bin").c_str(), "rb"); size_t got = fread(Ntab.data(), 4, C[NL], f); fclose(f); if (got != C[NL]) { puts("short read"); return 1; } }
    // group
    vector<array<int, NL>> G;
    for (int u1 = 1; u1 < 8; u1++) for (int u2 = 1; u2 < 8; u2++) if (u2 != u1) for (int u3 = 1; u3 < 8; u3++) {
        if (u3 == u1 || u3 == u2 || u3 == (u1 ^ u2)) continue;
        for (int c = 0; c < 8; c++) for (int s = 0; s < 2; s++) {
            array<int, NL> p;
            for (int i = 0; i < NL; i++) {
                int a = i / 2 + 1, b = i % 2;
                int Aa = ((a & 1) ? u1 : 0) ^ ((a & 2) ? u2 : 0) ^ ((a & 4) ? u3 : 0);
                int nb = b ^ (__builtin_popcount(a & c) & 1) ^ s;
                p[i] = 2 * (Aa - 1) + nb;
            }
            G.push_back(p);
        }
    }
    fprintf(stderr, "group size %zu\n", G.size());
    // read top
    FILE* f = fopen((dir + "/top.bin").c_str(), "rb");
    vector<array<uint8_t, 18>> recs; array<uint8_t, 18> r;
    while (fread(r.data(), 1, 18, f) == 18) recs.push_back(r);
    fclose(f);
    map<u64, pair<int, u32>> orbits;  // canonical key -> (size, N)
    map<u32, u64> levelCount;
    long bad = 0;
    for (auto& rec : recs) {
        u32 idx; memcpy(&idx, rec.data(), 4);
        u32 N = Ntab[idx];
        if (N < Nmin) continue;
        levelCount[N]++;
        int rg[NL]; for (int i = 0; i < NL; i++) rg[i] = rec[4 + i];
        if (rankRGS(rg) != idx) { puts("rank mismatch"); return 1; }
        u64 best = ~0ull; set<u64> imgs;
        for (auto& p : G) {
            int arr[NL]; for (int i = 0; i < NL; i++) arr[p[i]] = rg[i];
            int mp[NL]; memset(mp, -1, sizeof mp); int k = 0; u64 key = 0; int out[NL];
            for (int i = 0; i < NL; i++) { if (mp[arr[i]] < 0) mp[arr[i]] = k++; out[i] = mp[arr[i]]; key = key * 16 + out[i]; }
            if (Ntab[rankRGS(out)] != N) bad++;
            imgs.insert(key); best = min(best, key);
        }
        auto it = orbits.find(best);
        if (it == orbits.end()) orbits[best] = {(int)imgs.size(), N};
    }
    fprintf(stderr, "invariance violations: %ld\n", bad);
    map<u32, pair<int, u64>> perLevel;
    for (auto& [key, v] : orbits) { perLevel[v.second].first++; perLevel[v.second].second += v.first; }
    for (auto& [N, v] : perLevel) fprintf(stderr, "N=%u (%u/896): patterns %llu, orbits %d, sum of orbit sizes %llu\n", N, N / 256, (unsigned long long)levelCount[N], v.first, (unsigned long long)v.second);
    // output reps sorted by N desc
    vector<pair<u32, u64>> reps; for (auto& [key, v] : orbits) reps.push_back({v.second, key});
    sort(reps.begin(), reps.end(), [](auto& a, auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
    for (auto& [N, key] : reps) {
        printf("%u %d", N, orbits[key].first);
        for (int i = NL - 1; i >= 0; i--) printf(" %d", (int)((key >> (4 * i)) & 15));
        printf("\n");
    }
    return 0;
}
