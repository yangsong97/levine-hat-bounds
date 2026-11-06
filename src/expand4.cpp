// Expand d=4 orbit representatives to all labelled partitions (images under AGL(4,2) x swap),
// as restricted-growth strings over the standard labels 0..29, packed 4 bits per label
// (label 0 in the most significant nibble) into unsigned __int128.  Fails if a partition
// has more than 16 blocks.
// Input: orbit lines "value size rgs0..rgs29" (from orbit4).  Output (binary): for each orbit,
// 30 bytes of the rep's RGS, uint64 count n, then n sorted keys (16 bytes each).
#include <bits/stdc++.h>
using namespace std;
typedef unsigned __int128 u128;
static int par(int v) { return __builtin_popcount(v) & 1; }
int main(int argc, char** argv) {
    vector<array<uint8_t, 30>> G;
    for (int u1 = 1; u1 < 16; u1++) for (int u2 = 1; u2 < 16; u2++) for (int u3 = 1; u3 < 16; u3++) for (int u4 = 1; u4 < 16; u4++) {
        int us[4] = {u1, u2, u3, u4}; set<int> span = {0}; bool ok = true;
        for (int u : us) { if (span.count(u)) { ok = false; break; } set<int> s2 = span; for (int v : span) s2.insert(v ^ u); span = s2; }
        if (!ok) continue;
        for (int c = 0; c < 16; c++) for (int s = 0; s < 2; s++) {
            array<uint8_t, 30> p;
            for (int i = 0; i < 30; i++) { int a = i / 2 + 1, b = i % 2, Ma = 0; for (int k = 0; k < 4; k++) if (a >> k & 1) Ma ^= us[k]; p[i] = 2 * (Ma - 1) + (b ^ par(a & c) ^ s); }
            G.push_back(p);
        }
    }
    FILE* out = fopen(argv[1], "wb");
    string line; long total = 0; int norb = 0;
    while (getline(cin, line)) {
        istringstream is(line); string val; int sz; is >> val >> sz; int asg[30]; for (int i = 0; i < 30; i++) is >> asg[i];
        vector<u128> keys; keys.reserve(G.size());
        for (auto& p : G) {
            int arr[30]; for (int i = 0; i < 30; i++) arr[p[i]] = asg[i];
            int mp[64]; memset(mp, -1, sizeof mp); int k = 0; u128 key = 0;
            for (int i = 0; i < 30; i++) { if (mp[arr[i]] < 0) mp[arr[i]] = k++; if (mp[arr[i]] > 15) { fprintf(stderr, "more than 16 blocks\n"); return 1; } key = (key << 4) | (u128)mp[arr[i]]; }
            keys.push_back(key << 8);   // 120 bits used, left-aligned in 128
        }
        sort(keys.begin(), keys.end()); keys.erase(unique(keys.begin(), keys.end()), keys.end());
        if ((long)keys.size() != sz) { fprintf(stderr, "orbit size mismatch: %zu vs %d\n", keys.size(), sz); return 2; }
        uint8_t rep[30]; for (int i = 0; i < 30; i++) rep[i] = asg[i];
        uint64_t n = keys.size();
        fwrite(rep, 1, 30, out); fwrite(&n, 8, 1, out); fwrite(keys.data(), 16, n, out);
        total += n; norb++;
    }
    fclose(out);
    fprintf(stderr, "%d orbits, %ld labelled partitions\n", norb, total);
}
