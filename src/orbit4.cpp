// Reduce d=4 partitions to orbit representatives under G4 = AGL(4,2) x swap (order 645120),
// acting on labels (a,b) by (a,b) -> (M a, b ^ <a,c> ^ s).  Canonical form = lex-min RGS.
// Input lines: "value asg0 ... asg29" (value is passed through).  Output: one line per orbit:
// "value orbitSize rgs0 ... rgs29", plus orbit counts on stderr.
#include <bits/stdc++.h>
using namespace std;
static int par(int v) { return __builtin_popcount(v) & 1; }
int main() {
    // GL(4,2) via images of basis vectors
    vector<array<uint8_t, 30>> G;
    for (int u1 = 1; u1 < 16; u1++) for (int u2 = 1; u2 < 16; u2++) for (int u3 = 1; u3 < 16; u3++) for (int u4 = 1; u4 < 16; u4++) {
        int us[4] = {u1, u2, u3, u4}; set<int> span = {0}; bool ok = true;
        for (int u : us) { if (span.count(u)) { ok = false; break; } set<int> s2 = span; for (int v : span) s2.insert(v ^ u); span = s2; }
        if (!ok) continue;
        for (int c = 0; c < 16; c++) for (int s = 0; s < 2; s++) {
            array<uint8_t, 30> p;
            for (int i = 0; i < 30; i++) {
                int a = i / 2 + 1, b = i % 2, Ma = 0;
                for (int k = 0; k < 4; k++) if (a >> k & 1) Ma ^= us[k];
                p[i] = 2 * (Ma - 1) + (b ^ par(a & c) ^ s);
            }
            G.push_back(p);
        }
    }
    fprintf(stderr, "group order %zu\n", G.size());
    map<array<uint8_t, 30>, pair<string, int>> reps;
    string line; long nin = 0;
    while (getline(cin, line)) {
        istringstream is(line); string val; is >> val; int asg[30]; for (int i = 0; i < 30; i++) is >> asg[i];
        nin++;
        array<uint8_t, 30> best; best.fill(255); int cnt = 0;
        for (auto& p : G) {
            int arr[30]; for (int i = 0; i < 30; i++) arr[p[i]] = asg[i];
            int mp[64]; memset(mp, -1, sizeof mp); int k = 0; int cmp = 0; array<uint8_t, 30> img;
            for (int i = 0; i < 30; i++) {
                if (mp[arr[i]] < 0) mp[arr[i]] = k++;
                img[i] = mp[arr[i]];
                if (cmp == 0) { if (img[i] < best[i]) cmp = -1; else if (img[i] > best[i]) { cmp = 1; break; } }
            }
            if (cmp < 0) { best = img; cnt = 1; } else if (cmp == 0) cnt++;
        }
        if (!reps.count(best)) reps[best] = {val, (int)(G.size() / cnt)};
    }
    map<string, pair<int, long>> byVal;
    for (auto& [k, v] : reps) { byVal[v.first].first++; byVal[v.first].second += v.second; }
    fprintf(stderr, "%ld input partitions -> %zu orbits\n", nin, reps.size());
    for (auto it = byVal.rbegin(); it != byVal.rend(); ++it) fprintf(stderr, "  value %s: %d orbits, %ld labelled partitions\n", it->first.c_str(), it->second.first, it->second.second);
    vector<pair<string, array<uint8_t, 30>>> out; for (auto& [k, v] : reps) out.push_back({v.first, k});
    sort(out.begin(), out.end(), [](auto& a, auto& b) { return a.first > b.first; });
    for (auto& [v, k] : out) { printf("%s %d", v.c_str(), reps[k].second); for (int i = 0; i < 30; i++) printf(" %d", k[i]); printf("\n"); }
}
