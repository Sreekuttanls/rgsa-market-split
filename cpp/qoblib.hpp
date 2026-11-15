// Shared QOBLIB Market Split instance reader (used by rgsa.cpp, fj_driver.cpp).
#pragma once
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using i64 = std::int64_t;

struct Instance {
    int m = 0, n = 0;
    std::vector<i64> col;  // column-major: col[j*m + r] = A[r][j]
    std::vector<i64> b;
    const i64* w(int j) const { return &col[static_cast<size_t>(j) * m]; }
};

inline Instance read_instance(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open instance: " + path);
    std::vector<std::vector<i64>> rows;
    std::string line;
    int m = -1, n = -1;
    while (std::getline(in, line)) {
        size_t p = line.find_first_not_of(" \t\r");
        if (p == std::string::npos || line[p] == '#') continue;
        std::istringstream ss(line);
        std::vector<i64> vals;
        i64 v;
        while (ss >> v) vals.push_back(v);
        if (m < 0) {
            if (vals.size() < 2) throw std::runtime_error("bad header line");
            m = static_cast<int>(vals[0]);
            n = static_cast<int>(vals[1]);
            continue;
        }
        if (static_cast<int>(rows.size()) < m) rows.push_back(vals);
    }
    if (m <= 0 || n <= 0 || static_cast<int>(rows.size()) != m)
        throw std::runtime_error("malformed instance: " + path);
    Instance inst;
    inst.m = m;
    inst.n = n;
    inst.col.assign(static_cast<size_t>(m) * n, 0);
    inst.b.assign(m, 0);
    for (int r = 0; r < m; ++r) {
        if (static_cast<int>(rows[r].size()) != n + 1)
            throw std::runtime_error("row has wrong length in " + path);
        for (int j = 0; j < n; ++j) inst.col[static_cast<size_t>(j) * m + r] = rows[r][j];
        inst.b[r] = rows[r][n];
    }
    return inst;
}

