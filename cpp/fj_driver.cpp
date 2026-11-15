// Feasibility Jump (Luteberget & Sartor, 2023) driver for QOBLIB Market Split.
//
// Model given to FJ (pure feasibility, zero objective, as in the FJ paper):
//   x_j in {0,1}                       (integer variables with bounds [0,1])
//   sum_j a_ij x_j  = b_i              for every row i      (equality rows)
//   q-k <= sum_j x_j <= q+k            cardinality band, same as RGSA's
// FJ is run with the reference implementation's default settings
// (weight decay 1.0, its internal move-selection constants), single thread.
//
// The unmodified FJ code only reports feasible solutions.  A documented,
// read-only observer hook (third_party/feasibilityjump/rgsa-observer.patch)
// lets this driver record the best incumbent among those satisfying the
// cardinality band, measured by the Euclidean residual ||Ax-b||_2 exactly as
// for every other heuristic.  Output: one JSON object, same keys as cpp/rgsa.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "feasibilityjump.hh"
#include "qoblib.hpp"

using Clock = std::chrono::steady_clock;

static std::string num(double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.10g", v);
    return buf;
}

int main(int argc, char** argv) {
    std::string path;
    int seed = 42, band = 0, q = -1;
    double time_limit = 0, decay = 1.0;
    std::vector<double> cps;
    bool print_x = true;
    try {
        for (int a = 1; a < argc; ++a) {
            std::string k = argv[a];
            auto val = [&]() -> std::string {
                if (a + 1 >= argc) throw std::runtime_error("missing value for " + k);
                return argv[++a];
            };
            if (k == "--instance") path = val();
            else if (k == "--seed") seed = std::stoi(val());
            else if (k == "--band") band = std::stoi(val());
            else if (k == "--q") q = std::stoi(val());
            else if (k == "--time") time_limit = std::stod(val());
            else if (k == "--decay") decay = std::stod(val());
            else if (k == "--checkpoints") {
                std::stringstream ss(val());
                std::string t;
                while (std::getline(ss, t, ',')) if (!t.empty()) cps.push_back(std::stod(t));
                std::sort(cps.begin(), cps.end());
            } else if (k == "--no-x") print_x = false;
            else throw std::runtime_error("unknown option " + k);
        }
        if (path.empty() || time_limit <= 0)
            throw std::runtime_error("--instance and --time are required");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\nusage: fj_driver --instance F --time S [--seed S] "
                             "[--band K] [--q Q] [--decay D] [--checkpoints a,b] [--no-x]\n", e.what());
        return 2;
    }
    Instance I = read_instance(path);
    const int q0 = q >= 0 ? q : I.n / 2;
    const int qlo = std::max(0, q0 - band), qhi = std::min(I.n, q0 + band);

    FeasibilityJumpSolver solver(seed, 0, decay);
    for (int j = 0; j < I.n; ++j) solver.addVar(VarType::Integer, 0.0, 1.0, 0.0);
    std::vector<int> idx(I.n);
    std::vector<double> coef(I.n);
    std::vector<int> row_cid(I.m);
    for (int r = 0; r < I.m; ++r) {
        int nz = 0;
        for (int j = 0; j < I.n; ++j)
            if (I.w(j)[r] != 0) { idx[nz] = j; coef[nz] = static_cast<double>(I.w(j)[r]); ++nz; }
        row_cid[r] = solver.addConstraint(RowType::Equal, static_cast<double>(I.b[r]), nz,
                                          idx.data(), coef.data(), 0);
        if (row_cid[r] < 0 || row_cid[r] == INT_MAX) {
            std::fprintf(stderr, "error: row %d could not be added\n", r);
            return 2;
        }
    }
    for (int j = 0; j < I.n; ++j) { idx[j] = j; coef[j] = 1.0; }
    solver.addConstraint(RowType::Gte, qlo, I.n, idx.data(), coef.data(), 0);
    solver.addConstraint(RowType::Lte, qhi, I.n, idx.data(), coef.data(), 0);

    const auto t0 = Clock::now();
    auto elapsed = [&]() { return std::chrono::duration<double>(Clock::now() - t0).count(); };
    i64 best_sq = std::numeric_limits<i64>::max();
    std::vector<char> best_x;
    double time_to_best = 0;
    std::vector<i64> cp_sq(cps.size(), -1);
    size_t next_cp = 0;
    long long steps = 0, late_ignored = 0;
    auto tick = [&](double now) {
        while (next_cp < cps.size() && now >= cps[next_cp]) {
            cp_sq[next_cp] = best_x.empty() ? -1 : best_sq;
            ++next_cp;
        }
    };

    solver.observer = [&](const Problem& p) {
        ++steps;
        if ((steps & 255) == 0) tick(elapsed());
        // cardinality is the LHS of the band rows (last two constraints)
        double card = p.constraints.back().incumbentLhs;
        if (card < qlo - 0.5 || card > qhi + 0.5) return;
        i64 sq = 0;
        for (int r = 0; r < I.m; ++r) {
            i64 d = static_cast<i64>(std::llround(p.constraints[row_cid[r]].incumbentLhs)) - I.b[r];
            sq += d * d;
        }
        if (sq < best_sq) {
            const double now = elapsed();
            // FJ only checks the time limit between batches of steps, so it can
            // run slightly past the budget; answers found after it do not count.
            if (now > time_limit) { ++late_ignored; return; }
            tick(now);
            best_sq = sq;
            best_x.assign(I.n, 0);
            for (int j = 0; j < I.n; ++j) best_x[j] = p.incumbentAssignment[j] > 0.5;
            time_to_best = now;
        }
    };

    bool found_feasible = false;
    solver.solve(nullptr, [&](FJStatus st) -> CallbackControlFlow {
        if (st.solution != nullptr) found_feasible = true;
        if (found_feasible || best_sq == 0) return CallbackControlFlow::Terminate;
        return elapsed() >= time_limit ? CallbackControlFlow::Terminate
                                       : CallbackControlFlow::Continue;
    });
    double total = elapsed();
    tick(std::numeric_limits<double>::infinity());

    std::ostringstream out;
    out << "{\"instance\":\"" << path << "\",\"m\":" << I.m << ",\"n\":" << I.n
        << ",\"mode\":\"feasibility_jump\",\"seed\":" << seed << ",\"band\":" << band
        << ",\"decay\":" << num(decay) << ",\"time_limit\":" << num(time_limit);
    if (best_x.empty()) {
        out << ",\"status\":\"no_vector\",\"residual\":null,\"feasible\":false";
    } else {
        // verify from scratch
        std::vector<i64> res(I.b.size());
        for (int r = 0; r < I.m; ++r) res[r] = -I.b[r];
        int card = 0;
        for (int j = 0; j < I.n; ++j)
            if (best_x[j]) { ++card; for (int r = 0; r < I.m; ++r) res[r] += I.w(j)[r]; }
        i64 sq = 0, linf = 0;
        for (i64 v : res) { sq += v * v; linf = std::max<i64>(linf, v < 0 ? -v : v); }
        if (sq != best_sq) { std::fprintf(stderr, "internal error: residual mismatch\n"); return 3; }
        out << ",\"status\":\"" << (sq == 0 ? "feasible" : "time_limit") << "\""
            << ",\"residual\":" << num(std::sqrt(static_cast<double>(sq)))
            << ",\"squared_residual\":" << sq << ",\"max_abs_violation\":" << linf
            << ",\"feasible\":" << (sq == 0 ? "true" : "false") << ",\"cardinality\":" << card;
    }
    out << ",\"time_to_best\":" << num(time_to_best) << ",\"total_time\":" << num(total)
        << ",\"fj_steps\":" << steps << ",\"late_improvements_ignored\":" << late_ignored << ",\"deadline_enforced\":true,\"checkpoints\":[";
    for (size_t k = 0; k < cps.size(); ++k) {
        out << (k ? "," : "") << "{\"t\":" << num(cps[k]) << ",\"residual\":";
        if (cp_sq[k] < 0) out << "null"; else out << num(std::sqrt(static_cast<double>(cp_sq[k])));
        out << "}";
    }
    out << "]";
    if (print_x && !best_x.empty()) {
        out << ",\"x\":\"";
        for (char c : best_x) out << (c ? '1' : '0');
        out << "\"";
    }
    out << "}\n";
    std::fputs(out.str().c_str(), stdout);
    return 0;
}
