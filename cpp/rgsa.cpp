// Residual-Guided Simulated Annealing (RGSA) for QOBLIB Market Split.
//
// C++17 implementation of the algorithms in src/rgsa_solver.py, extended for
// the revision experiments:
//   * time-based or step-based cooling (time mode uses the whole budget);
//   * a cardinality band q in [floor(N/2)-k, floor(N/2)+k];
//   * polishing that either mirrors the legacy Python rule or runs 1-/2-swap
//     descent to convergence;
//   * a multi-start greedy exchange baseline (no annealing);
//   * best-so-far residual recorded at wall-clock checkpoints.
//
// All residual arithmetic is exact (64-bit integers); Euclidean norms are only
// computed for reporting and for the Metropolis rule.  Output is one JSON
// object on stdout.  Build: see cpp/Makefile.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "qoblib.hpp"

using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------- state
// Selected set S and complement are kept as index arrays with position maps
// so that swaps and uniform sampling are O(1).
struct State {
    const Instance* I = nullptr;
    std::vector<int> in, out;     // members of S and of its complement
    std::vector<int> pos;         // position of item in `in` or `out`
    std::vector<char> sel;        // 1 if item in S
    std::vector<i64> res;         // residual A x - b
    i64 sq = 0;                   // squared Euclidean norm of res

    void init(const Instance& inst, const std::vector<char>& x) {
        I = &inst;
        in.clear(); out.clear();
        pos.assign(inst.n, 0);
        sel = x;
        res.assign(inst.m, 0);
        for (int r = 0; r < inst.m; ++r) res[r] = -inst.b[r];
        for (int j = 0; j < inst.n; ++j) {
            if (x[j]) {
                pos[j] = static_cast<int>(in.size());
                in.push_back(j);
                const i64* wj = inst.w(j);
                for (int r = 0; r < inst.m; ++r) res[r] += wj[r];
            } else {
                pos[j] = static_cast<int>(out.size());
                out.push_back(j);
            }
        }
        sq = 0;
        for (i64 v : res) sq += v * v;
    }
    // squared norm after removing i (in S) and inserting j (not in S)
    i64 swap_sq(int i, int j) const {
        const i64* wi = I->w(i);
        const i64* wj = I->w(j);
        i64 s = 0;
        for (int r = 0; r < I->m; ++r) {
            i64 v = res[r] - wi[r] + wj[r];
            s += v * v;
        }
        return s;
    }
    void apply_swap(int i, int j) {
        const i64* wi = I->w(i);
        const i64* wj = I->w(j);
        sq = 0;
        for (int r = 0; r < I->m; ++r) {
            res[r] += wj[r] - wi[r];
            sq += res[r] * res[r];
        }
        int pi = pos[i], pj = pos[j];
        in[pi] = j; pos[j] = pi;
        out[pj] = i; pos[i] = pj;
        sel[i] = 0; sel[j] = 1;
    }
};

static double norm_of(i64 sq) { return std::sqrt(static_cast<double>(sq)); }

// ---------------------------------------------------------------- options
struct Options {
    std::string instance;
    std::string mode = "sa";        // sa | greedy
    std::string init = "random";    // random | lagrangian | given
    std::string init_x;             // for init=given: string of 0/1
    std::uint64_t seed = 42;
    int q = -1;                     // cardinality; -1 => floor(N/2)
    int band = 0;                   // k: try q in [q-k, q+k]
    double p_guide = 0.65;
    double T0 = 600.0, Tf = 1e-3, Tfloor = 1e-4;
    long long steps = 150000;       // step budget per cardinality (if time<=0)
    double time_limit = 0.0;        // seconds for the whole run (>0 => time mode)
    int lag_iters = 3000;
    double alpha0 = 0.001, lam_floor = 1e-4;
    std::string polish = "none";    // none | legacy | converge
    int polish_passes = 1;          // legacy mode: number of 2-swap passes
    double tol = 1e-5;
    std::vector<double> checkpoints; // seconds
    bool print_x = true;
};

static void usage() {
    std::fprintf(stderr,
        "usage: rgsa --instance FILE [--mode sa|greedy] [--init random|lagrangian|given]\n"
        "  [--init-x 0101..] [--seed S] [--q Q] [--band K] [--p-guide P] [--T0 T] [--Tf T]\n"
        "  [--steps K | --time SECONDS] [--lag-iters L] [--alpha0 A] [--lam-floor E]\n"
        "  [--polish none|legacy|converge] [--polish-passes P] [--tol T]\n"
        "  [--checkpoints s1,s2,...] [--no-x]\n");
}

static std::vector<double> parse_list(const std::string& s) {
    std::vector<double> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (!tok.empty()) out.push_back(std::stod(tok));
    std::sort(out.begin(), out.end());
    return out;
}

static Options parse_args(int argc, char** argv) {
    Options o;
    for (int a = 1; a < argc; ++a) {
        std::string k = argv[a];
        auto val = [&]() -> std::string {
            if (a + 1 >= argc) throw std::runtime_error("missing value for " + k);
            return argv[++a];
        };
        if (k == "--instance") o.instance = val();
        else if (k == "--mode") o.mode = val();
        else if (k == "--init") o.init = val();
        else if (k == "--init-x") o.init_x = val();
        else if (k == "--seed") o.seed = std::stoull(val());
        else if (k == "--q") o.q = std::stoi(val());
        else if (k == "--band") o.band = std::stoi(val());
        else if (k == "--p-guide") o.p_guide = std::stod(val());
        else if (k == "--T0") o.T0 = std::stod(val());
        else if (k == "--Tf") o.Tf = std::stod(val());
        else if (k == "--steps") o.steps = std::stoll(val());
        else if (k == "--time") o.time_limit = std::stod(val());
        else if (k == "--lag-iters") o.lag_iters = std::stoi(val());
        else if (k == "--alpha0") o.alpha0 = std::stod(val());
        else if (k == "--lam-floor") o.lam_floor = std::stod(val());
        else if (k == "--polish") o.polish = val();
        else if (k == "--polish-passes") o.polish_passes = std::stoi(val());
        else if (k == "--tol") o.tol = std::stod(val());
        else if (k == "--checkpoints") o.checkpoints = parse_list(val());
        else if (k == "--no-x") o.print_x = false;
        else if (k == "-h" || k == "--help") { usage(); std::exit(0); }
        else throw std::runtime_error("unknown option " + k);
    }
    if (o.instance.empty()) throw std::runtime_error("--instance is required");
    if (o.mode != "sa" && o.mode != "greedy") throw std::runtime_error("bad --mode");
    if (o.init != "random" && o.init != "lagrangian" && o.init != "given")
        throw std::runtime_error("bad --init");
    if (o.polish != "none" && o.polish != "legacy" && o.polish != "converge")
        throw std::runtime_error("bad --polish");
    if (o.p_guide < 0 || o.p_guide > 1) throw std::runtime_error("--p-guide must be in [0,1]");
    if (o.band < 0) throw std::runtime_error("--band must be >= 0");
    return o;
}

// ---------------------------------------------------------------- tracker
// Global best-so-far across all cardinalities, plus checkpoint snapshots.
struct Tracker {
    Clock::time_point t0;
    double limit = 0.0;
    i64 best_sq = std::numeric_limits<i64>::max();
    std::vector<char> best_x;
    int best_q = -1;
    double time_to_best = 0.0;
    std::vector<double> cps;
    std::vector<i64> cp_sq;   // -1 = no vector yet
    size_t next_cp = 0;

    double elapsed() const {
        return std::chrono::duration<double>(Clock::now() - t0).count();
    }
    void tick(double now) {
        while (next_cp < cps.size() && now >= cps[next_cp]) {
            cp_sq[next_cp] = best_x.empty() ? -1 : best_sq;
            ++next_cp;
        }
    }
    void offer(const State& s, int q) {
        if (s.sq < best_sq) {
            double now = elapsed();
            tick(now);  // checkpoints that passed before this improvement
            best_sq = s.sq;
            best_x = s.sel;
            best_q = q;
            time_to_best = now;
        }
    }
    bool out_of_time() const { return limit > 0 && elapsed() >= limit; }
};

// ---------------------------------------------------------------- stage 1
// Lagrangian dual projection (Algorithm 1). Mirrors solve_stage1_lagrangian
// with the "prefix" strategy.
static std::vector<char> lagrangian(const Instance& I, int q, const Options& o) {
    std::vector<double> lam(I.m, 1.0 / I.m), score(I.n);
    std::vector<int> idx(I.n);
    std::vector<char> x(I.n), best(I.n, 0);
    std::vector<i64> res(I.m);
    i64 best_sq = std::numeric_limits<i64>::max();
    for (int t = 0; t < o.lag_iters; ++t) {
        for (int j = 0; j < I.n; ++j) {
            const i64* wj = I.w(j);
            double s = 0;
            for (int r = 0; r < I.m; ++r) s += lam[r] * static_cast<double>(wj[r]);
            score[j] = s;
        }
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(),
                         [&](int a, int c) { return score[a] < score[c]; });
        std::fill(x.begin(), x.end(), 0);
        for (int k = 0; k < q; ++k) x[idx[k]] = 1;
        for (int r = 0; r < I.m; ++r) res[r] = -I.b[r];
        for (int k = 0; k < q; ++k) {
            const i64* wj = I.w(idx[k]);
            for (int r = 0; r < I.m; ++r) res[r] += wj[r];
        }
        i64 sq = 0;
        for (i64 v : res) sq += v * v;
        if (sq < best_sq) { best_sq = sq; best = x; }
        double step = o.alpha0 / std::sqrt(static_cast<double>(t) + 1.0);
        double tot = 0;
        for (int r = 0; r < I.m; ++r) {
            lam[r] = std::max(lam[r] + step * static_cast<double>(res[r]), o.lam_floor);
            tot += lam[r];
        }
        for (int r = 0; r < I.m; ++r) lam[r] /= tot;
    }
    return best;
}

static std::vector<char> random_x(int n, int q, std::mt19937_64& rng) {
    std::vector<int> p(n);
    std::iota(p.begin(), p.end(), 0);
    std::shuffle(p.begin(), p.end(), rng);
    std::vector<char> x(n, 0);
    for (int k = 0; k < q; ++k) x[p[k]] = 1;
    return x;
}

// nearest unselected vector to w_i - res, i.e. argmin_j ||res - w_i + w_j||
static int guided_partner(const State& s, int i, i64& out_sq) {
    int best_j = -1;
    i64 best = std::numeric_limits<i64>::max();
    for (int j : s.out) {
        i64 v = s.swap_sq(i, j);
        if (v < best) { best = v; best_j = j; }
    }
    out_sq = best;
    return best_j;
}

// ---------------------------------------------------------------- stage 2
// RGSA / blind SA (Algorithm 2). Energy is the residual norm R, as in the
// Python implementation.  Time mode: T = T0 (Tf/T0)^(elapsed/budget).
struct SAResult { long long steps = 0; double secs = 0; };

static SAResult anneal(State& s, int q, const Options& o, std::mt19937_64& rng,
                       Tracker& tr, double budget_secs) {
    SAResult out;
    auto start = Clock::now();
    std::uniform_real_distribution<double> U(0.0, 1.0);
    const bool timed = budget_secs > 0;
    const double ratio = o.Tf / o.T0;
    const double cool = std::pow(ratio, 1.0 / std::max<long long>(1, o.steps));
    double T = o.T0;
    tr.offer(s, q);
    if (s.in.empty() || s.out.empty()) return out;
    double R = norm_of(s.sq);
    std::vector<char> lbest = s.sel;   // best partition at this cardinality
    i64 lbest_sq = s.sq;
    for (long long k = 0;; ++k) {
        if (!timed && k >= o.steps) break;
        if (timed && (k & 255) == 0) {
            double el = std::chrono::duration<double>(Clock::now() - start).count();
            if (el >= budget_secs || tr.out_of_time()) break;
            T = o.T0 * std::pow(ratio, el / budget_secs);
            tr.tick(tr.elapsed());
        }
        int i = s.in[static_cast<size_t>(rng() % s.in.size())];
        int j;
        i64 cand_sq;
        if (o.p_guide > 0 && U(rng) < o.p_guide) {
            j = guided_partner(s, i, cand_sq);
        } else {
            j = s.out[static_cast<size_t>(rng() % s.out.size())];
            cand_sq = s.swap_sq(i, j);
        }
        double Rc = norm_of(cand_sq);
        double dE = Rc - R;
        if (dE < 0 || U(rng) < std::exp(-dE / std::max(T, o.Tfloor))) {
            s.apply_swap(i, j);
            R = Rc;
            if (s.sq < lbest_sq) { lbest_sq = s.sq; lbest = s.sel; tr.offer(s, q); }
            if (s.sq == 0) { out.steps = k + 1; break; }  // exact solution
        }
        if (!timed) T *= cool;
        out.steps = k + 1;
    }
    if (lbest_sq < s.sq) s.init(*s.I, lbest);  // return the best annealed partition
    out.secs = std::chrono::duration<double>(Clock::now() - start).count();
    return out;
}

// ---------------------------------------------------------------- stage 3
// Legacy polish: exactly the Python rule (one first-improvement 1-swap sweep,
// then up to `passes` best-improvement 2-swap passes).
// Converge polish: alternate full best-improvement 1-swap and 2-swap descent
// until neither improves, so the result is a local optimum for both
// neighbourhoods.
static bool best_two_swap(State& s, double tol, const Tracker* tr = nullptr) {
    const Instance& I = *s.I;
    const int m = I.m;
    std::vector<int> A(s.in), B(s.out);
    std::sort(A.begin(), A.end());
    std::sort(B.begin(), B.end());
    if (A.size() < 2 || B.size() < 2) return false;
    // pair sums of B
    std::vector<std::pair<int, int>> bp;
    std::vector<i64> bsum;
    for (size_t x = 0; x < B.size(); ++x)
        for (size_t y = x + 1; y < B.size(); ++y) {
            bp.emplace_back(B[x], B[y]);
            const i64* u = I.w(B[x]);
            const i64* v = I.w(B[y]);
            for (int r = 0; r < m; ++r) bsum.push_back(u[r] + v[r]);
        }
    double best = norm_of(s.sq);
    int ba1 = -1, ba2 = -1, bb = -1;
    std::vector<i64> base(m);
    for (size_t x = 0; x < A.size(); ++x) {
        if (tr && tr->out_of_time()) return false;  // hard budget: abandon sweep
        for (size_t y = x + 1; y < A.size(); ++y) {
            const i64* u = I.w(A[x]);
            const i64* v = I.w(A[y]);
            for (int r = 0; r < m; ++r) base[r] = s.res[r] - u[r] - v[r];
            i64 local = std::numeric_limits<i64>::max();
            int lk = -1;
            for (size_t k = 0; k < bp.size(); ++k) {
                const i64* ps = &bsum[k * m];
                i64 t = 0;
                for (int r = 0; r < m; ++r) { i64 d = base[r] + ps[r]; t += d * d; }
                if (t < local) { local = t; lk = static_cast<int>(k); }
            }
            double c = norm_of(local);
            if (c < best - tol) { best = c; ba1 = A[x]; ba2 = A[y]; bb = lk; }
        }
    }
    if (ba1 < 0) return false;
    s.apply_swap(ba1, bp[bb].first);
    s.apply_swap(ba2, bp[bb].second);
    return true;
}

static void polish_legacy(State& s, const Options& o) {
    // one sweep, first improvement per removed item (Python semantics)
    std::vector<int> A(s.in);
    std::sort(A.begin(), A.end());
    for (int i : A) {
        if (!s.sel[i]) continue;
        std::vector<int> B(s.out);
        std::sort(B.begin(), B.end());
        double R = norm_of(s.sq);
        for (int j : B) {
            if (s.sel[j]) continue;
            if (norm_of(s.swap_sq(i, j)) < R - o.tol) { s.apply_swap(i, j); break; }
        }
    }
    for (int p = 0; p < o.polish_passes; ++p)
        if (!best_two_swap(s, o.tol)) break;
}

static bool best_one_swap(State& s, double tol) {
    double R = norm_of(s.sq);
    i64 best = std::numeric_limits<i64>::max();
    int bi = -1, bj = -1;
    for (int i : s.in) {
        i64 v; int j = guided_partner(s, i, v);
        if (j >= 0 && v < best) { best = v; bi = i; bj = j; }
    }
    if (bi < 0 || !(norm_of(best) < R - tol)) return false;
    s.apply_swap(bi, bj);
    return true;
}

// Returns the number of improving moves; sets `complete` to true when the
// result is a verified local optimum for both neighbourhoods (i.e. polishing
// was not cut short by the time limit).
static int polish_converge(State& s, const Options& o, Tracker& tr, int q, bool& complete) {
    int rounds = 0;
    complete = false;
    for (;;) {
        while (!tr.out_of_time() && best_one_swap(s, o.tol)) { ++rounds; tr.offer(s, q); }
        if (s.sq == 0) { complete = true; break; }
        if (tr.out_of_time()) break;
        if (!best_two_swap(s, o.tol, &tr)) { complete = !tr.out_of_time(); break; }
        ++rounds;
        tr.offer(s, q);
    }
    return rounds;
}

// ---------------------------------------------------------------- driver
static std::string json_num(double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.10g", v);
    return buf;
}

int main(int argc, char** argv) {
    Options o;
    try {
        o = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        usage();
        return 2;
    }
    Instance I;
    try {
        I = read_instance(o.instance);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
    const int q0 = o.q >= 0 ? o.q : I.n / 2;
    std::vector<int> qs;
    qs.push_back(q0);  // centre first, then alternate outwards
    for (int d = 1; d <= o.band; ++d) {
        if (q0 - d >= 1) qs.push_back(q0 - d);
        if (q0 + d <= I.n - 1) qs.push_back(q0 + d);
    }

    std::mt19937_64 rng(o.seed);
    Tracker tr;
    tr.t0 = Clock::now();
    tr.limit = o.time_limit;
    tr.cps = o.checkpoints;
    tr.cp_sq.assign(tr.cps.size(), -1);

    double stage1_secs = 0, stage2_secs = 0, stage3_secs = 0;
    long long total_steps = 0, restarts = 0;
    int polish_rounds = 0, polish_incomplete = 0;
    std::vector<std::string> per_q;

    if (o.mode == "sa") {
        for (size_t qi = 0; qi < qs.size(); ++qi) {
            int q = qs[qi];
            if (tr.out_of_time()) break;
            // budget for this cardinality: equal share of the remaining time
            double share = 0;
            if (o.time_limit > 0) {
                double remain = o.time_limit - tr.elapsed();
                share = remain / static_cast<double>(qs.size() - qi);
            }
            auto t1 = Clock::now();
            std::vector<char> x0;
            if (o.init == "given") {
                if (static_cast<int>(o.init_x.size()) != I.n)
                    throw std::runtime_error("--init-x length must equal N");
                x0.resize(I.n);
                for (int j = 0; j < I.n; ++j) x0[j] = o.init_x[j] == '1';
            } else if (o.init == "lagrangian") {
                x0 = lagrangian(I, q, o);
            } else {
                x0 = random_x(I.n, q, rng);
            }
            double s1 = std::chrono::duration<double>(Clock::now() - t1).count();
            stage1_secs += s1;
            State s;
            s.init(I, x0);
            i64 after1 = s.sq;
            tr.offer(s, q);

            // anneal.  In time mode, when polishing is on, keep 10% of this
            // cardinality's share for polishing so the whole pipeline respects
            // the global budget; polishing then runs to convergence or until
            // the global time limit, whichever comes first.
            double sa_share = (o.polish == "none") ? share : 0.9 * share;
            double sa_budget = o.time_limit > 0 ? std::max(0.0, sa_share - s1) : 0.0;
            SAResult sr;
            if (o.time_limit > 0 || o.steps > 0) {
                if (o.time_limit > 0 && sa_budget <= 0) {
                    // no time left for annealing at this cardinality
                } else {
                    sr = anneal(s, q, o, rng, tr, sa_budget);
                }
            }
            stage2_secs += sr.secs;
            total_steps += sr.steps;
            // continue from the best state found at this cardinality
            // (Python returns the best annealed partition, not the last one)
            const i64 after2 = s.sq;  // anneal() leaves s at its best partition
            auto t3 = Clock::now();
            if (o.polish == "legacy") polish_legacy(s, o), tr.offer(s, q);
            else if (o.polish == "converge") {
                bool complete = false;
                polish_rounds += polish_converge(s, o, tr, q, complete);
                polish_incomplete += complete ? 0 : 1;
            }
            stage3_secs += std::chrono::duration<double>(Clock::now() - t3).count();
            std::ostringstream js;
            js << "{\"q\":" << q << ",\"after_init\":" << json_num(norm_of(after1))
               << ",\"after_sa\":" << json_num(norm_of(after2))
               << ",\"after_polish\":" << json_num(norm_of(s.sq))
               << ",\"sa_steps\":" << sr.steps << "}";
            per_q.push_back(js.str());
            if (tr.best_sq == 0) break;
        }
    } else {
        // multi-start greedy exchange: random start at a cardinality from the
        // band (round robin), then best-improvement 1-swap descent using the
        // guided partner rule, until no improving swap exists; repeat.
        if (o.time_limit <= 0) {
            std::fprintf(stderr, "error: --mode greedy requires --time\n");
            return 2;
        }
        size_t qi = 0;
        while (!tr.out_of_time()) {
            int q = qs[qi++ % qs.size()];
            State s;
            s.init(I, random_x(I.n, q, rng));
            tr.offer(s, q);
            while (!tr.out_of_time() && best_one_swap(s, 0.0)) {
                ++total_steps;
                tr.offer(s, q);
                tr.tick(tr.elapsed());
            }
            ++restarts;
            if (tr.best_sq == 0) break;
        }
    }

    double total = tr.elapsed();
    tr.tick(std::numeric_limits<double>::infinity());

    // verify the reported vector from scratch
    State v;
    v.init(I, tr.best_x);
    i64 linf = 0;
    for (i64 r : v.res) linf = std::max<i64>(linf, r < 0 ? -r : r);
    int card = 0;
    for (char c : tr.best_x) card += c;
    if (v.sq != tr.best_sq) {
        std::fprintf(stderr, "internal error: residual mismatch\n");
        return 3;
    }

    std::ostringstream out;
    out << "{\"instance\":\"" << o.instance << "\",\"m\":" << I.m << ",\"n\":" << I.n
        << ",\"mode\":\"" << o.mode << "\",\"init\":\"" << o.init << "\",\"seed\":" << o.seed
        << ",\"p_guide\":" << json_num(o.p_guide) << ",\"T0\":" << json_num(o.T0)
        << ",\"Tf\":" << json_num(o.Tf) << ",\"band\":" << o.band
        << ",\"steps_per_q\":" << o.steps << ",\"time_limit\":" << json_num(o.time_limit)
        << ",\"polish\":\"" << o.polish << "\""
        << ",\"residual\":" << json_num(norm_of(v.sq)) << ",\"squared_residual\":" << v.sq
        << ",\"max_abs_violation\":" << linf << ",\"feasible\":" << (v.sq == 0 ? "true" : "false")
        << ",\"cardinality\":" << card << ",\"time_to_best\":" << json_num(tr.time_to_best)
        << ",\"total_time\":" << json_num(total) << ",\"stage1_time\":" << json_num(stage1_secs)
        << ",\"stage2_time\":" << json_num(stage2_secs) << ",\"stage3_time\":" << json_num(stage3_secs)
        << ",\"sa_steps\":" << total_steps << ",\"restarts\":" << restarts
        << ",\"polish_rounds\":" << polish_rounds
        << ",\"polish_incomplete\":" << polish_incomplete << ",\"per_q\":[";
    for (size_t k = 0; k < per_q.size(); ++k) out << (k ? "," : "") << per_q[k];
    out << "],\"checkpoints\":[";
    for (size_t k = 0; k < tr.cps.size(); ++k) {
        out << (k ? "," : "") << "{\"t\":" << json_num(tr.cps[k]) << ",\"residual\":";
        if (tr.cp_sq[k] < 0) out << "null"; else out << json_num(norm_of(tr.cp_sq[k]));
        out << "}";
    }
    out << "]";
    if (o.print_x) {
        out << ",\"x\":\"";
        for (char c : tr.best_x) out << (c ? '1' : '0');
        out << "\"";
    }
    out << "}\n";
    std::fputs(out.str().c_str(), stdout);
    return 0;
}
