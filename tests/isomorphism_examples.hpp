#pragma once
#include <grlina/isomorphism_test.hpp>
#include <cmath>
#include <map>
#include <numeric>
#include <random>

namespace iso_test {
using namespace graded_linalg;
using Mat = R2GradedSparseMatrix<int>;
using RNG = std::mt19937_64;
inline void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
inline bool same(const Mat& a, const Mat& b) {
    return a.data == b.data && a.row_degrees == b.row_degrees && a.col_degrees == b.col_degrees;
}
inline void minimize(Mat& a) { a.sort_compatibly(); a.minimize_variant(); }
inline std::size_t nnz(const Mat& a) {
    std::size_t n=0; for (const auto& c:a.data) n+=c.size(); return n;
}
inline int multiplicity(const Mat& a) {
    std::map<r2degree,int> counts;
    int k=0; for (auto d:a.row_degrees) k=std::max(k,++counts[d]); return k;
}

// O(n) nonzeros: up to three per relation, two relations per generator.
// Exactly k generators per distinct grade; strict relation grades prevent
// cancellation of generators. Redundant relations are removed before timing.
inline Mat sparse_random(int n, int k, RNG& rng) {
    require(n >= 0 && k > 0 && n%k == 0,"n must be a nonnegative multiple of k");
    if (!n) return Mat(0,0,{}, {}, {});
    int side=static_cast<int>(std::ceil(std::sqrt(2.0*n/k)));
    vec<r2degree> pool, rows, cols;
    for (int x=0;x<side;++x) for (int y=0;y<side;++y) pool.push_back({2.0*x,2.0*y});
    std::shuffle(pool.begin(),pool.end(),rng);
    for (int b=0;b<n/k;++b) for (int j=0;j<k;++j) rows.push_back(pool[b]);
    array<int> data;
    for (int j=0;j<2*n;++j) {
        int anchor=rng()%n;
        vec<int> candidates;
        for (int i=0;i<n;++i) if (i!=anchor && Degree_traits<r2degree>::smaller_equal(rows[i],rows[anchor]))
            candidates.push_back(i);
        std::shuffle(candidates.begin(),candidates.end(),rng);
        candidates.resize(std::min<std::size_t>(2,candidates.size()));
        candidates.push_back(anchor);
        std::sort(candidates.begin(),candidates.end());
        data.push_back(std::move(candidates));
        cols.push_back({rows[anchor].first+1+double(rng()%3), rows[anchor].second+1+double(rng()%3)});
    }
    Mat a(2*n,n,data,cols,rows);
    minimize(a);
    require(a.get_num_rows()==n && multiplicity(a)==k,"Random generator multiplicity changed");
    return a;
}

// Same Betti degrees AND Hilbert function, but different ranks from (-4,h)
// to (-2,h): 0 versus k. Other summands are invisible at these negative Xs.
inline std::pair<Mat,Mat> negative_pair(int n, int k, RNG& rng) {
    require(n>=2*k,"Negative pair needs n >= 2*k");
    Mat base=sparse_random(n-2*k,k,rng);
    auto rows=base.row_degrees, cols=base.col_degrees;
    auto ad=base.data, bd=base.data;
    const int offset=base.get_num_rows();
    const double h=4.0*n+10;
    for (int b=0;b<2;++b) for (int i=0;i<k;++i) rows.push_back({-4.0+b,h});
    for (int b=0;b<2;++b) for (int i=0;i<k;++i) {
        cols.push_back(b==0 ? r2degree{-2,h} : r2degree{-3,h+1});
        ad.push_back({offset+b*k+i});
        bd.push_back({offset+(1-b)*k+i});
    }
    Mat a(static_cast<int>(cols.size()),n,ad,cols,rows), b(static_cast<int>(cols.size()),n,bd,cols,rows);
    minimize(a); minimize(b);
    require(a.row_degrees==b.row_degrees && a.col_degrees==b.col_degrees,"Negative control lost matching degrees");
    return {a,b};
}

struct Operation { bool row; int source, target; };
inline void apply(Mat& a, Operation op) {
    if (!op.row) a.col_op(op.source,op.target);
    else for (auto& column:a.data)
        if (std::binary_search(column.begin(),column.end(),op.source))
            Column_traits<vec<int>,int>::add_to(vec<int>{op.target},column);
}
struct Changes { int rows=0, columns=0; };
inline Changes scramble(Mat& a, RNG& rng, int operations) {
    const Mat original=a;
    std::vector<Operation> history;
    Changes count;
    // Limit fill-in so the transformed random inputs also remain sparse.
    std::size_t cap=12;
    for (const auto& c:a.data) cap=std::max(cap,c.size());
    for (int attempt=0; attempt<100*operations && static_cast<int>(history.size())<operations; ++attempt) {
        Operation op{bool(rng()%2),0,0};
        int size=op.row ? a.get_num_rows() : a.get_num_cols();
        if (size<2) continue;
        op.source=rng()%size; op.target=rng()%size;
        if (op.source==op.target) continue;
        bool changes=false, fits=true;
        if (op.row) {
            if (!a.is_admissible_row_operation(op.source,op.target)) continue;
            for (const auto& c:a.data) if (std::binary_search(c.begin(),c.end(),op.source)) {
                changes=true;
                if (c.size()>=cap && !std::binary_search(c.begin(),c.end(),op.target)) { fits=false; break; }
            }
        } else {
            if (!a.is_admissible_column_operation(op.source,op.target) || a.data[op.source].empty()) continue;
            auto c=a.data[op.target]; Column_traits<vec<int>,int>::add_to(a.data[op.source],c);
            changes=true; fits=c.size()<=cap;
        }
        if (!changes || !fits) continue;
        apply(a,op); history.push_back(op);
        if (op.row) ++count.rows; else ++count.columns;
    }
    // Each addition is its own inverse over F2; verify the transformation's
    // certificate independently of is_isomorphic, including its exact entries.
    Mat undo=a;
    for (auto it=history.rbegin();it!=history.rend();++it) apply(undo,*it);
    require(same(undo,original),"Admissible operation inverse certificate failed");
    vec<int> permutation(a.get_num_rows());
    std::iota(permutation.begin(),permutation.end(),0);
    std::shuffle(permutation.begin(),permutation.end(),rng);
    a.permute_rows_graded(permutation);
    permutation.resize(a.get_num_cols());
    std::iota(permutation.begin(),permutation.end(),0);
    std::shuffle(permutation.begin(),permutation.end(),rng);
    auto data=a.data; auto degrees=a.col_degrees;
    for (int j=0;j<a.get_num_cols();++j) { data[j]=a.data[permutation[j]]; degrees[j]=a.col_degrees[permutation[j]]; }
    a=Mat(a.get_num_cols(),a.get_num_rows(),data,degrees,a.row_degrees);
    a.validate();
    return count;
}
} // namespace iso_test
