#include "isomorphism_examples.hpp"
#include <grlina/modules.hpp>
#include <iostream>
#include <set>

using namespace iso_test;

// Independent bitmask linear algebra for the tiny exhaustive oracle.
unsigned span(const std::vector<unsigned>& columns) {
    unsigned members=1; // bit v denotes membership of the vector v
    for (unsigned c:columns) {
        unsigned old=members;
        for (unsigned v=0;v<8;++v) if (old&(1u<<v)) members|=1u<<(v^c);
    }
    return members;
}
unsigned vector_mask(const vec<int>& c) { unsigned v=0; for (int r:c) v^=1u<<r; return v; }
bool brute_iso(Mat a, Mat b) {
    minimize(a); minimize(b);
    if (a.row_degrees!=b.row_degrees) return false;
    int n=a.get_num_rows(); require(n<=3,"Oracle restricted to at most three generators");
    std::vector<std::pair<int,int>> positions;
    for (int c=0;c<n;++c) for (int r=0;r<n;++r)
        if (Degree_traits<r2degree>::smaller_equal(b.row_degrees[r],a.row_degrees[c])) positions.push_back({c,r});
    auto degrees=a.col_degrees; degrees.insert(degrees.end(),b.col_degrees.begin(),b.col_degrees.end());
    for (unsigned mask=0;mask<(1u<<positions.size());++mask) {
        std::vector<unsigned> f(n);
        for (unsigned i=0;i<positions.size();++i) if (mask&(1u<<i)) f[positions[i].first]^=1u<<positions[i].second;
        if (span(f)!=(1u<<(1u<<n))-1) continue; // invertible on F0
        bool equal=true;
        for (auto d:degrees) {
            std::vector<unsigned> left,right;
            for (int c=0;c<a.get_num_cols();++c) if (Degree_traits<r2degree>::smaller_equal(a.col_degrees[c],d)) {
                unsigned v=0; for (int r:a.data[c]) v^=f[r]; left.push_back(v);
            }
            for (int c=0;c<b.get_num_cols();++c) if (Degree_traits<r2degree>::smaller_equal(b.col_degrees[c],d))
                right.push_back(vector_mask(b.data[c]));
            if (span(left)!=span(right)) { equal=false; break; }
        }
        if (equal) return true;
    }
    return false;
}

void check_pair(const Mat& a, const Mat& b, bool expected) {
    const Mat before_a=a,before_b=b;
    for (auto method:{IsomorphismHomMethod::optimised,IsomorphismHomMethod::full_restriction}) {
        require(is_isomorphic(a,b,false,method)==expected,"Wrong isomorphism answer");
        require(is_isomorphic(b,a,false,method)==expected,"Wrong reverse isomorphism answer");
        Mat am=a,bm=b; minimize(am); minimize(bm);
        require(is_isomorphic(am,bm,true,method)==expected,"Minimal-input answer differs");
    }
    require(same(a,before_a) && same(b,before_b),"Inputs mutated");
}

void small_oracle_tests() {
    RNG rng(84319);
    int positive=0,negative=0;
    for (int trial=0;trial<100;++trial) {
        auto make=[&] {
            array<int> data(3);
            for (auto& c:data) {
                unsigned bits=1+rng()%7;
                for (int r=0;r<3;++r) if (bits&(1u<<r)) c.push_back(r);
            }
            const vec<r2degree> rows = trial%3==0 ? vec<r2degree>{{0,0},{0,0},{0,0}}
                : trial%3==1 ? vec<r2degree>{{0,0},{1,0},{0,1}}
                            : vec<r2degree>{{0,0},{0,0},{1,0}};
            return Mat(3,3,data,{{2,1},{1,2},{2,2}},rows);
        };
        Mat a=make(),b=make();
        if (trial%4==0) { b=a; scramble(b,rng,100); }
        bool expected=brute_iso(a,b);
        if (expected) ++positive; else ++negative;
        check_pair(a,b,expected);
    }
    require(positive>10 && negative>10,"Oracle needs both positive and negative cases");
    std::cout << "Tiny exhaustive oracle: " << positive << " positive, " << negative << " negative pairs\n";
}

void local_block_tests() {
    // Enumerate ALL 67 subspaces of M_2(F2), not just matrices containing I.
    std::set<unsigned> spaces{1};
    for (auto it=spaces.begin();it!=spaces.end();++it)
        for (unsigned v=1;v<16;++v) {
            unsigned next=*it;
            for (unsigned u=0;u<16;++u) if (*it&(1u<<u)) next|=1u<<(u^v);
            spaces.insert(next);
        }
    require(spaces.size()==67,"Incorrect enumeration of 2x2 matrix spaces");
    for (unsigned members:spaces) {
        SparseMatrix<int> basis(0,4);
        bool expected=false;
        for (unsigned v=1;v<16;++v) if (members&(1u<<v)) {
            vec<int> c; for (int i=0;i<4;++i) if (v&(1u<<i)) c.push_back(i);
            isomorphism_detail::extend_span(basis,c);
            expected|=(((v&1)*((v>>3)&1)) ^ (((v>>1)&1)*((v>>2)&1)))!=0;
        }
        require(isomorphism_detail::contains_invertible(basis,2)==expected,"Wrong GL(2) block answer");
    }
    // k >= 3: only the nonidentity cyclic permutation is invertible in its
    // one-dimensional span; also test a nonzero all-singular block space.
    for (int k:{3,4,8}) {
        SparseMatrix<int> unit(0,k*k), singular(0,k*k);
        vec<int> cycle;
        for (int c=0;c<k;++c) cycle.push_back(c*k+(c+1)%k);
        isomorphism_detail::extend_span(unit,cycle);
        for (int c=0;c<k;++c) isomorphism_detail::extend_span(singular,vec<int>{c*k});
        require(isomorphism_detail::contains_invertible(unit,k),"Missed nonidentity invertible block");
        require(!isomorphism_detail::contains_invertible(singular,k),"Singular space accepted");
    }
}

void graded_examples() {
    RNG rng(21981);
    Mat zero(0,0,{}, {}, {}), free(0,1,{}, {}, {{0,0}});
    check_pair(zero,zero,true); check_pair(zero,free,false);
    Mat killed(1,1,{{0}},{{0,0}},{{0,0}});
    check_pair(zero,killed,true);
    Mat a(1,1,{{0}},{{1,1}},{{0,0}});
    Mat redundant(3,2,{{0},{0},{1}},{{1,1},{2,2},{3,3}},{{0,0},{3,3}});
    check_pair(a,redundant,true); // redundant relation + cancellable generator/relation
    for (int k:{1,2,3,4,8}) {
        auto pair=negative_pair(2*k,k,rng);
        Module<Mat> ma(pair.first),mb(pair.second);
        const vec<double> xs{-5,-4,-3,-2,-1}, ys{0,4.0*(2*k)+10,4.0*(2*k)+11,4.0*(2*k)+12};
        require(ma.hilbert_function_on_grid(xs,ys).values==mb.hilbert_function_on_grid(xs,ys).values,
                "Negative control must have the same Hilbert function");
        auto original=pair.first;
        scramble(pair.second,rng,200);
        check_pair(pair.first,pair.second,false);
        scramble(pair.first,rng,200);
        check_pair(original,pair.first,true);
        if (k>4) continue; // Large repeated blocks are timed separately with process deadlines.
        for (int seed=0;seed<3;++seed) {
            Mat random=sparse_random(4*k,k,rng), copy=random;
            auto changes=scramble(copy,rng,20*(copy.get_num_rows()+copy.get_num_cols()));
            require(changes.rows+changes.columns>0,"No random operations applied");
            check_pair(random,copy,true);
        }
    }
}

int main() {
    try { local_block_tests(); small_oracle_tests(); graded_examples(); }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
    std::cout << "Isomorphism regression tests passed\n";
}
