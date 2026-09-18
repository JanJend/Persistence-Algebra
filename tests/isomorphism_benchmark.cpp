#include "isomorphism_examples.hpp"
#include <grlina/modules.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>

using namespace iso_test;
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }

int main(int argc,char** argv) {
    try {
        if (argc<2) throw std::invalid_argument("Usage: isomorphism_benchmark random N K SEED positive|negative | file PATH SEED | block K");
        std::cout << std::setprecision(12);
        const std::string mode=argv[1];
        if (mode=="block" && argc==3) {
            int k=std::stoi(argv[2]); require(k>=2 && k<=16,"Block size must be 2..16");
            SparseMatrix<int> singular(0,k*k);
            for (int c=0;c<k;++c) for (int r=0;r<k-1;++r)
                isomorphism_detail::extend_span(singular,vec<int>{c*k+r});
            auto start=Clock::now();
            bool result=isomorphism_detail::contains_invertible(singular,k);
            require(!result,"All-singular block returned true");
            std::cout << "{\"k\":" << k << ",\"span_dimension\":" << k*(k-1)
                      << ",\"iso_seconds\":" << seconds(start) << "}\n";
            return 0;
        }
        Mat a,b;
        int seed=0,requested_k=0;
        bool expected=true;
        auto setup=Clock::now();
        if (mode=="random" && argc==6) {
            int n=std::stoi(argv[2]); requested_k=std::stoi(argv[3]); seed=std::stoi(argv[4]);
            require(n>0 && n<=16384 && requested_k>0 && n%requested_k==0,"Invalid n/k");
            RNG rng(seed);
            std::string kind=argv[5]; require(kind=="positive" || kind=="negative","Unknown case kind");
            expected=kind=="positive";
            if (expected) a=sparse_random(n,requested_k,rng),b=a;
            else { auto pair=negative_pair(n,requested_k,rng); a=std::move(pair.first); b=std::move(pair.second); }
        } else if (mode=="file" && argc==4) {
            seed=std::stoi(argv[3]);
            Module<Mat> loaded{std::string(argv[2])};
            a=loaded.presentation(); minimize(a); b=a;
        } else throw std::invalid_argument("Invalid arguments; see usage in source or README");
        RNG rng(seed+937);
        const auto changes=scramble(b,rng,20*(b.get_num_rows()+b.get_num_cols()));
        const double setup_seconds=seconds(setup);
        const Mat before_a=a,before_b=b;
        Mat ac=a,bc=b; ac.sort_compatibly(); bc.sort_compatibly();
        const bool identical=same(ac,bc);
        std::cout << "{\"n\":" << a.get_num_rows() << ",\"relations\":" << a.get_num_cols()
                  << ",\"k\":" << multiplicity(a) << ",\"setup_seconds\":" << setup_seconds
                  << ",\"stage\":\"minimal_isomorphism\"}" << std::endl;
        auto start=Clock::now(); bool result=is_isomorphic(a,b,true);
        const double iso_seconds=seconds(start);
        require(result==expected,"Wrong isomorphism result (minimal path)");
        std::cout << "{\"iso_seconds\":" << iso_seconds << ",\"stage\":\"default_isomorphism\"}" << std::endl;
        start=Clock::now(); bool automatic=is_isomorphic(a,b);
        const double default_seconds=seconds(start);
        require(automatic==expected,"Wrong isomorphism result (default path)");
        require(same(a,before_a) && same(b,before_b),"Inputs mutated");
        int max_column=0; for (const auto& c:b.data) max_column=std::max(max_column,int(c.size()));
        std::cout << "{\"n\":" << a.get_num_rows() << ",\"relations\":" << a.get_num_cols()
                  << ",\"k\":" << multiplicity(a) << ",\"requested_k\":" << requested_k
                  << ",\"seed\":" << seed << ",\"nnz_a\":" << nnz(a) << ",\"nnz_b\":" << nnz(b)
                  << ",\"max_column_b\":" << max_column << ",\"row_ops\":" << changes.rows
                  << ",\"column_ops\":" << changes.columns << ",\"identical\":" << int(identical)
                  << ",\"isomorphic\":" << int(result) << ",\"setup_seconds\":" << setup_seconds
                  << ",\"iso_seconds\":" << iso_seconds << ",\"default_seconds\":" << default_seconds << ",\"stage\":\"complete\"}\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
