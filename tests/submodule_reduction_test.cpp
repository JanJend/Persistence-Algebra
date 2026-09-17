#include <grlina/modules.hpp>
#include <cassert>

using namespace graded_linalg;
using Mat = R2GradedSparseMatrix<int>;

struct TrackedMatrix : GradedSparseMatrix<r2degree, int, TrackedMatrix> {
    using Base = GradedSparseMatrix<r2degree, int, TrackedMatrix>;
    using Base::Base;
    explicit TrackedMatrix(SparseMatrix<int>&& matrix) : Base(std::move(matrix)) {}
    inline static vec<int> kernel_input_sizes;
    inline static int row_deletions = 0, column_deletions = 0, sorting_refreshes = 0;
    TrackedMatrix graded_kernel() {
        kernel_input_sizes.push_back(this->get_num_cols());
        Mat source(this->get_num_cols(), this->get_num_rows(), this->data,
                   this->col_degrees, this->row_degrees);
        auto K = source.graded_kernel();
        return TrackedMatrix(K.get_num_cols(), K.get_num_rows(), K.data, K.col_degrees, K.row_degrees);
    }
    bool refresh_compatible_sorted() { ++sorting_refreshes; return Base::refresh_compatible_sorted(); }
    void delete_rows(vec<int>& rows) { ++row_deletions; Base::delete_rows(rows); }
    void delete_columns(vec<int>& columns) { ++column_deletions; Base::delete_columns(columns); }
    static void reset_counts() { kernel_input_sizes.clear(); row_deletions = column_deletions = sorting_refreshes = 0; }
};

void test_lazy_reduction() {
    // Fixed ambient row order is deliberately not degree-sorted.
    Mat P(3, 3, {{0,2}, {0,1}, {0}}, {{1,0}, {0,1}, {1,1}},
          {{0,0}, {-1,-1}, {-2,-2}});
    auto parent = std::make_shared<Module<Mat>>(P);
    Mat G(5, 3, {{1,2}, {2}, {2}, {0,2}, {}},
          {{1,1}, {1,0}, {0,1}, {0,0}, {-1,-1}}, P.row_degrees);
    Submodule<Mat> S(parent, G);
    S.reduce_generators_lazy();
    // First column reduces twice to zero. Second becomes e0 but cannot use
    // the relation in degree (1,1). Third cannot use the incomparable (1,0).
    // Fourth cannot use any relation, and the initially zero column is removed.
    assert(S.generator_map().generator_lift().data == array<int>({{0}, {2}, {0,2}}));
    assert(S.generator_map().generator_lift().col_degrees == vec<r2degree>({{1,0}, {0,1}, {0,0}}));
    assert(S.generator_map().generator_lift().row_degrees == P.row_degrees);
    assert(!S.generator_map().generator_lift().compatibly_sorted);
    assert(parent->presentation().data == P.data && parent->presentation().col_degrees == P.col_degrees);
    const auto reduced = S.generator_map().generator_lift();
    S.reduce_generators_lazy();
    assert(S.generator_map().generator_lift().data == reduced.data); // idempotent, no reordering

    // With several relations at one pivot, do not stop at an inadmissible one.
    Mat bucket(2, 1, {{0}, {0}}, {{2,0}, {0,1}}, {{0,0}});
    auto bucket_parent = std::make_shared<Module<Mat>>(bucket);
    Submodule<Mat> killed(bucket_parent, Mat(1, 1, {{0}}, {{0,1}}, bucket.row_degrees));
    killed.reduce_generators_lazy();
    assert(killed.number_of_generators() == 0);
}

void test_lazy_is_not_exact() {
    // a and b have pivot 2, while c=a+b has pivot 1. Lazy relation lookup
    // intentionally does not construct the combination needed to kill c.
    Mat P(2, 3, {{0,2}, {1,2}}, {{0,1}, {1,0}}, {{0,0}, {0,0}, {0,0}});
    auto parent = std::make_shared<Module<Mat>>(P);
    Mat G(1, 3, {{0,1}}, {{1,1}}, P.row_degrees);
    Submodule<Mat> S(parent, G), direct(parent, G);
    S.reduce_generators_lazy();
    assert(S.generator_map().generator_lift().data == G.data);
    S.minimize_generators();
    direct.minimize_generators(false);
    assert(S.number_of_generators() == 0 && direct.number_of_generators() == 0);
}

void test_preprocessing_and_deferred_deletion() {
    using T = TrackedMatrix;
    auto parent = std::make_shared<Module<T>>(T(1, 2, {{0}}, {{1,1}}, {{0,0}, {0,0}}));
    T G(3, 2, {{0}, {1}, {0,1}}, {{1,1}, {0,0}, {1,1}}, parent->presentation().row_degrees);
    Submodule<T> preprocessed(parent, G), direct(parent, G);
    T::reset_counts();
    preprocessed.minimize_generators();
    assert(T::kernel_input_sizes == vec<int>({3})); // one relation + two surviving generators
    assert(T::row_deletions == 0 && T::column_deletions == 2); // one batch per phase
    T::reset_counts();
    direct.minimize_generators(false);
    assert(T::kernel_input_sizes == vec<int>({4})); // opt out: all original columns
    assert(T::row_deletions == 0 && T::column_deletions == 1);
    assert(preprocessed.generator_map().generator_lift().data == array<int>({{1}}));
    assert(preprocessed.generator_map().generator_lift().col_degrees == vec<r2degree>({{0,0}}));
    assert(direct.generator_map().generator_lift().data == preprocessed.generator_map().generator_lift().data);
    assert(direct.generator_map().generator_lift().col_degrees == preprocessed.generator_map().generator_lift().col_degrees);
    // Deletion can leave sorted lists, but we do not rescan to discover that.
    assert(!preprocessed.generator_map().generator_lift().compatibly_sorted);

    // If preprocessing kills everything, no kernel is needed at all.
    Submodule<T> zero(parent, T(2, 2, {{0}, {}}, {{1,1}, {0,0}}, parent->presentation().row_degrees));
    T::reset_counts();
    zero.minimize_generators();
    assert(zero.number_of_generators() == 0 && T::kernel_input_sizes.empty());

    // Many exact eliminations, but only one physical generator compaction.
    auto free = std::make_shared<Module<T>>(T(0, 1, {}, {}, {{0,0}}));
    T duplicates(128, 1, array<int>(128, vec<int>{0}), vec<r2degree>(128, {0,0}), {{0,0}});
    Submodule<T> redundant(free, duplicates);
    T::reset_counts();
    redundant.minimize_generators(false);
    assert(redundant.generator_map().generator_lift().data == array<int>({{0}}));
    assert(T::row_deletions == 0 && T::column_deletions == 1);
    assert(T::kernel_input_sizes == vec<int>({128}));

    auto zero_parent = std::make_shared<Module<T>>(T(0, 0, {}, {}, {}));
    auto empty = Submodule<T>::zero(zero_parent);
    T::reset_counts();
    empty.minimize_generators();
    assert(empty.number_of_generators() == 0 && T::kernel_input_sizes.empty());
}

void test_no_kernel_required_for_lazy() {
    using Higher = R4GradedSparseMatrix<int>;
    Higher P(1, 1, {{0}}, {r4degree(1,0,0,0)}, {r4degree(0,0,0,0)});
    auto parent = std::make_shared<Module<Higher>>(P);
    Higher G(2, 1, {{0}, {0}}, {r4degree(0,0,0,0), r4degree(1,0,0,0)}, P.row_degrees);
    Submodule<Higher> S(parent, G);
    S.reduce_generators_lazy();
    assert(S.generator_map().generator_lift().data == array<int>({{0}}));
    assert(S.generator_map().generator_lift().col_degrees == vec<r4degree>({r4degree(0,0,0,0)}));
}

void test_identity_paths_do_not_refresh_sorting() {
    using T = TrackedMatrix;
    using M = Module<T>;
    using H = Homomorphism<T>;
    auto parent = std::make_shared<const M>(T(0, 2, {}, {}, {{0,0}, {1,1}}));
    auto eta = H::canonical_shift(parent, {1,1});
    auto next = H::canonical_shift(eta.target(), {1,1});
    H ordinary(eta.domain(), eta.target(), eta.lifts());
    H ordinary_next(next.domain(), next.target(), next.lifts());
    auto whole = Submodule<T>::whole(parent);
    T unknown(2, 2, "Identity");
    unknown.col_degrees = unknown.row_degrees = parent->presentation().row_degrees;
    T::reset_counts();
    Submodule<T> supplied(parent, std::move(unknown));
    auto image = eta.image(whole, false);
    auto image_unknown = eta.image(supplied, false);
    auto both = eta.compose(next);
    auto left = eta.compose(ordinary_next);
    auto right = ordinary.compose(next);
    auto zero = H::zero(parent, next.target());
    assert(T::sorting_refreshes == 0);
    assert(image.generator_map().generator_lift().compatibly_sorted);
    assert(!supplied.generator_map().generator_lift().compatibly_sorted);
    assert(!image_unknown.generator_map().generator_lift().compatibly_sorted);
    assert(both.generator_lift().compatibly_sorted);
    assert(left.generator_lift().compatibly_sorted && right.generator_lift().compatibly_sorted);
    assert(image.generator_map().generator_lift().data == whole.generator_map().generator_lift().data);
    assert(image.generator_map().generator_lift().row_degrees == eta.target()->presentation().row_degrees);
}

int main() {
    test_identity_paths_do_not_refresh_sorting();
    test_lazy_reduction();
    test_lazy_is_not_exact();
    test_preprocessing_and_deferred_deletion();
    test_no_kernel_required_for_lazy();
}
