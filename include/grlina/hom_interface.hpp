/** @file hom_interface.hpp @brief Matrix and module interfaces for Hom algorithms. */
#pragma once

#include <memory>
#include <random>
#include <vector>

#include <grlina/hom_algorithms.hpp>
#include <grlina/module/submodule.hpp>
#include <grlina/matrices/base.hpp>
#include <grlina/matrices/matrix_algorithms.hpp>

namespace graded_linalg {

/** Basis of degree-zero module homomorphisms, using the selected Hom algorithm.
 * Matrices stay inside the presentation-level implementation; callers receive
 * maps with their domain and target attached.
 */
template <typename Matrix>
std::vector<Homomorphism<Matrix>> module_hom_space_basis(
    std::shared_ptr<const Module<Matrix>> domain,
    std::shared_ptr<const Module<Matrix>> target,
    HomAlgorithm algorithm, bool info = false) {
    GRLINA_DEBUG_CHECK(if (!domain || !target)
        throw std::invalid_argument("Hom basis requires non-null domain and target modules"));
    Matrix source_presentation = domain->presentation();
    source_presentation.compute_rows_forward();
    std::vector<Matrix> generator_lifts =
        hom_space_basis(source_presentation, target->presentation(), algorithm, info);
    std::vector<Homomorphism<Matrix>> homomorphisms;
    for (Matrix& generator_lift : generator_lifts)
        homomorphisms.emplace_back(domain, target, std::move(generator_lift));
    return homomorphisms;
}

template <typename Matrix>
std::vector<Homomorphism<Matrix>> module_hom_space_basis(
    std::shared_ptr<const Module<Matrix>> domain,
    std::shared_ptr<const Module<Matrix>> target,
    bool use_hom_exactness = false, bool info = false) {
    GRLINA_DEBUG_CHECK(if (!domain || !target)
        throw std::invalid_argument("Hom basis requires non-null domain and target modules"));
    Matrix source = domain->presentation();
    Matrix destination = target->presentation();
    source.compute_rows_forward();
    // Module homomorphisms are equivalence classes of lifts, so remove maps that
    // differ by a factorisation through the target relations.
    auto lifts = hom_space_basis_new(source, destination, use_hom_exactness, info, true);
    std::vector<Homomorphism<Matrix>> result;
    result.reserve(lifts.size());
    for (auto& lift : lifts) result.emplace_back(domain, target, std::move(lift));
    return result;
}

template <typename Matrix>
std::vector<Homomorphism<Matrix>> module_endomorphism_basis(
    std::shared_ptr<const Module<Matrix>> module,
    HomAlgorithm algorithm, bool info = false) {
    return module_hom_space_basis(module, module, algorithm, info);
}

template <typename Matrix>
std::vector<Homomorphism<Matrix>> module_endomorphism_basis(
    std::shared_ptr<const Module<Matrix>> module,
    bool use_hom_exactness = false, bool info = false) {
    return module_hom_space_basis(module, module, use_hom_exactness, info);
}

/** Flatten a supplied map family without copying full graded matrices. */
template<class Storage = DefaultSparseStorage<std::ptrdiff_t>, class Matrix>
SparseMatrix<std::ptrdiff_t, Storage> homomorphism_coefficients(
    const std::vector<Homomorphism<Matrix>>& maps) {
    if (maps.empty()) return SparseMatrix<std::ptrdiff_t, Storage>(0, 0);
    const auto& first = maps.front();
    MapCoefficientContext<Matrix> context(first.domain()->presentation(), first.target()->presentation());
    SparseMatrix<std::ptrdiff_t, Storage> result(maps.size(), context.coefficient_rows());
    result.resize_data(maps.size());
    for (std::size_t j = 0; j < maps.size(); ++j) {
        GRLINA_DEBUG_CHECK(if (maps[j].domain() != first.domain() || maps[j].target() != first.target())
            throw std::invalid_argument("Map coefficients require common modules"));
        vec<std::ptrdiff_t> entries;
        const auto& lift = maps[j].generator_lift();
        for (matrix_index_t<Matrix> c = 0; c < lift.get_num_cols(); ++c)
            for (auto r : lift.column(c)) entries.push_back(std::ptrdiff_t(c) * context.rows() + r);
        result.set_col(j, context.normal(entries));
    }
    return result;
}

/** Materialize only the maps a caller actually needs. Common grades and module
 * pointers are attached here, after quotient reduction / generator selection.
 */
template<class Matrix, class Storage>
std::vector<Matrix> graded_maps_from_coefficients(
    const SparseMatrix<std::ptrdiff_t, Storage>& coefficients,
    const Matrix& source, const Matrix& target) {
    const auto rows = target.get_num_rows();
    std::vector<Matrix> result;
    result.reserve(coefficients.get_num_cols());
    for (const auto& entries : coefficients.data) {
        auto lift = empty_matrix_like(target, source.get_num_rows(), rows);
        set_geometric_degrees(lift, geometric_row_degrees(source), geometric_row_degrees(target));
        for (auto entry : entries) lift.append_entry(entry / rows, entry % rows);
        result.push_back(std::move(lift));
    }
    return result;
}

template<class Matrix, class Storage>
std::vector<Homomorphism<Matrix>> homomorphisms_from_coefficients(
    const SparseMatrix<std::ptrdiff_t, Storage>& coefficients,
    std::shared_ptr<const Module<Matrix>> domain, std::shared_ptr<const Module<Matrix>> target) {
    const auto columns = domain->num_of_gens(), rows = target->num_of_gens();
    if (rows && std::size_t(columns) > std::size_t(std::numeric_limits<std::ptrdiff_t>::max() / rows))
        throw std::overflow_error("Flattened map coordinates exceed ptrdiff_t");
    if (coefficients.get_num_rows() != std::ptrdiff_t(columns) * rows)
        throw std::invalid_argument("Map coefficients have incompatible modules");
    std::vector<Homomorphism<Matrix>> result;
    result.reserve(coefficients.get_num_cols());
    auto lifts = graded_maps_from_coefficients(coefficients, domain->presentation(), target->presentation());
    for (auto& lift : lifts) {
        result.emplace_back(domain, target, std::move(lift));
    }
    return result;
}

template<class Coefficients>
struct EndQuotientCoefficients {
    Coefficients quotient;
    Coefficients subspace;
};

/** Q = Hom(X,X_d)/(eta E), retaining the reduced eta E for compositions.
 * eta must be the canonical shift with identity coefficients. E is preserved;
 * only its sparse coefficients are copied into the disposable shifted family.
 */
template<class Matrix, class Storage>
EndQuotientCoefficients<SparseMatrix<std::ptrdiff_t, Storage>> End_d_0_coefficients(
    const Homomorphism<Matrix>& eta, const SparseMatrix<std::ptrdiff_t, Storage>& E,
    HomAlgorithm algorithm = HomAlgorithm::optimised) {
    auto source = eta.domain()->presentation();
    source.compute_rows_forward();
    const auto& target = eta.target()->presentation();
    auto H = hom_space_coefficients<Storage>(source, target, algorithm);
    auto W = E;
    MapCoefficientContext<Matrix>(source, target).normalize(W);
    reduce_quotient_in_place(W, H);
    return {std::move(H), std::move(W)};
}

template<class Storage = DefaultSparseStorage<std::ptrdiff_t>, class Matrix>
EndQuotientCoefficients<SparseMatrix<std::ptrdiff_t, Storage>> End_d_0_coefficients(
    const Homomorphism<Matrix>& eta, HomAlgorithm algorithm = HomAlgorithm::optimised) {
    auto source = eta.domain()->presentation();
    source.compute_rows_forward();
    const auto& target = eta.target()->presentation();
    auto H = hom_space_coefficients<Storage>(source, target, algorithm);
    auto W = algorithm == HomAlgorithm::optimised
        ? hom_space_shift_image_coefficients<Storage>(source, source, target)
        : hom_space_coefficients<Storage>(source, source, algorithm);
    if (algorithm != HomAlgorithm::optimised) MapCoefficientContext<Matrix>(source, target).normalize(W);
    reduce_quotient_in_place(W, H);
    return {std::move(H), std::move(W)};
}

template <typename Matrix>
Matrix canonical_shift_lift(const Matrix& presentation, const typename Matrix::degree_type& amount) {
    return Homomorphism<Matrix>::canonical_shift(
        std::make_shared<const Module<Matrix>>(presentation), amount).generator_lift();
}

/** Representatives of (span(maps) + span(subspace)) / span(subspace).
 * All maps must have the same domain and target. The existing matrix-family
 * reduction identifies zero-inducing lifts before taking the linear quotient.
 */
template <typename Matrix>
std::vector<Homomorphism<Matrix>> homomorphism_quotient_basis(
    const std::vector<Homomorphism<Matrix>>& maps,
    const std::vector<Homomorphism<Matrix>>& subspace) {
    if (maps.empty()) return {};
    GRLINA_DEBUG_CHECK(if (!subspace.empty() && (subspace.front().domain() != maps.front().domain() ||
            subspace.front().target() != maps.front().target()))
        throw std::invalid_argument("Hom quotient requires common domain and target modules"));
    auto H = homomorphism_coefficients(maps);
    auto W = homomorphism_coefficients(subspace);
    if (subspace.empty()) W.set_num_rows(H.get_num_rows());
    reduce_quotient_in_place(W, H);
    return homomorphisms_from_coefficients(H, maps.front().domain(), maps.front().target());
}

/** Express homomorphisms in an already supplied basis of H/W over F2.
 * Write basis = (f_1,...,f_n), subspace = (w_1,...,w_t), and W = span(w_a).
 * The classes [f_j] must be a basis of H/W; each query p_l in maps must lie in H.
 * All homomorphisms must have the same domain and target.
 *
 * Return the n-by-maps.size() matrix C determined by
 *     p_l = sum_j C[j,l] f_j + sum_a D[a,l] w_a
 * as module homomorphisms. The coefficients D are auxiliary and are discarded.
 * They need not be unique, but C is unique because the [f_j] are independent.
 * With an empty subspace this is ordinary coordinate computation in H.
 *
 * Implementation: normalize generator lifts modulo target relations and flatten
 * each whole map to a coefficient vector N(f). Solve the F2 linear system
 *     [N(w_1) ... N(w_t) N(f_1) ... N(f_n)] [D; C] = [N(p_1) ... N(p_m)].
 * Two normalized lift vectors agree exactly when they induce the same module map.
 * This function neither chooses W nor computes a quotient basis: both are
 * inputs. It throws if a query lies outside W + span(f_1,...,f_n).
 */
template <typename Matrix>
SparseMatrix<std::ptrdiff_t> homomorphism_quotient_coordinates(
    const std::vector<Homomorphism<Matrix>>& basis,
    const std::vector<Homomorphism<Matrix>>& maps,
    const std::vector<Homomorphism<Matrix>>& subspace = {}) {
    if (maps.empty()) return SparseMatrix<std::ptrdiff_t>(0, basis.size());
    // The unknown coefficients are ordered as (W coefficients, basis coordinates).
    for (const std::vector<Homomorphism<Matrix>>* family : {&subspace, &basis}) {
        for (const Homomorphism<Matrix>& map : *family) {
            GRLINA_DEBUG_CHECK(if (map.domain() != maps.front().domain() || map.target() != maps.front().target())
                throw std::invalid_argument("Hom coordinates require common domain and target modules"));
        }
    }
    for (const Homomorphism<Matrix>& map : maps) {
        GRLINA_DEBUG_CHECK(if (map.domain() != maps.front().domain() || map.target() != maps.front().target())
            throw std::invalid_argument("Hom coordinates require common domain and target modules"));
    }
    using Storage = vec<vec<std::ptrdiff_t>>;
    auto spanning_coefficients = homomorphism_coefficients<Storage>(subspace);
    const auto basis_coefficients = homomorphism_coefficients<Storage>(basis);
    if (subspace.empty()) spanning_coefficients.set_num_rows(basis_coefficients.get_num_rows());
    if (!basis.empty()) spanning_coefficients.append_matrix(basis_coefficients);
    auto query_coefficients = homomorphism_coefficients<Storage>(maps);
    spanning_coefficients.set_num_rows(query_coefficients.get_num_rows());
    SparseMatrix<std::ptrdiff_t> solution;
    if (!spanning_coefficients.solve_col_reduction(query_coefficients, std::move(solution), true))
        throw std::invalid_argument("Homomorphism lies outside the supplied quotient span");
    SparseMatrix<std::ptrdiff_t> coordinates(0, basis.size());
    // Keep C from the solution (D,C); W was already specified by the caller.
    for (std::ptrdiff_t j = 0; j < solution.get_num_cols(); ++j) {
        std::vector<std::ptrdiff_t> column;
        for (std::ptrdiff_t entry : solution.column(j))
            if (entry >= std::ptrdiff_t(subspace.size()))
                column.push_back(entry - std::ptrdiff_t(subspace.size()));
        coordinates.append_col(std::move(column));
    }
    coordinates.compute_num_cols();
    return coordinates;
}

/** Basis of the space of generator lifts, without identifying lifts differing
 * by target relations. Use module_hom_space_basis for actual Hom classes.
 */
template <typename Matrix>
std::vector<Matrix> homomorphism_lift_basis(const Matrix& domain, const Matrix& target) {
    Matrix source = domain;
    source.compute_rows_forward();
    return hom_space_basis_new(source, target);
}

/** Representatives of End(X)_d / End(X)_0 = Hom(X,X_d) / eta_d End(X).
 * d is the actual shift; this function does not double it. Normalize both map
 * families modulo the relations of X_d before taking their linear quotient.
 * Algorithm A-1/2 computes eta_d End(X) directly by projected elimination;
 * Algorithms A and B retain the two-Hom computation for comparison.
 */
template <typename Matrix>
std::vector<Matrix> End_d_0(
    const Matrix& X, const typename Matrix::degree_type& d,
    bool info = false, HomAlgorithm hom_algorithm = HomAlgorithm::optimised) {
    Matrix X_d = X;
    X_d.shift(d);
    Matrix X_presentation = X;
    X_presentation.compute_rows_forward();
    auto End_d = hom_space_coefficients(X_presentation, X_d, hom_algorithm);
    auto End_0 = hom_algorithm == HomAlgorithm::optimised
        ? hom_space_shift_image_coefficients(X_presentation, X, X_d)
        : hom_space_coefficients(X_presentation, X, hom_algorithm);
    const auto shifted_dimension = End_d.get_num_cols();
    if (hom_algorithm != HomAlgorithm::optimised) MapCoefficientContext<Matrix>(X_presentation, X_d).normalize(End_0);
    reduce_quotient_in_place(End_0, End_d);
    if (info)
        std::cout << "dim eta End_0 = " << End_0.get_num_cols()
                  << ", dim End_d = " << shifted_dimension
                  << ", dim End_d/0 = " << End_d.get_num_cols()
                  << " (module maps; " << hom_algorithm_name(hom_algorithm) << ")" << std::endl;
    return graded_maps_from_coefficients(End_d, X, X_d);
}

/** Compute End(X)_d / End(X)_0, reusing a supplied basis of End(X).
 * Composition algorithms can retain this basis instead of computing it again.
 * Trusted input: eta_d has identity coefficients in the presentation coordinates,
 * as returned by Homomorphism<Matrix>::canonical_shift.
 * Results retain the supplied X and X_d objects for later submodule operations.
 */
template <typename Matrix>
std::vector<Homomorphism<Matrix>> End_d_0(
    const Homomorphism<Matrix>& eta_d,
    const std::vector<Homomorphism<Matrix>>& endomorphism_basis, bool info = false,
    HomAlgorithm hom_algorithm = HomAlgorithm::optimised) {
    GRLINA_DEBUG_CHECK(if (!eta_d.domain() || !eta_d.target())
        throw std::invalid_argument("End_d_0 requires X and X_d"));
    auto E = homomorphism_coefficients(endomorphism_basis);
    if (endomorphism_basis.empty()) {
        const MapCoefficientContext<Matrix> context(eta_d.domain()->presentation(), eta_d.target()->presentation());
        E.set_num_rows(context.coefficient_rows());
    }
    auto result = End_d_0_coefficients(eta_d, E, hom_algorithm);
    if (info)
        std::cout << "dim End_0 = " << endomorphism_basis.size()
                  << ", dim End_d = " << result.quotient.get_num_cols() + result.subspace.get_num_cols()
                  << ", dim End_d/0 = " << result.quotient.get_num_cols()
                  << " (module maps; " << hom_algorithm_name(hom_algorithm) << ")" << std::endl;
    return homomorphisms_from_coefficients(result.quotient, eta_d.domain(), eta_d.target());
}

/** Compute the same quotient, directly computing eta_d End(X) with A-1/2. */
template <typename Matrix>
std::vector<Homomorphism<Matrix>> End_d_0(
    const Homomorphism<Matrix>& eta_d, bool info = false,
    HomAlgorithm hom_algorithm = HomAlgorithm::optimised) {
    GRLINA_DEBUG_CHECK(if (!eta_d.domain() || !eta_d.target())
        throw std::invalid_argument("End_d_0 requires X and X_d"));
    auto result = End_d_0_coefficients(eta_d, hom_algorithm);
    if (info)
        std::cout << "dim eta End_0 = " << result.subspace.get_num_cols()
                  << ", dim End_d = " << result.quotient.get_num_cols() + result.subspace.get_num_cols()
                  << ", dim End_d/0 = " << result.quotient.get_num_cols()
                  << " (module maps; " << hom_algorithm_name(hom_algorithm) << ")" << std::endl;
    return homomorphisms_from_coefficients(result.quotient, eta_d.domain(), eta_d.target());
}

/** Build coordinate matrices for composition by E = End_0(X) on V = H/W.
 * Here W is a subspace of H, and H is a subspace of Hom_0(X,X(delta)).
 * Both must be invariant under the chosen action. The caller supplies
 * module_basis = (f_1,...,f_n), whose classes form a basis of V,
 * endomorphism_basis = (e_1,...,e_m), a vector-space basis of E,
 * and quotient_subspace, a spanning family of W. Empty W means V = H.
 *
 * Return one n-by-n matrix A_i per e_i. Its column j is determined by
 *     e_i(delta) o f_j = sum_k A_i[k,j] f_k modulo W    (left_action = true),
 *     f_j o e_i        = sum_k A_i[k,j] f_k modulo W    (left_action = false).
 * Thus A_i c is the coordinate vector of e_i * v, or v * e_i, if c represents v.
 *
 * First form these compositions as Homomorphism objects, then call
 * homomorphism_quotient_coordinates to express them in the supplied basis.
 * That call solves for coordinates in V; it does not choose a further quotient.
 * delta is the actual shift. All f_j and W representatives have domain X and
 * target X(delta); all e_i have domain and target X.
 */
template <typename Matrix>
std::vector<SparseMatrix<std::ptrdiff_t>> endomorphism_composition_action(
    const std::vector<Homomorphism<Matrix>>& module_basis,
    const std::vector<Homomorphism<Matrix>>& endomorphism_basis,
    const typename Matrix::degree_type& delta,
    bool left_action,
    const std::vector<Homomorphism<Matrix>>& quotient_subspace = {}) {
    if (module_basis.empty())
        return std::vector<SparseMatrix<std::ptrdiff_t>>(
            endomorphism_basis.size(), SparseMatrix<std::ptrdiff_t>(0, 0));
    std::vector<Homomorphism<Matrix>> products;
    for (const Homomorphism<Matrix>& endomorphism : endomorphism_basis) {
        if (left_action) {
            Matrix shifted_lift = endomorphism.generator_lift();
            shifted_lift.shift(delta);
            Homomorphism<Matrix> shifted_endomorphism(
                module_basis.front().target(), module_basis.front().target(), std::move(shifted_lift));
            // Forward composition: f.compose(e(delta)) = e(delta) o f.
            for (const Homomorphism<Matrix>& representative : module_basis)
                products.push_back(representative.compose(shifted_endomorphism));
        } else {
            // Forward composition: e.compose(f) = f o e.
            for (const Homomorphism<Matrix>& representative : module_basis)
                products.push_back(endomorphism.compose(representative));
        }
    }
    // Each consecutive block of n products is the action of one e_i on the f_j.
    SparseMatrix<std::ptrdiff_t> product_coordinates = homomorphism_quotient_coordinates(
        module_basis, products, quotient_subspace);
    std::vector<SparseMatrix<std::ptrdiff_t>> actions;
    for (std::size_t g = 0; g < endomorphism_basis.size(); ++g) {
        SparseMatrix<std::ptrdiff_t> action(0, module_basis.size());
        for (std::size_t j = 0; j < module_basis.size(); ++j)
            action.append_col(product_coordinates.get_col(g * module_basis.size() + j));
        action.compute_num_cols();
        actions.push_back(std::move(action));
    }
    return actions;
}

/** Find b_1,...,b_r generating V as a module over a unital F2 algebra E.
 * Input: module_basis = (f_1,...,f_n) represents a vector-space basis of V.
 * Each n-by-n matrix actions[i] = A_i represents the action of e_i in that
 * basis, where (e_1,...,e_m) is a vector-space basis of the WHOLE algebra E.
 * Supplying only algebra generators is insufficient for the algorithm below.
 * V may be a Hom space or H/W; its choice is entirely encoded in these inputs.
 * Identify V with F2^n via the supplied basis in the following formulas.
 *
 * A candidate c in F2^n represents b = sum_j c_j [f_j]. Its cyclic submodule is
 *     E b = span_F2(A_1 c,...,A_m c)                     (left action),
 * or b E with the same coordinate formula for a right action. This is already
 * the full cyclic submodule: any e in E is a linear combination of the e_i,
 * and further actions stay in it because E is closed under multiplication.
 *
 * Starting with S = 0, evaluate each candidate by the rank of
 *     [basis columns of S | A_1 c | ... | A_m c].
 * Choose the candidate giving the largest dimension increase, replace S by
 * S + E b (or S + b E), and repeat until dim S = n. The returned Homomorphism
 * is the actual sum of the f_j selected by c; in H/W its class is the generator.
 *
 * Candidates are the total basis sum, 64 fixed-seed random sums, and all basis
 * vectors. The basis vectors guarantee progress since E contains the identity.
 * The result is a generating family, without a minimum-cardinality guarantee.
 * There is no coordinate solve or quotient computation in this function.
 */
template<class Matrix>
std::vector<Homomorphism<Matrix>> select_module_generators_from_action(
    const std::vector<Homomorphism<Matrix>>& module_basis,
    const std::vector<SparseMatrix<std::ptrdiff_t>>& actions) {
    const std::size_t dimension = module_basis.size();
    if (dimension == 0) return {};

    // Zero-based sparse coordinates: {0,2} means c=(1,0,1,0,...).
    std::vector<std::vector<std::ptrdiff_t>> candidates(1);
    for (std::size_t j = 0; j < dimension; ++j) candidates.front().push_back(j);
    std::mt19937 random(923407);
    for (int trial = 0; trial < 64; ++trial) {
        std::vector<std::ptrdiff_t> candidate;
        for (std::size_t j = 0; j < dimension; ++j)
            if (random() & 1) candidate.push_back(j);
        candidates.push_back(std::move(candidate));
    }
    for (std::size_t j = 0; j < dimension; ++j) candidates.push_back({std::ptrdiff_t(j)});

    // Independent coordinate columns spanning S = sum of the selected E b.
    SparseMatrix<std::ptrdiff_t> generated_span(0, dimension);
    std::vector<Homomorphism<Matrix>> selected;
    while (std::size_t(generated_span.get_num_cols()) < dimension) {
        const std::vector<std::ptrdiff_t>* best_candidate = nullptr;
        SparseMatrix<std::ptrdiff_t> best_span = generated_span;
        for (const std::vector<std::ptrdiff_t>& candidate : candidates) {
            SparseMatrix<std::ptrdiff_t> trial_span = generated_span;
            // trial_span = S + E b. XOR the columns indexed by c to obtain A_i c.
            for (const SparseMatrix<std::ptrdiff_t>& action : actions)
                trial_span.append_col(vectorXORMulti<std::ptrdiff_t>(action.data, candidate));
            trial_span.compute_num_cols();
            // Delete dependent/zero columns: the column count becomes dim(S + E b).
            trial_span.column_reduction_triangular(true);
            if (trial_span.get_num_cols() > best_span.get_num_cols()) {
                best_span = std::move(trial_span);
                best_candidate = &candidate;
            }
            if (std::size_t(best_span.get_num_cols()) == dimension) break;
        }
        if (!best_candidate)
            throw std::logic_error("Module generator selection failed to span the supplied algebra action");
        generated_span = std::move(best_span);
        // Lift b's coordinates back to a Homomorphism, rather than returning c.
        Homomorphism<Matrix> representative = module_basis[best_candidate->front()];
        for (std::size_t j = 1; j < best_candidate->size(); ++j)
            representative = representative + module_basis[(*best_candidate)[j]];
        selected.push_back(std::move(representative));
    }
    return selected;
}

/** Compute End(X)_d / End(X)_0, retaining X and a shared shifted target X_d.
 * d is the actual shift. The shifted target stores its presentation only.
 */
template <typename Matrix>
std::vector<Homomorphism<Matrix>> End_d_0(
    std::shared_ptr<const Module<Matrix>> X,
    const typename Matrix::degree_type& d, bool info = false,
    HomAlgorithm hom_algorithm = HomAlgorithm::optimised) {
    GRLINA_DEBUG_CHECK(if (!X) throw std::invalid_argument("End_d_0 requires X"));
    Matrix X_d_presentation = X->presentation();
    X_d_presentation.shift(d);
    if (X->presentation().get_num_rows() == 0) {
        if (info)
            std::cout << "dim End_0 = 0, dim End_d = 0, dim End_d/0 = 0" << std::endl;
        return {};
    }

    Matrix X_presentation = X->presentation();
    X_presentation.compute_rows_forward();
    auto End_d = hom_space_coefficients(X_presentation, X_d_presentation, hom_algorithm);
    auto End_0 = hom_algorithm == HomAlgorithm::optimised
        ? hom_space_shift_image_coefficients(X_presentation, X->presentation(), X_d_presentation)
        : hom_space_coefficients(X_presentation, X->presentation(), hom_algorithm);
    const auto shifted_dimension = End_d.get_num_cols();
    if (hom_algorithm != HomAlgorithm::optimised)
        MapCoefficientContext<Matrix>(X_presentation, X_d_presentation).normalize(End_0);
    reduce_quotient_in_place(End_0, End_d);
    if (info)
        std::cout << "dim eta End_0 = " << End_0.get_num_cols()
                  << ", dim End_d = " << shifted_dimension
                  << ", dim End_d/0 = " << End_d.get_num_cols()
                  << " (module maps; " << hom_algorithm_name(hom_algorithm) << ")" << std::endl;

    std::vector<Matrix> differentials;
    differentials.push_back(std::move(X_d_presentation));
    auto X_d = std::make_shared<const Module<Matrix>>(ChainComplex<Matrix>(std::move(differentials)));
    return homomorphisms_from_coefficients(End_d, X, X_d);
}

template <typename Matrix>
std::vector<Homomorphism<Matrix>> End_d_0(
    std::shared_ptr<Module<Matrix>> X,
    const typename Matrix::degree_type& d, bool info = false,
    HomAlgorithm hom_algorithm = HomAlgorithm::optimised) {
    return End_d_0<Matrix>(std::shared_ptr<const Module<Matrix>>(std::move(X)), d, info, hom_algorithm);
}

} // namespace graded_linalg
