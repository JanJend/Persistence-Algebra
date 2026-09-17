#include <grlina/modules.hpp>
#include <cassert>
#include <limits>

using namespace graded_linalg;
using Mat = R2GradedSparseMatrix<int>;
using Mod = Module<Mat>;
using Sub = Submodule<Mat>;
using Ptr = std::shared_ptr<const Mod>;

template <typename F> void rejects(F action) {
    bool threw = false;
    try { action(); } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
}

int main() {
    // Unequal birth coordinates distinguish the shift direction and both axes.
    Ptr free = std::make_shared<Mod>(Mat(0, 1, {}, {}, {{1, 2}}));
    auto image = epsilon_image(free, 0.5);
    assert(image.parent() == free);
    assert(image.generator_map().generator_lift().row_degrees == free->presentation().row_degrees);
    assert(image.generator_map().generator_lift().col_degrees == vec<r2degree>({{1.5, 2.5}}));
    assert(image.generator_map().generator_lift().data == vec<vec<int>>({{0}}));
    auto image_module = image.presented_module();
    assert(image_module.dimension_at({1.5, 2.5}) == 1);
    assert(image_module.dimension_at({1.25, 2.5}) == 0);
    assert(image_module.dimension_at({1.5, 2.25}) == 0);
    assert(epsilon_kernel(free, 0.5).is_zero());

    // Shifting a presented submodule must discard its old module presentation.
    auto shifted = Sub::whole(free);
    shifted.compute_projective_resolution();
    shifted.shift_generators({0.5, 0.5});
    assert(!shifted.has_presentation());
    assert(!shifted.has_injective_resolution());
    assert(shifted.parent() == free && shifted.equals(image));
    assert(shifted.presented_module().dimension_at({1, 2}) == 0);
    rejects([&] { shifted.shift_generators({-1, 0}); });
    assert(shifted.equals(image));

    // The relation e0 + e1 dies at (3,4); its kernel starts at (2.5,3.5).
    // Neither individual generator is killed by the structure map.
    Ptr coupled = std::make_shared<Mod>(
        Mat(1, 2, {{0, 1}}, {{3, 4}}, {{0, 1}, {1, 0}}));
    const auto original = coupled->presentation();
    auto kernel = epsilon_kernel(coupled, 0.5);
    Sub expected(coupled, Mat(1, 2, {{0, 1}}, {{2.5, 3.5}}, original.row_degrees));
    assert(kernel.parent() == coupled);
    assert(kernel.generator_map().generator_lift().row_degrees == original.row_degrees);
    assert(kernel.equals(expected));
    auto kernel_module = kernel.presented_module();
    assert(kernel_module.dimension_at({2.5, 3.5}) == 1);
    assert(kernel_module.dimension_at({2.25, 3.5}) == 0);
    assert(kernel_module.dimension_at({3, 4}) == 0);
    assert(epsilon_image(coupled, 0).equals(Sub::whole(coupled)));
    assert(epsilon_kernel(coupled, 0).is_zero());
    // Large shifts must still respect the birth degrees of every summand.
    assert(epsilon_kernel(coupled, 10).equals(
        Sub(coupled, Mat(1, 2, {{0, 1}}, {{1, 1}}, original.row_degrees))));
    assert(coupled->presentation().data == original.data);
    assert(coupled->presentation().row_degrees == original.row_degrees);
    assert(coupled->presentation().col_degrees == original.col_degrees);

    Ptr torsion = std::make_shared<Mod>(Mat(1, 1, {{0}}, {{1, 1}}, {{0, 0}}));
    assert(epsilon_image(torsion, 1).is_zero());
    assert(epsilon_kernel(torsion, 1).equals(Sub::whole(torsion)));
    Ptr empty = std::make_shared<Mod>(Mat(0, 0));
    assert(epsilon_image(empty, 1).is_zero());
    assert(epsilon_kernel(empty, 1).is_zero());

    for (double invalid : {-1.0, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
        rejects([&] { epsilon_image(free, invalid); });
        rejects([&] { epsilon_kernel(free, invalid); });
    }
    rejects([] { epsilon_image<Mat>(nullptr, 1); });
    rejects([] { epsilon_kernel<Mat>(nullptr, 1); });
}
