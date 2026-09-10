#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::string read_file(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Unable to read " + path.string());
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

static std::string shell_quote(const fs::path& path) {
    std::string result = "'";
    for (char c : path.string()) result += c == '\'' ? "'\\''" : std::string(1, c);
    return result + "'";
}

static std::string shell_quote(const std::string& value) {
    return shell_quote(fs::path(value));
}

static std::string run(const fs::path& executable, const std::vector<std::string>& arguments,
                       const fs::path& working_directory, const std::string& label) {
    const fs::path output = working_directory / (label + ".log");
    std::string command = "cd " + shell_quote(working_directory) + " && " + shell_quote(executable);
    for (const auto& argument : arguments) command += " " + shell_quote(argument);
    command += " > " + shell_quote(output) + " 2>&1";
    const int status = std::system(command.c_str());
    const std::string text = read_file(output);
    if (status != 0) {
        throw std::runtime_error(label + " failed with status " + std::to_string(status) + "\n" + text);
    }
    return text;
}

static void require_equal(const fs::path& actual, const fs::path& expected) {
    if (read_file(actual) != read_file(expected)) {
        throw std::runtime_error("Output mismatch: " + actual.string() +
                                 " != " + expected.string());
    }
}

static void require_contains(const std::string& output, const std::string& expected,
                             const std::string& label) {
    if (output.find(expected) == std::string::npos)
        throw std::runtime_error(label + " did not contain: " + expected + "\n" + output);
}

int main(int argc, char** argv) {
    // fixture directory followed by every module-aware CLI in CMake order.
    if (argc != 17) {
        std::cerr << "Expected fixture directory and 15 executable paths\n";
        return 2;
    }

    const fs::path fixtures = argv[1];
    std::vector<fs::path> executable;
    for (int i = 2; i < argc; ++i) executable.emplace_back(argv[i]);
    const fs::path temporary = fs::temp_directory_path() /
        ("pa-cli-regression-" + std::to_string(
            std::chrono::high_resolution_clock::now().time_since_epoch().count()));

    try {
        fs::create_directories(temporary);

        fs::copy_file(fixtures / "cancellation.scc", temporary / "minimize.scc");
        run(executable[0], {(temporary / "minimize.scc").string()}, temporary, "minimize");
        require_equal(temporary / "minimize_min.scc", fixtures / "cancellation_min_expected.scc");

        const std::string hom = run(executable[1],
            {(fixtures / "interval.scc").string(), (fixtures / "interval.scc").string(), "0", "0"},
            temporary, "hom");
        require_contains(hom, "Dimension: 1", "hom");

        const std::string shifted = run(executable[2], {(fixtures / "interval.scc").string()},
                                        temporary, "shift_endo");
        require_contains(shifted, "1 x 1", "shift_endo");
        for (const std::string epsilon : {"0", "0.005", "0.01", "0.015", "0.02"})
            require_contains(shifted, "Epsilon: " + epsilon + " Number of endomorphisms: 1",
                             "shift_endo");

        run(executable[3], {(fixtures / "interval.scc").string(), "0", "0",
                            (temporary / "quiver.txt").string()}, temporary, "pres_to_quiver");
        require_equal(temporary / "quiver.txt", fixtures / "interval_quiver_expected.txt");

        run(executable[4], {(fixtures / "interval.scc").string(),
                            (temporary / "resolution.scc").string()}, temporary, "resolution");
        require_equal(temporary / "resolution.scc", fixtures / "interval_resolution_expected.scc");

        run(executable[5], {(fixtures / "interval.scc").string(), "2",
                            (temporary / "snapped.scc").string()}, temporary, "snap_grid");
        require_equal(temporary / "snapped.scc", fixtures / "interval.scc");

        fs::copy_file(fixtures / "interval.scc", temporary / "submodule.scc");
        run(executable[6], {(temporary / "submodule.scc").string(), "0,0"},
            temporary, "submodule_at");
        require_equal(temporary / "submodule_min.scc", fixtures / "interval.scc");

        const std::string analysis = run(executable[7], {(fixtures / "interval.scc").string()},
                                         temporary, "analyse_ind");
        require_contains(analysis, "Dimension of hom-space: 1", "analyse_ind");
        require_contains(analysis, "thickness: 1", "analyse_ind");

        const std::string thickness = run(executable[8], {(fixtures / "interval.scc").string()},
                                          temporary, "thickness");
        require_contains(thickness, "1 x 1: thickness: 1", "thickness");

        const std::string size = run(executable[9], {(fixtures / "interval.scc").string()},
                                     temporary, "size");
        require_contains(size, "\n1\n", "size");

        run(executable[10], {(fixtures / "interval.scc").string(), "1", "1",
                             (temporary / "bounded.scc").string()}, temporary, "cut_module");
        require_equal(temporary / "bounded.scc", fixtures / "interval_bound_expected.scc");

        fs::copy_file(fixtures / "interval.scc", temporary / "birth_death.scc");
        run(executable[11], {(temporary / "birth_death.scc").string()}, temporary, "birth_death");
        require_equal(temporary / "birth_death_birth.scc", fixtures / "interval_birth_expected.scc");
        require_equal(temporary / "birth_death_death.scc", fixtures / "interval_death_expected.scc");

        run(executable[12], {(fixtures / "interval.scc").string(), "0.5", "0.5",
                             (temporary / "deleted.scc").string()}, temporary,
            "delete_gens_and_rels");
        require_equal(temporary / "deleted.scc", fixtures / "free_origin_expected.scc");

        fs::copy_file(fixtures / "cancellation.scc", temporary / "minimize_pres.scc");
        run(executable[13], {(temporary / "minimize_pres.scc").string()},
            temporary, "minimize_pres");
        require_equal(temporary / "minimize_pres_min.scc", fixtures / "cancellation_min_expected.scc");

        run(executable[14], {(fixtures / "interval_resolution_expected.scc").string(),
                             (temporary / "homology.scc").string()}, temporary, "mpfree_clone");
        require_equal(temporary / "homology.scc", fixtures / "zero_module_expected.scc");

        fs::remove_all(temporary);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        std::cerr << "CLI artifacts retained in " << temporary << std::endl;
        return 1;
    }
}
