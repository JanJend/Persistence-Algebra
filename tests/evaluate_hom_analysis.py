import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# -------- regex patterns --------
header_re = re.compile(r"Noninterval of size (\d+)x(\d+) and thickness: (\d+)")
time_re = re.compile(r"After .*?, time:\s*(\d+)")
reduce_re = re.compile(r"Time to reduce to basis:\s*(\d+)ms")
algorithms = ["Alg_B", "Naive", "Mixed", "Alg_A"]
algorithms_extended = ["Alg_B", "Naive", "Naive_corr", "Mixed", "Alg_A"]
algorithms_comp = ["Alg_B", "Naive_corr", "Mixed", "Alg_A"]  # for comparison without Naive
algorithms_name_keys = {"Alg_B": "Algorithm B", "Naive": "Direct Computation", "Naive_corr": "No Reduction", "Mixed": "Algorithm A 1/2", "Alg_A": "Algorithm A"}
algorithms_names = [algorithms_name_keys[alg] for alg in algorithms_extended]
colors = {
    "Alg_B": "magenta",
    "Naive": "green",
    "Naive_corr": "blue",
    "Mixed": "cyan",
    "Alg_A": "red",
}

colors_list = [colors[alg] for alg in algorithms_extended]

def parse_markdown(path):
    results = []
    with open(path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        m = header_re.search(lines[i])
        if not m:
            i += 1
            continue

        A, B, thickness = map(int, m.groups())
        size_sum = A + B
        times = {}

        i += 1
        # process until next ## header or end of file
        while i < len(lines) and not lines[i].startswith("##"):
            line = lines[i].strip()
            for alg in algorithms:
                if line.startswith(alg + ":"):
                    # parse until next algorithm or block end
                    reduce_time = 0
                    j = i + 1
                    while j < len(lines):
                        subline = lines[j].strip()
                        if any(subline.startswith(a + ":") for a in algorithms) or subline.startswith("##"):
                            break
                        tm = time_re.search(subline)
                        if tm:
                            times[alg] = int(tm.group(1))
                        if alg == "Naive":
                            m_reduce = reduce_re.search(subline)
                            if m_reduce:
                                reduce_time = int(m_reduce.group(1))
                        j += 1
                    if alg == "Naive":
                        times["Naive_corr"] = max(1, times.get("Naive", 0) - reduce_time)
                    i = j - 1  # skip lines already processed
            i += 1

        # append block if all five keys are present
        required_keys = ["Alg_B", "Naive", "Naive_corr", "Mixed", "Alg_A"]
        if all(k in times for k in required_keys):
            results.append({
                "size": size_sum,
                "thickness": thickness,
                "Alg_B": max(1, times["Alg_B"]),
                "Naive": max(1, times["Naive"]),
                "Naive_corr": times["Naive_corr"],
                "Mixed": max(1, times["Mixed"]),
                "Alg_A": max(1, times["Alg_A"]),
            })

    return results


# -------- files --------
md_files = [
    Path("/home/jan/MP-Workspace/data/1.5mmRegions/CD8_scc/ind_hom_time_analysis_H1.md"),
    Path("/home/jan/MP-Workspace/data/1.5mmRegions/CD68_scc/ind_hom_time_analysis_H1.md"),
    Path("/home/jan/MP-Workspace/data/1.5mmRegions/FoxP3_scc/ind_hom_time_analysis_H1.md"),
]


md_file_names = [
    "CD8",
    "CD68",
    "FoxP3"
]

all_data = []      # concatenated
per_file_data = [] # list of lists
for md in md_files:
    file_data = parse_markdown(md)
    per_file_data.append(file_data)
    all_data.extend(file_data)
# -------- helper to unpack --------
def unpack(data_list):
    AplusB  = [d["size"] for d in data_list]
    thickness = [d["thickness"] for d in data_list]
    algB  = [d["Alg_B"] for d in data_list]
    naive = [d["Naive"] for d in data_list]
    naive_corr = [d["Naive_corr"] for d in data_list]
    mixed = [d["Mixed"] for d in data_list]
    algA  = [d["Alg_A"] for d in data_list]
    return AplusB, thickness, algB, naive, naive_corr, mixed, algA

# -------- plot thickness vs size per file --------
plt.figure()
markers = ["o", "s", "D"]
colors_cell_types = ["r", "g", "b"]
marker_size = 4
for idx, data in enumerate(per_file_data):
    AplusB, thickness, *_ = unpack(data)
    plt.plot(AplusB, thickness, markers[idx], linestyle="none", color=colors_cell_types[idx], label=md_file_names[idx], markersize=marker_size)

plt.xlabel("#Generators + #Relations")
plt.ylabel("Thickness")
plt.title("Thickness of non-interval indecomposables")
plt.legend()
plt.show()

# -------- plot time vs size per file with smaller points --------
plt.figure()
alg_labels = ["Algorithm B", "Direct Computation", "No Redcution", "A 1/2", "Alg_A"]
point_size = 10  # smaller points

for idx, data in enumerate(per_file_data):
    AplusB, _, algB, naive, naive_corr, mixed, algA = unpack(data)
    all_alg = [algB, naive, naive_corr, mixed, algA]
    for a_idx, times in enumerate(all_alg):
        plt.scatter(AplusB, times, s=point_size,
                    label=f"{algorithms_names[a_idx]} - {md_file_names[idx]}",
                    marker=markers[idx], color=colors_list[a_idx])

plt.xlabel("#Generators + #Relations")
plt.ylabel("Time (ms)")
plt.yscale("log")
plt.title("Runtime vs Size for non-interval indecomposables")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()


# -------- log-log fit with corrected Naive and distinct line styles --------

# Unpack concatenated data
AplusB, _, algB, naive, naive_corr, mixed, algA = unpack(all_data)
x = np.array(AplusB)


# Unpack concatenated data
AplusB, _, algB, naive, naive_corr, mixed, algA = unpack(all_data)

x = np.array(AplusB)

# Size filter
mask_size = x <= 3000
x_filtered = x[mask_size]

# Algorithms including Naive
algorithms_dict = {
    "Alg_B": np.array(algB),
    "Naive": np.array(naive),
    "Naive_corr": np.array(naive_corr),
    "Mixed": np.array(mixed),
    "Alg_A": np.array(algA),
}

line_styles = {
    "Alg_B": "-",
    "Naive": "--",
    "Naive_corr": ":",
    "Mixed": "-.",
    "Alg_A": (0, (3, 1, 1, 1)),  # dash-dot-dot
}


# Filtered algorithms
algorithms_dict = {
    "Alg_B": np.array(algB)[mask_size],
    "Naive_corr": np.array(naive_corr)[mask_size],
    "Mixed": np.array(mixed)[mask_size],
    "Alg_A": np.array(algA)[mask_size]
}



# ------------------ 1) Scatter-only plot ------------------
plt.figure(figsize=(8, 6))

for alg in algorithms:
    x_vals = []
    y_vals = []

    for d in all_data:
        if d["size"] <= 3000 and d[alg] <= 10000:
            x_vals.append(d["size"])
            y_vals.append(d[alg])

    plt.scatter(
        x_vals,
        y_vals,
        s=marker_size,
        alpha=0.7,
        color=colors[alg],
        label=algorithms_names[algorithms_extended.index(alg)]
    )

plt.xlabel("#Generators + #Relations")
plt.ylabel("Time (ms)")
plt.title("Runtime of End(X) for non-interval indecomposables, size <= 3000")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

# ------------------ 1) Scatter-only unfiltered log plot ------------------
plt.figure(figsize=(8, 6))

for alg in algorithms_extended:
    x_vals = []
    y_vals = []

    for d in all_data:
        if d["size"] > 0 and d[alg] > 9:
            x_vals.append(d["size"])
            y_vals.append(d[alg])

    plt.scatter(
        x_vals,
        y_vals,
        s=marker_size,
        alpha=0.7,
        color=colors[alg],
        label=algorithms_names[algorithms_extended.index(alg)]
    )

plt.xlabel("#Generators + #Relations")
plt.ylabel("Time (ms)")
plt.yscale("log")
plt.ylim(bottom=10)
plt.title("Runtime (log) of End(X) for non-interval indecomposables")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()


from scipy.stats import linregress

# ------------------ 2) Scatter + log-log fitted lines (with log-binning) ------------------
plt.figure(figsize=(8, 6))

for alg in algorithms_comp:
    # Collect data with size > 1000
    x_vals = [d["size"] for d in all_data if d["size"] <= 1000]
    y_vals = [d[alg] for d in all_data if d["size"] <= 1000]

    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)

    # Safety for log
    x = np.maximum(x, 1e-6)
    y = np.maximum(y, 1e-6)

    # --- LOG-BINNING ---
    n_bins = 20  # adjust as needed
    bins = np.logspace(np.log10(x.min()), np.log10(x.max()), n_bins + 1)
    bin_indices = np.digitize(x, bins)

    x_bin = []
    y_bin = []
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            x_bin.append(np.mean(x[mask]))
            y_bin.append(np.mean(y[mask]))

    x_bin = np.array(x_bin)
    y_bin = np.array(y_bin)

    # --- LOG-LOG REGRESSION ON BINS ---
    slope, intercept, *_ = linregress(np.log(x_bin), np.log(y_bin))
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = np.exp(intercept) * x_fit**slope

    # Regression line
    plt.plot(
        x_fit,
        y_fit,
        color=colors[alg],
        linestyle=line_styles[alg],
        linewidth=2,
        label=f"{algorithms_names[algorithms_extended.index(alg)]} reg (T ∼ n^{slope:.2f})"
    )

    # Scatter points
    plt.scatter(
        x,
        y,
        s=marker_size,
        alpha=0.45,
        color=colors[alg]
    )

plt.xlabel("#Generators + #Relations")
plt.ylabel("Time (ms)")
plt.title("Runtime of End(X) with Power-Law Fits (log-binned)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()
# ------------------ 2) Scatter + log-log fitted lines (with log-binning) ------------------
plt.figure(figsize=(8, 6))

for alg in algorithms_extended:
    # Collect data with size > 1000
    x_vals = [d["size"] for d in all_data if d["size"] > 1000]
    y_vals = [d[alg] for d in all_data if d["size"] > 1000]

    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)

    # Safety for log
    x = np.maximum(x, 1e-6)
    y = np.maximum(y, 1e-6)

    # --- SCATTER POINTS (semi-transparent) ---
    plt.scatter(
        x,
        y,
        s=8,               # marker size
        color=colors[alg],
        alpha=0.4,          # semi-transparent
        label=f"{algorithms_names[algorithms_extended.index(alg)]} data"
    )

    # --- LOG-BINNING ---
    n_bins = 20  # adjust as needed
    bins = np.logspace(np.log10(x.min()), np.log10(x.max()), n_bins + 1)
    bin_indices = np.digitize(x, bins)

    x_bin = []
    y_bin = []
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            x_bin.append(np.mean(x[mask]))
            y_bin.append(np.mean(y[mask]))

    x_bin = np.array(x_bin)
    y_bin = np.array(y_bin)

    # --- LOG-LOG REGRESSION ON BINS ---
    slope, intercept, *_ = linregress(np.log(x_bin), np.log(y_bin))
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = np.exp(intercept) * x_fit**slope

    # Regression line
    plt.plot(
        x_fit,
        y_fit,
        color=colors[alg],
        linestyle=line_styles[alg],
        linewidth=2,
        label=f"{algorithms_names[algorithms_extended.index(alg)]} reg (T ∼ n^{slope:.2f})"
    )

plt.xlabel("#Generators + #Relations")
plt.ylabel("Time (ms)")
plt.title("Runtime of End(X) with Power-Law Fits (log-binned)")

plt.ylim(0, 30000) # cutoff y-axis at 30000
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()
