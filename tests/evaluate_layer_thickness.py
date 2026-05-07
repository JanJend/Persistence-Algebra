import os
from collections import defaultdict
import matplotlib.pyplot as plt

MARKDOWN_FILE = "/home/jan/MP-Workspace/data/1.5mmRegions/CD8_scc/a_thickness_analysis.md"
SCC_FOLDER = "/home/jan/MP-Workspace/data/1.5mmRegions/CD8_scc"

def parse_markdown_file():
    """Parse markdown for thickness values and header names"""
    data = defaultdict(dict)
    with open(MARKDOWN_FILE, "r") as f:
        lines = [line.strip() for line in f]

    current_base = None
    current_h = None

    for i, line in enumerate(lines):
        if line.startswith("## ") and line.endswith(".sccsum"):
            # header line
            header = line[3:-7]  # remove '## ' and '.sccsum'
            if "_H0" in header:
                current_h = "0"
                current_base = header.replace("_H0", "")
            elif "_H1" in header:
                current_h = "1"
                current_base = header.replace("_H1", "")
            else:
                current_h = None
                current_base = None

        elif line.startswith("layer thickness") and current_base and current_h:
            # parse thickness
            try:
                fields = line.split(",")
                thickness = None
                for f in fields:
                    if ":" not in f:
                        continue
                    key, value = f.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    if key == "layer thickness":
                        thickness = float(value)
                if thickness is not None:
                    data[current_base][current_h] = {"thickness": thickness, "size": None}
                else:
                    print(f"Line {i}: Could not parse thickness in line: {line}")
            except Exception as e:
                print(f"Line {i}: Error parsing thickness: {e}")

    return data

def add_sizes_from_scc(data):
    """For each header in data, read the corresponding .scc file and compute size"""
    for base_name, h_dict in data.items():
        for h_type in ["0", "1"]:
            if h_type not in h_dict:
                continue
            # build the expected filename
            scc_files = [f for f in os.listdir(SCC_FOLDER) if f.endswith(f"{base_name}_H{h_type}.scc")]
            if not scc_files:
                print(f"Warning: no .scc file found for {base_name}_H{h_type}")
                continue
            scc_file = os.path.join(SCC_FOLDER, scc_files[0])
            try:
                with open(scc_file, "r") as f:
                    # read first three lines
                    f.readline()
                    f.readline()
                    third_line = f.readline().strip()
                    # third line looks like: "2640 1235" possibly followed by other stuff
                    nums = third_line.split()[:2]  # first two numbers
                    size = sum(int(float(n)) for n in nums)
                    h_dict[h_type]["size"] = size
            except Exception as e:
                print(f"Error reading {scc_file}: {e}")

    # convert to final format (thickness, size)
    result = {}
    for base_name, h_dict in data.items():
        result[base_name] = [
            (h_dict.get("0", {}).get("size"), h_dict.get("0", {}).get("thickness")),
            (h_dict.get("1", {}).get("size"), h_dict.get("1", {}).get("thickness"))
        ]
    return result

def plot_data(data):
    plt.figure(figsize=(10, 6))

    h0_data = []
    h1_data = []
    labels = []
    point_size = 8

    for base_name, pairs in data.items():
        h0, h1 = pairs
        if h0[0] is not None and h0[1] is not None:
            h0_data.append(h0)
            labels.append(base_name + "_H0")
        if h1[0] is not None and h1[1] is not None:
            h1_data.append(h1)
            labels.append(base_name + "_H1")

    if not h0_data and not h1_data:
        print("No valid data points found for plotting")
        return

    if h0_data:
        h0_sizes, h0_thicknesses = zip(*h0_data)
        plt.scatter(h0_sizes, h0_thicknesses, c='blue', marker='o', label='H0', s=8, alpha=0.3)

    if h1_data:
        h1_sizes, h1_thicknesses = zip(*h1_data)
        plt.scatter(h1_sizes, h1_thicknesses, c='red', marker='s', label='H1', s=8, alpha=0.3)

    plt.xlabel('Size')
    plt.ylabel('Layer Thickness for CD8 immune cells')
    plt.title('Layer Thickness vs Size for H0 and H1')
    plt.legend()
    plt.grid(True)

    if len(labels) <= 20:
        for i, (x, y) in enumerate(h0_data + h1_data):
            plt.text(x, y, labels[i], fontsize=8, ha='right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data = parse_markdown_file()
    data = add_sizes_from_scc(data)

    plot_data(data)
