"""Run with Matplotlib: python test_heatmap_colours.py HILBERT_IMAGE LANDSCAPE_IMAGE."""

from io import BytesIO
import base64
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import matplotlib
import matplotlib.image
import numpy as np


def main(hilbert, landscape):
    with TemporaryDirectory() as temporary:
        source = Path(temporary) / "free.scc"
        source.write_text("scc2020\n2\n0 1 0\n0 0 ;\n")
        base = [str(source), "257", "257", "--bounds", "0", "0", "1", "1"]

        def render(binary, *options):
            result = subprocess.run([str(binary), *base, *options], check=True, capture_output=True)
            return np.rint(matplotlib.image.imread(BytesIO(result.stdout))[:, :, :3] * 255).astype(np.uint8)

        blue = render(landscape, "--resolution", "0.25")
        explicit_blue = render(landscape, "--resolution", "0.25", "--colormap", "blue")
        np.testing.assert_array_equal(blue, explicit_blue)
        hilbert_blue = render(hilbert, "--scale", "1", "256")
        # Entire colourbars must agree despite the different numerical scales.
        np.testing.assert_array_equal(blue[85:342, 392], hilbert_blue[85:342, 392])
        np.testing.assert_array_equal(blue[85, 100], [255, 255, 255])  # zero
        np.testing.assert_array_equal(blue[85, 228], [24, 109, 228])   # muted midpoint
        np.testing.assert_array_equal(blue[85, 356], [0, 0, 0])        # maximum

        inferno = render(landscape, "--resolution", "0.25", "--colormap", "inferno")
        expected = matplotlib.colormaps["inferno"](np.arange(257) / 256, bytes=True)[:, :3]
        # The free quadrant gives lambda(x,1)=x, covering every LUT entry.
        np.testing.assert_array_equal(inferno[85, 100:357], expected)
        np.testing.assert_array_equal(inferno[85:342, 392], expected[::-1])
        np.testing.assert_array_equal(inferno[420, 106], expected[0])  # zero swatch
        zero = render(landscape, "--layer", "2", "--colormap", "inferno")
        assert np.all(zero[85:342, 100:357] == expected[0])
        assert np.all(zero[85:342, 392] == expected[0])

        def maximum(*options):
            result = subprocess.run([str(landscape), *base, *options, "--maximum-only"],
                                    check=True, capture_output=True)
            return float(result.stdout)

        assert maximum() == 1 and maximum("--layer", "2") == 0
        command = [str(landscape), *base, "--include-values"]
        result = subprocess.run(command, check=True, capture_output=True)
        bundle = json.loads(result.stdout)
        assert bundle["maximum"] == 1 and bundle["layer"] == 1 and bundle["bounds"] == [0, 0, 1, 1]
        values = np.asarray(bundle["values"])
        axis = np.linspace(0, 1, 257)
        np.testing.assert_allclose(values, np.minimum(axis[:, None], axis[None, :]), atol=1e-15)
        assert base64.b64decode(bundle["png"]).startswith(b"\x89PNG\r\n\x1a\n")
        fresh = subprocess.run(command + ["--no-vineyard"], check=True, capture_output=True)
        assert json.loads(fresh.stdout) == bundle
        samples = subprocess.run([str(landscape), *base, "--values-only"], check=True, capture_output=True)
        assert "png" not in json.loads(samples.stdout)
        for palette in ("blue", "inferno"):
            rerender = subprocess.run([str(landscape), "--render-values", "--scale", "2", "--colormap", palette, "--json"],
                                     input=samples.stdout, check=True, capture_output=True)
            rerender = json.loads(rerender.stdout)
            assert rerender["maximum"] == 1 and "values" not in rerender
            direct = subprocess.run([str(landscape), *base, "--scale", "2", "--colormap", palette], check=True, capture_output=True)
            assert base64.b64decode(rerender["png"]) == direct.stdout
        invalid = dict(bundle, values=[[0, 1], [0]])
        failure = subprocess.run([str(landscape), "--render-values"], input=json.dumps(invalid).encode(), capture_output=True)
        assert failure.returncode != 0 and not failure.stdout
        assert maximum("--direction", "2", "2") == 0.5
        shifted = Path(temporary) / "shifted.scc"
        shifted.write_text("scc2020\n2\n0 1 0\n0.5 0.5 ;\n")
        base[0] = str(shifted)
        assert maximum() == 0.5
        for palette in ("blue", "inferno"):
            base[0] = str(source)
            full = render(landscape, "--colormap", palette, "--scale", "1")
            empty = render(landscape, "--colormap", palette, "--scale", "1", "--layer", "2")
            base[0] = str(shifted)
            half = render(landscape, "--colormap", palette, "--scale", "1")
            # The same height 0.25 gets exactly the same colour despite different maxima.
            np.testing.assert_array_equal(full[85, 164], half[85, 292])
            np.testing.assert_array_equal(full[85:342, 392], half[85:342, 392])
            np.testing.assert_array_equal(full[85:342, 392], empty[85:342, 392])
        base[0] = str(source)
        scaled = render(landscape, "--scale", "2")
        np.testing.assert_array_equal(scaled[85, 356], [24, 109, 228])

        for option in (["--colormap", "unknown"], ["--colormap"], ["--scale", "0"],
                       ["--scale", "-1"], ["--scale", "nan"], ["--scale", "inf"], ["--scale"]):
            result = subprocess.run([str(landscape), *base, *option], capture_output=True)
            assert result.returncode != 0 and not result.stdout
    print("Shared value scales, sampled maxima, both palettes, zero plots, and CLI validation passed.")


if __name__ == "__main__":
    main(*(Path(path).resolve() for path in sys.argv[1:]))
