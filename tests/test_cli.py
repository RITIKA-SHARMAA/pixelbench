from PIL import Image
from typer.testing import CliRunner

from pixelbench.cli import app


def test_edges_cli(tmp_path):
    source, output = tmp_path / "source.png", tmp_path / "edges.png"
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(source)
    result = CliRunner().invoke(app, ["edges", str(source), str(output)])
    assert result.exit_code == 0, result.output
    assert output.exists()
