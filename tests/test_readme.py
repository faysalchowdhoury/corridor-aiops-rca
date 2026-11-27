from __future__ import annotations

import pytest

from corridor_aiops.evaluation import readme


def test_update_replaces_only_the_generated_block(tmp_path, monkeypatch) -> None:
    path = tmp_path / "README.md"
    path.write_text(f"intro\n{readme.BEGIN}\nold numbers\n{readme.END}\noutro\n", encoding="utf-8")
    monkeypatch.setattr(readme, "render", lambda results: f"{readme.BEGIN}\nnew\n{readme.END}")

    readme.update_readme(path, tmp_path)

    assert path.read_text(encoding="utf-8") == f"intro\n{readme.BEGIN}\nnew\n{readme.END}\noutro\n"


def test_update_requires_markers(tmp_path) -> None:
    path = tmp_path / "README.md"
    path.write_text("no markers\n", encoding="utf-8")

    with pytest.raises(ValueError, match="markers"):
        readme.update_readme(path, tmp_path)
