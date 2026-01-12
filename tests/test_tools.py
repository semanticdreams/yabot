import json
from pathlib import Path

from yabot.tools.create_dir import create_dir
from yabot.tools.list_dir import list_dir
from yabot.tools.read_file import read_file
from yabot.tools.run_shell import run_shell
from yabot.tools.write_file import write_file


def test_list_dir(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hi", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    payload = json.loads(list_dir(str(tmp_path)))
    names = {entry["name"]: entry["type"] for entry in payload["entries"]}
    assert names["a.txt"] == "file"
    assert names["sub"] == "dir"


def test_read_file(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("hello", encoding="utf-8")
    assert read_file(str(path)) == "hello"


def test_write_file(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    result = write_file(str(path), "data")
    assert path.read_text(encoding="utf-8") == "data"
    assert "OK:" in result


def test_create_dir(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir"
    result = create_dir(str(path), exist_ok=True)
    assert path.exists()
    assert "OK:" in result


def test_run_shell(tmp_path: Path) -> None:
    payload = json.loads(run_shell("echo hi", workdir=str(tmp_path)))
    assert payload["returncode"] == 0
    assert payload["stdout"].strip() == "hi"
