from rag.loader import load_documents


def test_loads_txt_and_ignores_other_files(tmp_path):
    (tmp_path / "notes.txt").write_text("Migraine is a neurological disorder.", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"not text")
    (tmp_path / ".gitkeep").write_text("")
    pages = load_documents(tmp_path)
    assert len(pages) == 1 and pages[0]["source"] == "notes.txt"
