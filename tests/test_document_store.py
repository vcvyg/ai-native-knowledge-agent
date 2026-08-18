from __future__ import annotations

from app.document_store import resolve_document_target


def test_repeated_content_resolves_to_existing_document(tmp_path) -> None:
    first = resolve_document_target(tmp_path, "Architecture", "same\r\ncontent")
    first.path.write_text("same\ncontent", encoding="utf-8")

    repeated = resolve_document_target(tmp_path, "Renamed upload", "same\ncontent")

    assert repeated.already_exists is True
    assert repeated.path == first.path
    assert repeated.digest == first.digest


def test_changed_content_gets_a_new_content_address(tmp_path) -> None:
    first = resolve_document_target(tmp_path, "Runbook", "version one")
    first.path.write_text("version one", encoding="utf-8")

    changed = resolve_document_target(tmp_path, "Runbook", "version two")

    assert changed.already_exists is False
    assert changed.path != first.path
    assert changed.digest != first.digest
