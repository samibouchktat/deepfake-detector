"""Contrôler le regroupement des sources et les erreurs de manifeste."""

import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from data.prepare_manifest import ROOT, MANIFEST_FIELDS, assign_splits, read_inventory, source_groups


def inventory() -> list[dict[str, str]]:
    rows = []
    for index in range(6):
        for label, prefix in (("0", "original"), ("1", "fake")):
            rows.append({
                "video_id": f"{prefix}_{index}", "source_id": f"original_{index}",
                "label": label, "path": f"data/raw/{prefix}_{index}.mp4", "split": "",
            })
    return rows


class ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "inventory.csv"

    def write_rows(self, rows: list[dict[str, str]]) -> None:
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def test_reproducible_split_keeps_all_sources_together(self) -> None:
        rows = inventory()
        first = assign_splits(rows)
        self.assertEqual(first, assign_splits(rows))
        self.assertTrue(all(row["split"] == "" for row in rows))
        self.assertEqual({row["split"] for row in first}, {"train", "val", "test"})
        for group in source_groups(first):
            self.assertEqual(len({first[i]["split"] for i in group}), 1)
        reversed_result = assign_splits(list(reversed(rows)))
        self.assertEqual(
            {row["video_id"]: row["split"] for row in first},
            {row["video_id"]: row["split"] for row in reversed_result},
        )

    def test_multiple_sources_and_transitive_links_merge_groups(self) -> None:
        rows = inventory()
        rows.extend([
            {"video_id": "mixed", "source_id": "original_0|original_1", "label": "1",
             "path": "data/raw/mixed.mp4", "split": ""},
            {"video_id": "derived", "source_id": "mixed|original_2", "label": "1",
             "path": "data/raw/derived.mp4", "split": ""},
        ])
        result = assign_splits(rows)
        linked = {"original_0", "fake_0", "original_1", "fake_1", "original_2", "fake_2", "mixed", "derived"}
        self.assertEqual(len({row["split"] for row in result if row["video_id"] in linked}), 1)
        self.assertEqual(len(source_groups(rows)), 4)

    def test_detects_leakage_in_existing_manifest(self) -> None:
        rows = assign_splits(inventory())
        original_split = rows[0]["split"]
        rows[1]["split"] = "val" if original_split != "val" else "test"
        self.write_rows(rows)
        with self.assertRaisesRegex(ValueError, "Fuite de données"):
            read_inventory(self.path, require_split=True, check_files=False, root=self.root)

    def test_csv_round_trip_with_existing_paths(self) -> None:
        rows = assign_splits(inventory())
        # Ici, les fichiers vides servent à vérifier les chemins, pas le décodage vidéo.
        for row in rows:
            video = self.root / row["path"]
            video.parent.mkdir(parents=True, exist_ok=True)
            video.touch()
        self.write_rows(rows)
        self.assertEqual(read_inventory(self.path, require_split=True, root=self.root), rows)

    def test_rejects_missing_file_unless_example_mode(self) -> None:
        self.write_rows(inventory())
        with self.assertRaisesRegex(ValueError, "introuvable"):
            read_inventory(self.path, root=self.root)
        self.assertEqual(len(read_inventory(self.path, check_files=False, root=self.root)), 12)

    def test_rejects_invalid_rows(self) -> None:
        cases = (
            ("label", "fake", "label"),
            ("video_id", "original_0", "dupliqué"),
            ("source_id", "original_0|", "source_id"),
            ("path", "../outside.mp4", "relatif"),
            ("path", "C:/outside.mp4", "relatif"),
            ("path", "data/raw/original_0.mp4", "deux fois"),
        )
        for field, value, message in cases:
            with self.subTest(field=field, value=value):
                rows = inventory()
                rows[1][field] = value
                self.write_rows(rows)
                with self.assertRaisesRegex(ValueError, message):
                    read_inventory(self.path, check_files=False, root=self.root)

    def test_rejects_too_few_groups_or_invalid_ratios(self) -> None:
        with self.assertRaisesRegex(ValueError, "trois groupes"):
            assign_splits(inventory()[:4])
        for ratios in ((0.7, 0.2, 0.2), (1.0, 0.0, 0.0), (float("nan"), 0.1, 0.1)):
            with self.subTest(ratios=ratios), self.assertRaises(ValueError):
                assign_splits(inventory(), ratios=ratios)

    def test_three_groups_still_produce_three_nonempty_splits(self) -> None:
        result = assign_splits(inventory()[:6])
        self.assertEqual({row["split"] for row in result}, {"train", "val", "test"})

    def test_refuses_to_resplit_a_frozen_manifest(self) -> None:
        with self.assertRaisesRegex(ValueError, "déjà découpé"):
            assign_splits(assign_splits(inventory()))

    def test_cli_creates_checks_and_preserves_existing_manifest(self) -> None:
        self.write_rows(inventory())
        output = self.root / "manifest.csv"

        def run(*arguments: str) -> subprocess.CompletedProcess:
            return subprocess.run(
                [sys.executable, "-m", "data.prepare_manifest", *arguments],
                cwd=ROOT, capture_output=True, encoding="utf-8",
                env={**os.environ, "PYTHONUTF8": "1"},
            )

        arguments = ("create", "--input", str(self.path), "--output", str(output), "--allow-missing")
        created = run(*arguments)
        self.assertEqual(created.returncode, 0, created.stderr)
        before = output.read_bytes()
        checked = run("check", "--input", str(output), "--allow-missing")
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertIn("aucune fuite", checked.stdout)
        overwritten = run(*arguments)
        self.assertEqual(overwritten.returncode, 2)
        self.assertEqual(output.read_bytes(), before)
        missing = run("check", "--input", str(output))
        self.assertEqual(missing.returncode, 2)
        self.assertIn("introuvable", missing.stderr)


if __name__ == "__main__":
    unittest.main()
