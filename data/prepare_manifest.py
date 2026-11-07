"""Créer et contrôler le découpage des vidéos avant l'extraction des images.

Exemples :
    python -m data.prepare_manifest create --input data/source_videos.csv
    python -m data.prepare_manifest check --input data/manifest.csv
"""

import argparse
import csv
import math
from pathlib import Path, PureWindowsPath
import random


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FIELDS = ("video_id", "source_id", "label", "path")
MANIFEST_FIELDS = ("video_id", "source_id", "label", "split", "path")
SPLITS = ("train", "val", "test")


def source_groups(rows: list[dict[str, str]]) -> list[list[int]]:
    """Regrouper les vidéos reliées directement ou par plusieurs sources.

    Un source_id peut contenir plusieurs identifiants séparés par |.
    Une vidéo manipulée liée à deux sources réunit leurs deux groupes.
    """
    parents: dict[str, str] = {}

    def find(identifier: str) -> str:
        parents.setdefault(identifier, identifier)
        while parents[identifier] != identifier:
            parents[identifier] = parents[parents[identifier]]
            identifier = parents[identifier]
        return identifier

    for row in rows:
        for source in row["source_id"].split("|"):
            parents[find(source)] = find(row["video_id"])

    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(find(row["video_id"]), []).append(index)
    return sorted(groups.values(), key=lambda group: min(rows[i]["video_id"] for i in group))


def read_inventory(
    path: Path,
    *,
    require_split: bool = False,
    check_files: bool = True,
    root: Path = ROOT,
) -> list[dict[str, str]]:
    """Lire le CSV et refuser les entrées ambiguës ou les fuites entre ensembles."""
    root = root.resolve()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = MANIFEST_FIELDS if require_split else SOURCE_FIELDS
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"Colonnes requises : {', '.join(required)}.")
        rows = []
        ids: set[str] = set()
        paths: set[Path] = set()
        for line_number, raw in enumerate(reader, start=2):
            row = {key: (raw.get(key) or "").strip() for key in MANIFEST_FIELDS}
            if any(not row[key] for key in required):
                raise ValueError(f"Ligne {line_number} : un champ obligatoire est vide.")
            if row["label"] not in ("0", "1"):
                raise ValueError(f"Ligne {line_number} : label doit valoir 0 (real) ou 1 (fake).")
            if row["video_id"] in ids or "|" in row["video_id"]:
                raise ValueError(f"Ligne {line_number} : video_id dupliqué ou contenant |.")
            sources = [part.strip() for part in row["source_id"].split("|")]
            if not all(sources) or len(set(sources)) != len(sources):
                raise ValueError(f"Ligne {line_number} : source_id vide ou répété entre les |.")
            row["source_id"] = "|".join(sorted(sources))
            relative = Path(row["path"].replace("\\", "/"))
            resolved = (root / relative).resolve()
            if (
                relative.is_absolute()
                or PureWindowsPath(row["path"]).drive
                or ".." in relative.parts
                or not resolved.is_relative_to(root)
            ):
                raise ValueError(f"Ligne {line_number} : path doit rester relatif au dépôt.")
            if resolved in paths:
                raise ValueError(f"Ligne {line_number} : le même fichier vidéo apparaît deux fois.")
            if check_files and not resolved.is_file():
                raise ValueError(f"Ligne {line_number} : fichier vidéo introuvable : {relative}.")
            row["path"] = relative.as_posix()
            if require_split and row["split"] not in SPLITS:
                raise ValueError(f"Ligne {line_number} : split doit valoir train, val ou test.")
            ids.add(row["video_id"])
            paths.add(resolved)
            rows.append(row)

    if not rows:
        raise ValueError("Le CSV ne contient aucune vidéo.")
    if require_split:
        for group in source_groups(rows):
            if len({rows[i]["split"] for i in group}) != 1:
                raise ValueError("Fuite de données : des vidéos liées sont dans plusieurs ensembles.")
        if {row["split"] for row in rows} != set(SPLITS):
            raise ValueError("Le manifeste doit contenir train, val et test.")
    return rows


def assign_splits(
    rows: list[dict[str, str]],
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> list[dict[str, str]]:
    """Répartir les groupes entiers de façon reproductible, sans modifier l'entrée.

    Les proportions portent sur les groupes, pas sur le nombre de vidéos.
    Chaque ensemble reçoit au moins un groupe ; l'équilibre des classes
    doit ensuite être examiné dans le récapitulatif.
    """
    if len(ratios) != 3 or any(not math.isfinite(r) or r <= 0 for r in ratios):
        raise ValueError("Indiquer trois proportions strictement positives et finies.")
    if not math.isclose(sum(ratios), 1.0):
        raise ValueError("La somme des proportions doit valoir 1.")
    if any(row.get("split") for row in rows):
        raise ValueError("Ce CSV est déjà découpé. Utiliser la commande check pour le contrôler.")
    groups = source_groups(rows)
    if len(groups) < 3:
        raise ValueError("Il faut au moins trois groupes de sources indépendants.")
    random.Random(seed).shuffle(groups)

    quotas = [len(groups) * ratio for ratio in ratios]
    counts = [math.floor(quota) for quota in quotas]
    remainder_order = sorted(range(3), key=lambda i: quotas[i] - counts[i], reverse=True)
    for index in remainder_order[:len(groups) - sum(counts)]:
        counts[index] += 1
    for index in range(3):
        if counts[index] == 0:
            donor = max(range(3), key=lambda i: counts[i])
            counts[donor] -= 1
            counts[index] += 1

    result = [dict(row) for row in rows]
    offset = 0
    for split, count in zip(SPLITS, counts):
        for group in groups[offset:offset + count]:
            for index in group:
                result[index]["split"] = split
        offset += count
    return result


def print_summary(rows: list[dict[str, str]]) -> None:
    """Afficher les effectifs réels et manipulés pour examiner le découpage."""
    print(f"{len(rows)} vidéos, {len(source_groups(rows))} groupes de sources.")
    for split in SPLITS:
        selected = [row for row in rows if row["split"] == split]
        real = sum(row["label"] == "0" for row in selected)
        fake = sum(row["label"] == "1" for row in selected)
        print(f"  {split}: {len(selected)} vidéos (real={real}, fake={fake})")
        if real == 0 or fake == 0:
            print(f"  Attention : {split} ne contient pas les deux classes ; revoir le découpage.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("create", "check"):
        command = commands.add_parser(name)
        command.add_argument("--input", type=Path, required=True, help="CSV à lire.")
        command.add_argument(
            "--allow-missing", action="store_true",
            help="Autoriser les fichiers absents, uniquement pour essayer un exemple fictif.",
        )
        if name == "create":
            command.add_argument("--output", type=Path, default=ROOT / "data/manifest.csv")
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--ratios", nargs=3, type=float, default=(0.7, 0.15, 0.15))
    args = parser.parse_args()
    try:
        rows = read_inventory(
            args.input, require_split=args.command == "check",
            check_files=not args.allow_missing,
        )
        if args.command == "create":
            rows = assign_splits(rows, seed=args.seed, ratios=tuple(args.ratios))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            # Le mode x refuse d'écraser un découpage déjà figé.
            with args.output.open("x", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            print(f"Manifeste créé : {args.output}")
        else:
            print("Manifeste valide : aucune fuite de sources détectée.")
        if args.allow_missing:
            print("Mode exemple : la présence des fichiers vidéo n'a pas été vérifiée.")
        print_summary(rows)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Erreur : {error}\n")


if __name__ == "__main__":
    main()
