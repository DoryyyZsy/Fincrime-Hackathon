import csv
from pathlib import Path

from src.clean import clean_record, country_codes
from src.io import RAW, write_rows


def split(raw):
    return list(dict.fromkeys(v.strip() for v in next(csv.reader([raw or ""], delimiter=";")) if v.strip()))


def load(path: Path):
    csv.field_size_limit(16 * 1024 * 1024)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["schema"] != "Person":
                continue
            entity_id = row["id"]
            names = list(dict.fromkeys([row["name"], *split(row["aliases"])]))
            for name in names:
                for dob in split(row["birth_date"]) or [None]:
                    # The bulk export combines citizenship and other affiliations.
                    # Use only an explicit nationality property as identity evidence.
                    clean = clean_record(name, dob, row.get("nationality"), row.get("scoped_id"))
                    if clean["name_clean"]:
                        yield {
                            "record_id": "", "entity_id": entity_id, "caption": row["name"],
                            "name": name, "names": names, "dob_raw": dob,
                            "countries": country_codes(split(row["countries"])),
                            "nationality_raw": row.get("nationality"),
                            "id_raw": row.get("identifiers"), "datasets": split(row["dataset"]),
                            "source_url": row.get("source_url") or f"https://www.opensanctions.org/entities/{entity_id}/",
                            **clean,
                        }


def main():
    rows = list(load(RAW / "sanctions_targets.csv"))
    for index, row in enumerate(rows):
        row["record_id"] = f"E-{index:07d}"
    write_rows("entities", rows)
    print(f"{len(rows):,} alias/date rows")


if __name__ == "__main__":
    main()
