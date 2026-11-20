import hashlib
from collections import defaultdict
from datetime import date
from random import Random

from faker import Faker

from src.clean import clean_record
from src.io import PROCESSED, read_rows, write_json, write_rows

LOCALES = {"en_GB": "gb", "ar_AA": "ae", "zh_CN": "cn", "ru_RU": "ru", "es_ES": "es"}


def typo(name):
    # Swap neighboring letters in the longest token.
    parts = name.split()
    index = max(range(len(parts)), key=lambda i: len(parts[i]))
    token = parts[index]
    for j in range(len(token) - 2, 0, -1):
        if token[j] != token[j + 1]:
            parts[index] = token[:j] + token[j + 1] + token[j] + token[j + 2:]
            break
    return " ".join(parts)


def main():
    rng = Random(42)
    Faker.seed(42)
    fake = {locale: Faker(locale) for locale in LOCALES}
    customers, labels = [], []

    def add(name, dob, nationality, occupation, truth=None, planted=None):
        cid = f"C-{len(customers) + 1:05d}"
        customers.append({
            "record_id": cid, "customer_id": cid, "name": name,
            "dob_raw": dob, "nationality_raw": nationality, "id_raw": None,
            "occupation": occupation,
            **clean_record(name, dob, nationality),
        })
        labels.append({
            "customer_id": cid, "is_true_hit": truth is not None,
            "true_entity_id": truth, "planted_case": planted,
        })

    for i in range(5000):
        locale = list(LOCALES)[i % len(LOCALES)]
        generator = fake[locale]
        add(generator.name(), generator.date_between(date(1940, 1, 1), date(2007, 12, 31)).isoformat(),
            LOCALES[locale], generator.job())

    by_entity = defaultdict(list)
    for row in read_rows("entities"):
        by_entity[row["entity_id"]].append(row)
    dated = [
        rows for rows in by_entity.values()
        if any(r["dob"] and r["name"] == r["caption"] and r["first_token"] for r in rows)
    ]
    aliases = [
        rows for rows in dated
        if any(r["dob"] and r["name"] != r["caption"] and len(r["name_tokens"]) >= 3
               for r in rows)
    ]
    disguised = rng.choice(aliases)
    planted_rows = rng.sample([rows for rows in dated if rows is not disguised], 189)
    for i, rows in enumerate([disguised, *planted_rows[:39]]):
        eligible = [r for r in rows if r["dob"] and r["name"] == r["caption"]]
        row = rng.choice(eligible)
        name, dob = row["name"], row["dob"]
        variant = i % 6
        if variant == 0:
            alias = [r for r in rows if r["dob"] == dob and r["name"] != name
                     and len(r["name_tokens"]) >= 3]
            if alias:
                # Leave the alphabetically first token intact for DOB-year blocking.
                row = rng.choice(alias)
                candidate = typo(row["name"])
                name = candidate if clean_record(candidate)["first_token"] == row["first_token"] else "Dr. " + row["name"]
            else:
                name = "Dr. " + name
        elif variant == 1:
            name = " ".join(reversed(name.split()))
        elif variant == 2:
            name = "Dr. " + name
        elif variant == 3:
            dob = None
        elif variant == 4 and len(name.split()) >= 3:
            parts = name.split()
            candidate = " ".join([parts[0], *parts[2:]])
            name = candidate if clean_record(candidate)["first_token"] == row["first_token"] else "Mr. " + name
        else:
            candidate = typo(name)
            name = candidate if clean_record(candidate)["first_token"] == row["first_token"] else "Mr. " + name
        add(name, dob, next(iter(row["nationality"] or []), None),
            fake["en_GB"].job(), row["entity_id"], "disguised" if i == 0 else "true_hit")

    for rows in planted_rows[39:]:
        row = next(r for r in rows if r["dob"] and r["name"] == r["caption"])
        years = [y for y in range(1940, 2008) if all(
            not r["dob_year"] or abs(y - r["dob_year"]) >= 15 for r in rows
        )]
        dob = date(rng.choice(years), rng.randint(1, 12), rng.randint(1, 28)).isoformat()
        excluded = set(row["countries"] or []) | set(row["nationality"] or [])
        country = rng.choice(sorted(set(LOCALES.values()) - excluded) or ["nz"])
        add(row["name"], dob, country, fake["en_GB"].job(), planted="decoy")

    write_rows("customers", customers)
    write_rows("labels", labels)
    write_json("synthetic_manifest", {
        "customers_sha256": hashlib.sha256((PROCESSED / "customers.parquet").read_bytes()).hexdigest(),
    })
    print(f"{len(customers):,} synthetic customers including 40 planted matches and 150 decoys")


if __name__ == "__main__":
    main()
