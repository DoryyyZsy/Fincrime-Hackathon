"""Load a data folder (dev or unseen, same format) and build the lookup indexes."""
import json
import os
from collections import defaultdict

from . import normalize as N

TABLES = ["alerts", "hits", "customers", "accounts", "onboarding", "kyc_checks", "notes", "search_results"]


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class DataRoom:
    """All tables of one data folder plus the joins the SOP walks through."""

    def __init__(self, folder):
        self.folder = folder
        for name in TABLES:
            setattr(self, name, read_jsonl(os.path.join(folder, f"{name}.jsonl")))
        variants = os.path.join(folder, "name_variants.json")
        if os.path.exists(variants):
            N.load_name_variants(variants)
        labels = os.path.join(folder, "practice_labels.jsonl")
        self.practice_labels = read_jsonl(labels) if os.path.exists(labels) else []
        self._register_countries()
        self._index()

    def _register_countries(self):
        names = set()
        for c in self.customers:
            names |= {c.get("nationality"), c.get("residence_country")}
        for k in self.kyc_checks:
            names.add(k.get("document_country"))
        for h in self.hits:
            for field in ("countries", "nationality", "citizenship"):
                names |= set(h.get(field) or [])
        N.register_countries(n for n in names if n)

    def _index(self):
        self.alert_by_id = {a["alert_id"]: a for a in self.alerts}
        self.hit_by_id = {h["hit_id"]: h for h in self.hits}
        self.customer_by_id = {c["record_id"]: c for c in self.customers}
        self.customer_by_email = defaultdict(list)
        for c in self.customers:
            if c.get("email"):
                self.customer_by_email[c["email"].lower()].append(c["record_id"])

        self.holders_by_account = defaultdict(list)  # joint accounts have several holders
        self.customer_by_login = defaultdict(list)
        self.accounts_by_customer = defaultdict(list)
        for a in self.accounts:
            self.holders_by_account[a["account_id"]].append(a["customer_id"])
            if a.get("login_handle"):
                self.customer_by_login[a["login_handle"].lower()].append(a["customer_id"])
            self.accounts_by_customer[a["customer_id"]].append(a["account_id"])

        self.applicants_by_customer = defaultdict(list)
        self.customers_by_applicant = defaultdict(list)
        for o in self.onboarding:
            self.applicants_by_customer[o["customer_id"]].append(o["kyc_applicant_id"])
            self.customers_by_applicant[o["kyc_applicant_id"]].append(o["customer_id"])

        self.checks_by_applicant = defaultdict(list)
        for k in self.kyc_checks:
            self.checks_by_applicant[k["applicant_id"]].append(k)

        self.notes_by_email = defaultdict(list)
        for n in self.notes:
            self.notes_by_email[(n.get("requester_email") or "").lower()].append(n)

        self.search_by_query = defaultdict(list)
        for s in self.search_results:
            self.search_by_query[s["query"]].append(s)

        # blocking index for alerts without a link: name parts (and their variants) -> customers
        self.customers_by_part = defaultdict(set)
        self.customers_by_dob = defaultdict(set)
        for c in self.customers:
            company = c.get("entity_type") == "Company"
            for field in ("full_name", "preferred_name", "login_name"):
                for part in N.name_parts(c.get(field), company):
                    self.customers_by_part[part].add(c["record_id"])
            dob = N.parse_date(c.get("dob"))
            if dob:
                self.customers_by_dob[dob].add(c["record_id"])

        self._build_persons()

    # ------------------------------------------------------------ one person, one picture (principle 3)
    def _build_persons(self):
        parent = {cid: cid for cid in self.customer_by_id}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        by_phone_dob = defaultdict(list)
        for c in self.customers:
            dob = N.parse_date(c.get("dob"))
            phone = "".join(ch for ch in (c.get("phone") or "") if ch.isdigit())
            if phone and dob and all(dob):  # same phone AND the same full date of birth
                by_phone_dob[(phone, dob)].append(c["record_id"])
        by_document = defaultdict(set)
        for k in self.kyc_checks:
            number = (k.get("document_number") or "").replace(" ", "").upper()
            if k.get("result") == "approved" and number:
                for cid in self.customers_by_applicant.get(k["applicant_id"], []):
                    by_document[number].add(cid)
        for group in list(by_phone_dob.values()) + [sorted(g) for g in by_document.values()]:
            for other in group[1:]:
                if other in parent and group[0] in parent:
                    union(group[0], other)

        self.person_of = {cid: find(cid) for cid in parent}
        self.records_of_person = defaultdict(list)
        for cid, root in self.person_of.items():
            self.records_of_person[root].append(cid)
