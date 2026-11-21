"""Local demo server for the two screens (standard library only, binds to 127.0.0.1).

    python -m screening_ml.demo_server [--data data/screening/dev] [--port 8765]

  /staff.html        staff check: type a customer's details, see each piece of evidence and the verdict
  /management.html   management review of every hit, with an editable hit list
  POST /api/check    run the screening engine on one customer (JSON in, JSON out)
  GET  /api/examples demo presets taken from the development set
  /review.html       decision review: why each hit was decided, analyst / MLRO / QA queues, audit log
  GET  /api/review/queue, GET /api/review/hit?id=H-xxxxxx, POST /api/review/action

The pages show customer data from the data folder: keep the server local and never expose it.
"""
import argparse
import datetime
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import analyst_view
from . import normalize as N
from .compare import decide_case
from .customer import resolve
from .data import DataRoom
from .hit import build_hit
from .review_data import CHECK_LABEL, ReviewBook, fact_rows, plain_summary

HERE = os.path.dirname(__file__)
STAFF_PAGE = os.path.join(HERE, "staff_view.html")
REVIEW_PAGE = os.path.join(HERE, "review_view.html")
MAX_BODY = 20_000
DEMO_HITS = ["H-D861DA", "H-582410", "H-FF3096", "H-02115F", "H-13E1CF"]  # clear, match, review, company match, too young


class Engine:
    def __init__(self, folder, audit_path):
        t0 = time.time()
        self.room = DataRoom(folder)
        self.review = ReviewBook(self.room, self._present, audit_path)
        self.profiles = self._unique_profiles()
        payload = analyst_view.build(folder)
        with open(analyst_view.TEMPLATE, encoding="utf-8") as handle:
            template = handle.read()
        self.management_html = template.replace(
            "/*__DATA__*/null", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"))
        print(f"loaded {folder}: {len(self.room.hits)} hits, {len(self.profiles)} distinct watchlist profiles, "
              f"{len(self.room.customers)} customers ({time.time() - t0:.1f}s)")

    def _unique_profiles(self):
        """The same watchlist profile can appear in several alerts; screen each one once."""
        seen, out = set(), []
        for h in self.room.hits:
            key = ((h.get("name") or "").lower(), h.get("primary_category"), tuple(sorted(h.get("countries") or [])),
                   tuple(h.get("dobs") or []), h.get("entity_type"))
            if key not in seen:
                seen.add(key)
                company = h.get("entity_type") == "Company"
                names = [h.get("name"), h.get("best_matching_name")] + list(h.get("aliases") or [])
                out.append((h, [N.name_parts(n, company) for n in names if n]))
        return out

    def examples(self):
        out = []
        for hid in DEMO_HITS:
            hit = self.room.hit_by_id.get(hid)
            if not hit:
                continue
            a = self.room.alert_by_id[hit["alert_id"]]
            c = a["customer"]
            label = c["name"]
            if any(o["name"] == c["name"] for o in out):  # same name, different customer: say which
                label += f" ({a.get('account_ref') or a.get('login_handle') or a.get('email') or c.get('dob')})"
            out.append({"label": label, "name": c["name"], "entity_type": c.get("entity_type") or "Individual",
                        "dob": c.get("dob") or "", "country": c.get("country") or "", "city": c.get("city") or "",
                        "account_ref": a.get("account_ref") or "", "login_handle": a.get("login_handle") or "",
                        "email": a.get("email") or ""})
        return out

    def screen(self, name, entity_type, limit=8):
        """Watchlist profiles whose names agree with the input name (POLICY principle 8), best first."""
        company = entity_type == "Company"
        parts = N.name_parts(name, company)
        found = []
        for hit, names in self.profiles:
            best = max(((N.name_agreement(parts, p), N.best_similarity(parts, p)) for p in names if p),
                       default=((False, 0, 0.0), 0.0))
            (agree, matched, frac), sim = best
            if agree:
                found.append((-matched, -frac, -sim, hit["hit_id"], hit))
        found.sort(key=lambda x: x[:4])
        return [f[-1] for f in found[:limit]]

    def check(self, form):
        name = (form.get("name") or "").strip()[:200]
        if not name:
            return {"error": "Enter the customer's name."}
        entity = "Company" if form.get("entity_type") == "Company" else "Individual"
        alert = {
            "alert_id": "A-DEMO00", "screened_at": datetime.date.today().isoformat(),
            "customer": {"name": name, "entity_type": entity, "dob": (form.get("dob") or "").strip() or None,
                         "country": (form.get("country") or "").strip() or None,
                         "city": (form.get("city") or "").strip() or None},
            "account_ref": (form.get("account_ref") or "").strip() or None,
            "login_handle": (form.get("login_handle") or "").strip() or None,
            "email": (form.get("email") or "").strip() or None,
        }
        t0 = time.time()
        hits = self.screen(name, entity)
        alert["hit_ids"] = [h["hit_id"] for h in hits]
        res = resolve(self.room, alert)
        results = []
        for raw in hits:
            case = decide_case(res, build_hit(self.room, raw), alert)
            item = self._present(case, raw)
            comp = case.comparisons[0] if case.comparisons else None
            rule = CHECK_LABEL.get(case.teacher_code)
            principle = ("POLICY principle 6: clear only on a contradiction" if case.teacher_decision == "DIFFERENT_ENTITY"
                         else "POLICY principle 7: confirm only on agreement" if case.teacher_decision == "SAME_ENTITY"
                         else "POLICY principle 3: several customers fit" if case.teacher_code == "AMBIGUOUS_CUSTOMER_RECORD"
                         else "POLICY: review when nothing settles it")
            item.update({"summary": plain_summary(case), "facts": fact_rows(case),
                         "matched_name": (comp.matched_hit_name or "").rstrip(". ") if comp else "",
                         "rule": rule or case.teacher_code, "principle": principle})
            results.append(item)
        order = {"SAME_ENTITY": 0, "REVIEW": 1, "DIFFERENT_ENTITY": 2}
        results.sort(key=lambda r: order[r["decision"]])
        decisions = {r["decision"] for r in results}
        overall = ("ALERT" if "SAME_ENTITY" in decisions else "REVIEW" if "REVIEW" in decisions else "CLEAR")
        link = {"account": f"account {alert['account_ref']}", "login": f"login {alert['login_handle']}",
                "email": f"email {alert['email']}"}.get(res.link_type, "a search on the name, date of birth and location")
        return {"overall": overall, "customer": {**self._customer(res), "link": link,
                                                 "candidates": len(res.candidates)}, "hits": results,
                "screened": alert["customer"], "ms": int((time.time() - t0) * 1000),
                "profiles_screened": len(self.profiles)}

    def _customer(self, res):
        if not res.persons:
            return {"found": False, "text": "No customer record fits these details, so the check uses what you entered."}
        if len(res.persons) > 1:
            return {"found": True, "ambiguous": True, "records": [p.primary for p in res.persons],
                    "text": f"{len(res.persons)} different customers fit these details."}
        p = res.persons[0]
        return {"found": True, "ambiguous": False, "record": p.primary, "records": p.records,
                "verified": (p.verified_name.sources if p.verified_name else p.dob.sources if p.dob_verified else [None])[0],
                "dob": N.fmt_date(p.dob.value) if p.dob else None, "countries": sorted(p.countries),
                "name": " ".join(p.verified_name.value).title() if p.verified_name else None}

    @staticmethod
    def _present(case, raw):
        """The simplified evidence list a member of staff sees, in the order it is animated."""
        comp = case.comparisons[0]
        p, h, chk = comp.person, comp.hit, comp.checks
        ev = []

        def add(status, label, detail=""):
            ev.append({"status": status, "label": label, "detail": detail})

        if p.primary:
            add("info", "Customer found", p.primary + (f" + {len(p.records) - 1} linked record(s)" if len(p.records) > 1 else ""))
        else:
            add("unknown", "No customer record", "using the details entered")
        if p.verified_name or p.dob_verified:
            src = (p.verified_name.sources if p.verified_name else p.dob.sources)[0]
            add("info", "ID verified", src)
        if chk["ENTITY_TYPE_MISMATCH"].fired:
            add("mismatch", "Person vs company", "one is a person, the other a company")
        if chk["NAME_MISMATCH_VERIFIED"].fired:
            add("mismatch", "Legal name differs", " ".join(p.verified_name.value).title())
        elif comp.name_agree:
            add("match", "Name matched", (comp.matched_hit_name or "").rstrip(". "))
        else:
            add("unknown", "Name partly matched", (comp.matched_hit_name or "").rstrip(". "))
        cust_dob = N.fmt_date(p.dob.value) if p.dob else None
        hit_dob = " or ".join(N.fmt_date(d) for d in sorted(h.dobs, key=N.date_key))
        if comp.dob_agree:
            add("match", "DOB matched", f"{cust_dob} vs {hit_dob}")
        elif comp.dob_contra:
            add("mismatch", "DOB doesn't match", f"{cust_dob} vs {hit_dob}")
        elif not h.dobs:
            add("unknown", "No DOB on the listing", "")
        elif not p.dob:
            add("unknown", "Customer DOB unknown", "")
        if comp.country_overlap:
            add("match", "Country matched", ", ".join(sorted(comp.country_overlap)))
        elif chk["LOCATION_MISMATCH"].fired:
            add("mismatch", "Country doesn't match", f"{', '.join(sorted(p.countries))} vs {', '.join(sorted(h.countries))}")
        elif p.countries and h.countries:
            add("warn", "Countries differ", "not enough on its own here")
        else:
            add("unknown", "Country not comparable", "")
        if chk["TEMPORAL_IMPOSSIBILITY"].fired:
            add("mismatch", "Too young at the time", f"report dated {N.fmt_date(h.earliest_report)}")
        if chk["DECEASED"].fired:
            add("mismatch", "Listed person has died", N.fmt_date(h.deceased))
        if "uncertain evidence" in case.teacher_reason:
            add("warn", "Held for safety", "a note could not be read reliably")
        if case.teacher_code == "AMBIGUOUS_CUSTOMER_RECORD":
            add("warn", "Several customers fit", "they lead to different answers")
        return {
            "hit_id": case.hit_id, "decision": case.teacher_decision, "code": case.teacher_code,
            "reason": case.teacher_reason, "evidence_ids": case.teacher_evidence, "evidence": ev,
            "watchlist": {"name": raw.get("name"), "category": h.category, "countries": sorted(h.countries),
                          "dobs": [N.fmt_date(d) for d in h.dobs], "position": (raw.get("position_or_allegation") or "")[:160]},
        }


def make_handler(engine):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self.send_response(302)
                query = self.path.partition("?")[2]
                self.send_header("Location", "/review.html" + (f"?{query}" if query else ""))
                self.end_headers()
            elif path == "/staff.html":
                with open(STAFF_PAGE, encoding="utf-8") as handle:
                    self._send(200, handle.read(), "text/html; charset=utf-8")
            elif path in ("/management.html", "/analyst_view.html"):
                self._send(200, engine.management_html, "text/html; charset=utf-8")
            elif path == "/review.html":
                with open(REVIEW_PAGE, encoding="utf-8") as handle:
                    self._send(200, handle.read(), "text/html; charset=utf-8")
            elif path == "/api/review/alerts":
                self._send(200, json.dumps(engine.review.alerts(), ensure_ascii=False), "application/json")
            elif path == "/api/review/queue":
                self._send(200, json.dumps(engine.review.queue(), ensure_ascii=False), "application/json")
            elif path == "/api/review/hit":
                from urllib.parse import parse_qs, urlparse
                hid = (parse_qs(urlparse(self.path).query).get("id") or [""])[0]
                found = engine.review.hit(hid)
                self._send(200 if found else 404, json.dumps(found or {"error": "unknown hit"}, ensure_ascii=False),
                           "application/json")
            elif path == "/api/examples":
                self._send(200, json.dumps(engine.examples(), ensure_ascii=False), "application/json")
            else:
                self._send(404, "not found", "text/plain")

        def do_POST(self):
            if self.path not in ("/api/check", "/api/review/action"):
                return self._send(404, "not found", "text/plain")
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY:
                return self._send(400, json.dumps({"error": "bad request"}), "application/json")
            try:
                form = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(form, dict):
                    raise ValueError("expected an object")
                result = engine.check(form) if self.path == "/api/check" else engine.review.act(form)
            except Exception as error:  # never crash the demo
                result = {"error": f"Could not run the check ({type(error).__name__})."}
            self._send(200, json.dumps(result, ensure_ascii=False), "application/json")

        def log_message(self, fmt, *args):
            sys.stderr.write("%s\n" % (fmt % args))

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data/screening/dev")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--audit", default="data/screening/review/audit_log.jsonl",
                    help="append-only log of reviewer actions (holds customer IDs: keep it out of git)")
    args = ap.parse_args(argv)
    engine = Engine(args.data, args.audit)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(engine))
    print(f"open http://127.0.0.1:{args.port}/  (screening review)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
