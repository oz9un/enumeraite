"""Discovery provenance trees for terminal output and a self-contained HTML report."""

from __future__ import annotations

import html
import json
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlencode

from ..core.origin_evidence import passive_collectors

DNS_STATUS_LABELS = {
    "verified": "DNS resolved",
    "pending": "DNS queued",
    "miss": "No DNS address",
    "wildcard": "Wildcard response",
    "inconclusive": "DNS inconclusive",
}


def _source_reference(source: str, name: str, record: dict) -> str:
    certificate = record.get("certificate_id")
    if (
        source == "crt.sh"
        and isinstance(certificate, str)
        and re.fullmatch(r"[0-9]{1,32}", certificate)
    ):
        url = "https://crt.sh/?" + urlencode({"id": certificate})
        label = "Open recorded certificate"
    elif source == "crt.sh":
        url = "https://crt.sh/?" + urlencode({"q": name})
        label = "Look up this hostname now (not an archived response)"
    else:
        return ""
    return f'<p><a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{label}</a></p>'


def _origin_details(node: dict, report: dict) -> str:
    origin = node.get("origin_evidence")
    origin = origin if isinstance(origin, dict) else {}

    def details(rows: list[tuple[str, object]]) -> str:
        return (
            "<dl>"
            + "".join(
                f"<dt>{html.escape(label)}</dt><dd>{html.escape(str(value))}</dd>"
                for label, value in rows
            )
            + "</dl>"
        )

    rows = [
        (
            "First queued by",
            (
                "Prior scan candidate"
                if node.get("source") == "history"
                else node.get("source", "Not recorded")
            ),
        ),
        ("Why this candidate", node.get("reason") or "Not recorded"),
        ("Queued at (UTC)", origin.get("queued_at", "Not recorded")),
        ("Discovery parent", node.get("parent") or "Target / initial discovery"),
        ("Discovery round", node.get("round", "Not recorded")),
        ("Naming pattern", node.get("pattern") or "Not recorded"),
    ]
    labels = {
        "provider": "Generation provider",
        "model": "Generation model",
        "focus": "Generation focus",
        "batch": "Generation batch",
        "plan_id": "Plan ID",
        "plan_description": "Plan description",
        "numeric_ceiling": "Numeric alternative ceiling",
        "layout": "Witnessed vocabulary layout",
        "word": "Proposed vocabulary",
    }
    rows.extend((label, origin[field]) for field, label in labels.items() if field in origin)
    if "confidence" in origin:
        rows.append(("Proposal/ranking score (not DNS probability)", origin["confidence"]))
    if origin.get("evidence"):
        rows.append(("Verified seed / plan evidence", ", ".join(origin["evidence"])))
    elif node.get("source") in {"model", "pattern", "numeric"}:
        rows.append(("Verified seed / plan evidence", "Not recorded for this candidate"))
    content = "<h4>Discovery origin</h4>" + details(rows)
    if node.get("source") == "model":
        content += "<p>AI proposed this name; only the separate DNS observation can verify address resolution.</p>"
    if node.get("source") == "history":
        content += "<p>Earlier scans verified this name; only this scan's separate DNS observation establishes its current result.</p>"
    collectors = passive_collectors(node, report.get("passive_sources", {}))
    if not collectors and node.get("source") == "passive":
        content += "<p>Collector identity and source-record details were not recorded.</p>"
    records = origin.get("passive", [])
    record_labels = {
        "query": "Recorded source query",
        "matched_name": "Source-returned name",
        "collected_at": "Collected at (UTC; not historical first-seen)",
        "certificate_id": "Certificate ID",
        "issuer": "Certificate issuer",
        "subject_common_name": "Certificate subject common name",
        "first_seen": "Provider first seen (not this scan's DNS check)",
        "entry_timestamp": "Certificate log entry timestamp",
        "not_before": "Certificate validity starts",
        "not_after": "Certificate validity ends",
        "address": "Historical source IP (not this scan's DNS answer)",
    }
    for collector in collectors:
        content += f"<h5>Observed by {html.escape(collector)}</h5>"
        samples = [
            record
            for record in records
            if isinstance(record, dict) and record.get("source") == collector
        ]
        if samples:
            for record in samples:
                content += details(
                    [
                        (label, record[field])
                        for field, label in record_labels.items()
                        if field in record
                    ]
                )
                content += _source_reference(collector, node["value"], record)
        else:
            content += "<p>Collector membership is saved, but original queries, collection times, and source records were not recorded.</p>"
            content += _source_reference(collector, node["value"], {})
        if collector == "subfinder":
            content += "<p>The local subfinder output did not report its upstream provider.</p>"
    if collectors:
        content += "<p>These are bounded source-record samples (up to three per collector), not complete historical coverage. Public-source records do not establish current DNS resolution.</p>"
    return content


def _dns_details(node: dict) -> str:
    """Describe only saved observations, including older scans with none."""
    evidence = node.get("dns_evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    records = evidence.get("records", {})
    records = records if isinstance(records, dict) else {}
    statuses = {
        "answered": "Address returned",
        "no_answer": "No address returned for this record type",
        "nxdomain": "DNS reports the name does not exist",
        "error": "Lookup failed or timed out",
        "not_checked": "Not checked",
    }
    rows = [
        ("Web availability", "Not checked — no HTTP/HTTPS or TLS checks"),
        ("Last DNS check (UTC)", evidence.get("checked_at", "Not recorded")),
        ("Completed DNS attempts", node.get("attempts", "Not recorded")),
        ("Saved addresses", ", ".join(node.get("addresses", [])) or "None recorded"),
    ]
    if evidence.get("aliases"):
        rows.append(("Observed CNAME targets", ", ".join(evidence["aliases"])))
    if "duration_ms" in evidence:
        rows.append(("Candidate DNS lookup duration", f"{evidence['duration_ms']} ms"))
    for kind, label in (("A", "IPv4 (A)"), ("AAAA", "IPv6 (AAAA)")):
        record = records.get(kind)
        if not isinstance(record, dict):
            rows.append((label, "Individual lookup details not recorded"))
            continue
        detail = statuses.get(record.get("status", ""), "Not recorded")
        if record.get("addresses"):
            detail += ": " + ", ".join(record["addresses"])
        if record.get("error"):
            detail += " — " + str(record["error"])
        rows.append((label, detail))
    wildcard = evidence.get("wildcard", {})
    wildcard = wildcard if isinstance(wildcard, dict) else {}
    wildcard_labels = {
        "not_checked": "Not checked for this candidate",
        "not_detected": "Not detected in sampled sibling names",
        "matched": "Candidate filtered as a wildcard response",
        "different_answer": "Wildcard addresses found; candidate returned a different address",
        "inconclusive": "Wildcard calibration was inconclusive",
    }
    rows.append(("Wildcard check", wildcard_labels.get(wildcard.get("status", ""), "Not recorded")))
    if wildcard.get("parent"):
        rows.append(("Wildcard parent", wildcard["parent"]))
    content = "".join(
        f"<dt>{html.escape(str(label))}</dt><dd>{html.escape(str(value))}</dd>"
        for label, value in rows
    )
    note = (
        "Two random sibling names are sampled per parent; calibration may be reused during "
        "the scan. This is a heuristic, not proof of a distinct working service."
        if wildcard.get("status") not in (None, "not_checked")
        else "Missing observation details are not reconstructed from saved addresses."
    )
    return f"<dl>{content}</dl><p>{note}</p>"


def text_tree(report: dict, limit: int = 100) -> str:
    nodes = [node for node in report["nodes"] if node["status"] == "verified"]
    children: dict[str | None, list[dict]] = defaultdict(list)
    for node in nodes:
        children[node["parent"]].append(node)
    lines = [f"{report['target']} — {len(nodes)} DNS resolved (web not checked)"]
    emitted = 0

    def walk(parent: str | None, prefix: str = "") -> None:
        nonlocal emitted
        for index, node in enumerate(children[parent]):
            if emitted >= limit:
                return
            last = index == len(children[parent]) - 1
            marker = "└── " if last else "├── "
            label = node["pattern"] or node["source"]
            if label == "history":
                label = "Prior scan candidate"
            collectors = passive_collectors(node, report.get("passive_sources", {}))
            if collectors:
                label += "; observed by " + ", ".join(collectors)
            lines.append(f"{prefix}{marker}{node['value']} [{label}]")
            emitted += 1
            walk(node["value"], prefix + ("    " if last else "│   "))

    walk(None)
    if emitted < len(nodes):
        lines.append(f"… {len(nodes) - emitted} more; use --tree-file for the full tree")
    return "\n".join(lines)


def render_tree(report: dict) -> str:
    """All dynamic content is escaped; no network resources or embedded credentials."""
    esc = lambda value: html.escape(str(value), quote=True)
    children: dict[str | None, list[dict]] = defaultdict(list)
    for node in report["nodes"]:
        children[node["parent"]].append(node)

    def branch(parent: str | None) -> str:
        rows = []
        for node in children[parent]:
            status = node["status"]
            descendants = branch(node["value"])
            detail = _origin_details(node, report) + "<h4>DNS observation</h4>" + _dns_details(node)
            rows.append(
                f"<li data-status='{esc(status)}'><details {'open' if status == 'verified' else ''}>"
                f"<summary><span class='badge {esc(status)}'>"
                f"{esc(DNS_STATUS_LABELS.get(status, status))}</span> "
                f"<strong>{esc(node['value'])}</strong></summary>{detail}{descendants}</details></li>"
            )
        return "<ul>" + "".join(rows) + "</ul>" if rows else ""

    patterns = "".join(
        "<li><code>"
        + esc(item.get("pattern", ""))
        + "</code><pre>"
        + esc(json.dumps(item, indent=2, ensure_ascii=False))
        + "</pre></li>"
        for item in report.get("patterns", [])
    )
    notices = "".join(f"<p>{esc(warning)}</p>" for warning in report.get("warnings", []))
    unresolved = int(report.get("inconclusive", 0))
    if unresolved:
        notices += f"<p>{unresolved} DNS checks were inconclusive; results are incomplete.</p>"
    notices = (
        f"<aside class='notice' aria-label='Scan limitations'>{notices}</aside>" if notices else ""
    )
    body = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'">
<title>Enumeraite — {esc(report['target'])}</title>
<style>
:root{{color-scheme:dark}}body{{font:16px system-ui;background:#10151d;color:#e5eaf0;margin:0}}
main{{max-width:1100px;margin:auto;padding:36px 24px}}h1{{font-size:32px;margin-bottom:8px}}
.muted,p{{color:#aeb9c7}}.stats{{display:flex;gap:24px;flex-wrap:wrap;margin:24px 0}}
.stats b{{display:block;font-size:28px;color:#76e1b3}}ul{{list-style:none;padding-left:24px}}
li{{margin:8px 0;border-left:1px solid #334253;padding-left:14px}}summary{{cursor:pointer;overflow-wrap:anywhere}}
.badge{{font-size:11px;text-transform:uppercase;background:#293545;border-radius:5px;padding:4px 7px}}
.verified{{background:#174b37;color:#92e8be}}.miss,.wildcard{{color:#efaaaa}}
.pattern,code,pre{{font:13px ui-monospace,monospace;overflow-wrap:anywhere;white-space:pre-wrap}}
dl{{display:grid;grid-template-columns:minmax(0,10rem) minmax(0,1fr);gap:8px 16px;font-size:14px}}
dt{{color:#aeb9c7}}dd{{margin:0;overflow-wrap:anywhere}}
button{{background:#26384a;border:1px solid #486178;color:white;padding:9px 14px;border-radius:6px;cursor:pointer}}
body.verified-only li[data-status]:not([data-status='verified']){{display:none}}
.notice{{border:1px solid #80663b;background:#282218;padding:0 16px;border-radius:8px;margin:18px 0}}
.foot{{margin-top:36px;border-top:1px solid #334253;padding-top:16px}}
</style><body class="verified-only"><main>
<div class="muted">ENUMERAITE / DISCOVERY TREE</div><h1>{esc(report['target'])}</h1>
<p>DNS resolved means an address lookup passed wildcard filtering, not that a website works. Web availability is not checked.</p>
<p>Edges show which DNS finding led to a hypothesis. Pattern labels are inferences, not proof of naming intent.</p>
<div class="stats"><div><b>{report['verified']}</b>DNS resolved</div><div><b>{report['known_verified']}</b>known DNS resolved</div>
<div><b>{report['attempted']}</b>attempted</div><div><b>{report['round']}</b>expansion rounds</div></div>
<p>Stopped: {esc(report['stop_reason'])}</p>
{notices}
<button id="filter">Show all attempts</button> <button id="expand">Expand all</button>
{branch(None)}
<details class="foot"><summary>Inferred pattern families</summary><ul>{patterns}</ul></details>
<p class="foot">DNS observations are time-specific, not continuous availability monitoring. Passive sources may be incomplete. This report contains scan findings; share it deliberately.</p>
</main><script>
document.getElementById('filter').onclick=function(){{document.body.classList.toggle('verified-only');this.textContent=document.body.classList.contains('verified-only')?'Show all attempts':'Show DNS resolved only'}};
document.getElementById('expand').onclick=function(){{document.querySelectorAll('details').forEach(d=>d.open=true)}};
</script></body></html>"""
    return body


def write_tree(report: dict, path: Path) -> None:
    path.write_text(render_tree(report), encoding="utf-8")
