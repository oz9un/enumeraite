"""Discovery provenance trees for terminal output and a self-contained HTML report."""

from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path


def text_tree(report: dict, limit: int = 100) -> str:
    nodes = [node for node in report["nodes"] if node["status"] == "verified"]
    children: dict[str | None, list[dict]] = defaultdict(list)
    for node in nodes:
        children[node["parent"]].append(node)
    lines = [f"{report['target']} — {len(nodes)} verified"]
    emitted = 0

    def walk(parent: str | None, prefix: str = "") -> None:
        nonlocal emitted
        for index, node in enumerate(children[parent]):
            if emitted >= limit:
                return
            last = index == len(children[parent]) - 1
            marker = "└── " if last else "├── "
            label = node["pattern"] or node["source"]
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
            detail = (
                f"<p>{esc(node['source'])} · round {node['round']} · {esc(node['reason'])}</p>"
                f"<p class='pattern'>{esc(node['pattern'])}</p>"
            )
            rows.append(
                f"<li data-status='{esc(status)}'><details {'open' if status == 'verified' else ''}>"
                f"<summary><span class='badge {esc(status)}'>{esc(status)}</span> "
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
button{{background:#26384a;border:1px solid #486178;color:white;padding:9px 14px;border-radius:6px;cursor:pointer}}
body.verified-only li[data-status]:not([data-status='verified']){{display:none}}
.notice{{border:1px solid #80663b;background:#282218;padding:0 16px;border-radius:8px;margin:18px 0}}
.foot{{margin-top:36px;border-top:1px solid #334253;padding-top:16px}}
</style><body class="verified-only"><main>
<div class="muted">ENUMERAITE / DISCOVERY TREE</div><h1>{esc(report['target'])}</h1>
<p>Edges show which verified hostname led to a hypothesis. Pattern labels are inferences, not proof of naming intent.</p>
<div class="stats"><div><b>{report['verified']}</b>verified</div><div><b>{report['known_verified']}</b>known verified</div>
<div><b>{report['attempted']}</b>attempted</div><div><b>{report['round']}</b>expansion rounds</div></div>
<p>Stopped: {esc(report['stop_reason'])}</p>
{notices}
<button id="filter">Show all attempts</button> <button id="expand">Expand all</button>
{branch(None)}
<details class="foot"><summary>Inferred pattern families</summary><ul>{patterns}</ul></details>
<p class="foot">DNS observations are time-specific. Passive sources may be incomplete. This report contains scan findings; share it deliberately.</p>
</main><script>
document.getElementById('filter').onclick=function(){{document.body.classList.toggle('verified-only');this.textContent=document.body.classList.contains('verified-only')?'Show all attempts':'Show verified only'}};
document.getElementById('expand').onclick=function(){{document.querySelectorAll('details').forEach(d=>d.open=true)}};
</script></body></html>"""
    return body


def write_tree(report: dict, path: Path) -> None:
    path.write_text(render_tree(report), encoding="utf-8")
