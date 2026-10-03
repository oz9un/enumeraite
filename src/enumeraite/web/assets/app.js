"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const svgNS = "http://www.w3.org/2000/svg";
  const state = {
    scan: null,
    scans: [],
    selectedId: null,
    selectedNode: null,
    credentials: {},
    features: {},
    polling: false,
    busy: false,
    page: 0,
    lastAnnounced: "",
    graphKey: "",
    graphBounds: null,
    graphPositions: new Map(),
    graphTransform: { x: 35, y: 35, scale: 1 },
    graphInitialized: false,
    findingsKey: "",
    inputPreview: null,
    inputError: "",
    inputErrorKind: "",
    previewSignature: "",
    resolveSequence: 0,
    resolving: false,
  };
  const activeStatuses = new Set(["running", "pausing"]);
  const graphLimit = 300;
  const pageSize = 50;
  const providers = {
    minimax: {
      name: "MiniMax",
      models: [
        ["MiniMax-M2.7", "MiniMax M2.7"],
        ["MiniMax-M3", "MiniMax M3"],
      ],
    },
    openai: {
      name: "OpenAI",
      models: [
        ["gpt-4.1-mini", "GPT-4.1 mini"],
        ["gpt-4.1", "GPT-4.1"],
      ],
    },
    anthropic: {
      name: "Anthropic",
      models: [["claude-sonnet-4-6", "Claude Sonnet 4.6"]],
    },
  };
  let pollTimer;
  let drag;
  let resolveTimer;

  function element(tag, className, content) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (content !== undefined) node.textContent = String(content);
    return node;
  }
  function svg(tag, attrs = {}, content) {
    const node = document.createElementNS(svgNS, tag);
    Object.entries(attrs).forEach(([key, value]) =>
      node.setAttribute(key, String(value)),
    );
    if (content !== undefined) node.textContent = String(content);
    return node;
  }
  function number(value, fallback = 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }
  function count(value) {
    return Math.max(0, number(value)).toLocaleString();
  }
  function money(value) {
    return (
      "$" +
      number(value).toFixed(number(value) > 0 && number(value) < 0.01 ? 4 : 2)
    );
  }
  function elapsed(seconds) {
    seconds = Math.max(0, Math.floor(number(seconds)));
    return seconds >= 3600
      ? `${Math.floor(seconds / 3600)}:${String(Math.floor(seconds / 60) % 60).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`
      : `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
  }
  function readable(value) {
    return String(value || "").replace(/[_-]/g, " ");
  }
  function notice(message, success = false) {
    $("notice-text").textContent = message;
    $("notice").className = success ? "success" : "";
    $("notice").hidden = false;
  }
  function setConnection(connected, message) {
    $("connection").className =
      "connection " + (connected ? "connected" : "error");
    $("connection").lastElementChild.textContent = message;
  }
  function statusBadge(status) {
    return element(
      "span",
      "status-label " + String(status || "pending").replace(/[^a-z]/g, ""),
      readable(status || "pending"),
    );
  }
  function activeGlobal() {
    return state.scans.some((scan) => activeStatuses.has(scan.status));
  }
  function providerName(scan) {
    return typeof scan.provider === "object"
      ? scan.provider?.name || "none"
      : scan.provider || "none";
  }
  function predictionProvider(scan) {
    const predictor = scan?.summary?.predictor;
    return predictor && typeof predictor === "object" ? predictor.name : null;
  }
  function predictionStartSettings() {
    const enabled = $("prediction-provider").value === "jev";
    if (!enabled) return { prediction_provider: "none" };
    if (!state.credentials.jev)
      throw new Error(
        "Save your separate TypeSafe API key for this session, or turn the Jev prediction layer off.",
      );
    const budget = Number($("prediction-budget").value);
    if (
      !Number.isFinite(budget) ||
      budget < 0 ||
      !$("prediction-budget").value.trim()
    )
      throw new Error("Enter a valid additional Jev prediction budget in USD.");
    return { prediction_provider: "jev", prediction_budget_usd: budget };
  }

  const launchParams = new URLSearchParams(location.hash.slice(1));
  if (launchParams.has("token")) {
    history.replaceState(null, "", location.pathname + location.search);
  }
  try {
    sessionStorage.removeItem("enumeraite.ui.token");
  } catch (_) {
    /* Storage is optional; the dashboard works without it. */
  }

  async function api(path, options = {}) {
    let response;
    try {
      response = await fetch(path, {
        cache: "no-store",
        ...options,
        headers: {
          ...(options.body ? { "Content-Type": "application/json" } : {}),
          ...options.headers,
        },
      });
    } catch (_) {
      throw new Error(
        "Cannot reach the local dashboard server. Make sure Enumeraite UI is running, then reopen its localhost address.",
      );
    }
    if (!response.ok) {
      let message = "Request failed (" + response.status + ").";
      try {
        const body = await response.json();
        if (typeof body.error === "string") message = body.error;
      } catch (_) {
        /* An unavailable server may return a non-JSON error. */
      }
      if (response.status === 401 || response.status === 403)
        message =
          "Local access was rejected. Open the dashboard directly at the localhost address printed by Enumeraite.";
      const error = new Error(message);
      error.status = response.status;
      throw error;
    }
    return response;
  }
  async function json(path, body) {
    return (
      await api(
        path,
        body === undefined
          ? {}
          : { method: "POST", body: JSON.stringify(body) },
      )
    ).json();
  }

  function discoveryInput() {
    const input = {
      target: $("target").value.trim(),
      seeds: $("starting-seeds")
        .value.split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean),
    };
    const scope = $("discovery-scope").value.trim();
    if (scope) input.scope = scope;
    return input;
  }
  function resolvedPassive(preview) {
    return $("passive").value === "auto"
      ? Boolean(preview.passive_default)
      : $("passive").value === "collect";
  }
  function previewIsCurrent() {
    return Boolean(
      state.inputPreview &&
      !state.resolving &&
      state.previewSignature === JSON.stringify(discoveryInput()),
    );
  }
  function renderInputPreview() {
    const preview = state.inputPreview;
    const panel = $("scope-preview");
    panel.className =
      "scope-preview" +
      (state.inputError ? " preview-error" : preview ? " preview-ready" : "");
    $("scope-preview-status").textContent = state.resolving
      ? "CHECKING SCOPE…"
      : state.inputError
        ? state.inputErrorKind === "validation"
          ? "CHECK YOUR INPUT"
          : "SCOPE PREVIEW UNAVAILABLE"
        : preview
          ? "REVIEW BEFORE STARTING"
          : "SCOPE PREVIEW";
    $("scope-preview-target").textContent =
      state.inputError ||
      (preview
        ? "Scope: " + preview.target
        : state.resolving
          ? "Resolving your discovery scope locally…"
          : "Enter a domain or known hostname.");
    $("scope-preview-seeds").hidden = !preview || Boolean(state.inputError);
    $("scope-preview-sources").hidden = !preview || Boolean(state.inputError);
    if (preview) {
      const seeds = preview.seeds || [];
      $("scope-preview-seeds").textContent = seeds.length
        ? "Start from: " +
          seeds.slice(0, 2).join(", ") +
          (seeds.length > 2 ? " + " + (seeds.length - 2) + " more" : "")
        : resolvedPassive(preview)
          ? "Start from: known-host discovery"
          : "Start from: built-in candidates";
      $("scope-preview-seeds").title = seeds.join("\n");
      $("scope-preview-sources").textContent = resolvedPassive(preview)
        ? "Public sources: collect known hosts first"
        : "Public sources: skipped";
    }
    $("scope-preview-note").textContent =
      preview && !state.inputError
        ? "This preview does not scan. Confirm you’re authorized to assess " +
          preview.target +
          " and its subdomains."
        : "This preview does not scan. Only assess domains and subdomains you’re authorized to scan.";
    renderHistory();
  }
  async function resolveInput() {
    clearTimeout(resolveTimer);
    const input = discoveryInput();
    const signature = JSON.stringify(input);
    const sequence = ++state.resolveSequence;
    if (!input.target) {
      state.inputPreview = null;
      state.inputError = "";
      state.inputErrorKind = "";
      state.resolving = false;
      renderInputPreview();
      return null;
    }
    state.resolving = true;
    state.inputError = "";
    state.inputErrorKind = "";
    renderInputPreview();
    try {
      const preview = await json("/api/resolve", input);
      if (
        sequence !== state.resolveSequence ||
        signature !== JSON.stringify(discoveryInput())
      )
        throw new Error(
          "Your input changed. Review the updated scope before starting.",
        );
      if (
        typeof preview.target !== "string" ||
        !Array.isArray(preview.seeds) ||
        typeof preview.passive_default !== "boolean"
      )
        throw new Error(
          "The server could not resolve this scope. Please retry.",
        );
      state.inputPreview = preview;
      state.previewSignature = signature;
      return preview;
    } catch (error) {
      if (sequence === state.resolveSequence) {
        state.inputPreview = null;
        state.previewSignature = "";
        state.inputError = error.message;
        state.inputErrorKind =
          error.status === 400 ? "validation" : "unavailable";
      }
      throw error;
    } finally {
      if (sequence === state.resolveSequence) {
        state.resolving = false;
        renderInputPreview();
      }
    }
  }
  function scheduleResolve(delay = 350) {
    clearTimeout(resolveTimer);
    state.resolveSequence++;
    state.inputPreview = null;
    state.previewSignature = "";
    state.inputError = "";
    state.inputErrorKind = "";
    state.resolving = Boolean($("target").value.trim());
    renderInputPreview();
    if (state.resolving)
      resolveTimer = setTimeout(() => resolveInput().catch(() => {}), delay);
  }

  function renderCredentials(credentials) {
    if (credentials) state.credentials = credentials;
    const configured = Boolean(state.credentials[$("provider").value]);
    $("credential-status").textContent = configured ? "READY" : "NOT SET";
    $("credential-status").className =
      "small-status" + (configured ? " configured" : "");
    $("clear-key").disabled = !configured;
    const predictionConfigured = Boolean(state.credentials.jev);
    $("prediction-credential-status").textContent = predictionConfigured
      ? "READY"
      : "NOT SET";
    $("prediction-credential-status").className =
      "small-status" + (predictionConfigured ? " configured" : "");
    $("clear-prediction-key").disabled = !predictionConfigured;
  }
  function renderHistory() {
    const history = $("scan-history");
    const signature = state.scans
      .map((scan) => `${scan.id}:${scan.target}:${scan.status}`)
      .join("|");
    if (history.dataset.signature !== signature) {
      history.replaceChildren();
      if (!state.scans.length)
        history.append(element("option", "", "No scans yet"));
      for (const scan of state.scans) {
        const option = element(
          "option",
          "",
          `${scan.target} · ${readable(scan.status)}`,
        );
        option.value = scan.id;
        history.append(option);
      }
      history.dataset.signature = signature;
    }
    if (state.selectedId) history.value = state.selectedId;
    $("start").disabled = state.busy || activeGlobal() || !previewIsCurrent();
    $("start").title = activeGlobal()
      ? "Pause the running scan before starting another."
      : !previewIsCurrent()
        ? "Enter your target and review the discovery scope first."
        : "";
  }
  function phaseIndex(scan) {
    if (!scan) return -1;
    if (["completed", "error"].includes(scan.status)) return 3;
    const phase = String(scan.phase || "").toLowerCase();
    if (/passive|known|collect|initial/.test(phase)) return 0;
    if (/expand|pattern|model|ai|learn|predict|rank/.test(phase)) return 2;
    if (/complete|finish|done/.test(phase)) return 3;
    if (/dns|verif|resolv|validat/.test(phase)) return 1;
    return number(scan.summary?.round) > 0 ? 2 : 0;
  }
  function renderSummary() {
    const scan = state.scan;
    const summary = scan?.summary || {};
    const verified = number(summary.verified);
    const known = number(summary.known_verified);
    const ledger = summary.ledger || scan?.ledger;
    const active = Array.isArray(scan?.active) ? scan.active : [];
    $("scan-target").textContent = scan?.target || "Map what’s out there.";
    $("scan-description").textContent = scan
      ? `${providerName(scan) === "none" ? (predictionProvider(scan) ? "Pattern discovery + Jev ranking" : "Free discovery") : (providers[providerName(scan)]?.name || "Hosted AI") + "-assisted discovery"} · ${scan.model || scan.provider?.model || (scan.options?.passive === false ? "Evidence-led recursive patterns" : "Passive sources + recursive patterns")} · ${scan.status === "running" ? "Live evidence, as it happens." : "Your progress is saved locally."}`
      : "Start with a domain or known host. Follow the patterns. Watch discoveries take shape.";
    $("scan-status").textContent = scan
      ? readable(scan.status).toUpperCase()
      : "READY TO DISCOVER";
    $("scan-status").className = "status-pill " + (scan?.status || "");
    $("pause").hidden = !scan || !activeStatuses.has(scan.status);
    $("pause").disabled = state.busy || scan?.status === "pausing";
    $("pause").textContent =
      scan?.status === "pausing" ? "Pausing…" : "Pause scan";
    $("resume").hidden =
      !scan || !scan.can_resume || activeStatuses.has(scan.status);
    $("resume").disabled = state.busy || activeGlobal();
    $("export").disabled = !scan || state.busy;
    $("metric-verified").textContent = count(verified);
    $("metric-known").textContent = count(known);
    $("metric-new").textContent = count(Math.max(0, verified - known));
    $("metric-attempted").textContent = count(summary.attempted);
    $("metric-queued").textContent = count(summary.queued);
    $("metric-active").replaceChildren(
      document.createTextNode(count(active.length)),
      element("span", "metric-unit", "queries"),
    );
    $("metric-round").textContent = count(summary.round);
    $("metric-spent").textContent = money(
      ledger?.spent_usd ?? ledger?.spent ?? 0,
    );
    const cap = ledger?.limit_usd ?? ledger?.limit;
    $("metric-budget").textContent = ledger
      ? `${cap !== undefined ? money(cap) + " model cap" : "Estimated model usage"}${number(ledger.reserved_usd ?? ledger.reserved) > 0 ? " · reserved " + money(ledger.reserved_usd ?? ledger.reserved) : ""}`
      : "No generation charges";
    const predictionLedger = summary.prediction_ledger;
    const predictionCap =
      predictionLedger?.limit_usd ?? predictionLedger?.limit;
    const predictionReserved = number(
      predictionLedger?.reserved_usd ?? predictionLedger?.reserved,
    );
    $("prediction-spend-panel").hidden = !predictionProvider(scan);
    $("prediction-model").textContent =
      summary.predictor?.model || "jev-1.13.0";
    $("prediction-spent").textContent = money(
      predictionLedger?.spent_usd ?? predictionLedger?.spent,
    );
    $("prediction-cap").textContent =
      predictionCap !== undefined
        ? "of " + money(predictionCap) + " prediction cap"
        : "Prediction budget pending";
    $("prediction-reserved").textContent =
      money(predictionReserved) + " reserved";
    $("prediction-reserved").hidden = predictionReserved <= 0;
    $("metric-time").textContent = elapsed(summary.elapsed_seconds);
    $("metric-stop").textContent = scan
      ? scan.status === "running"
        ? `${elapsed(scan.options?.time_limit)} limit per run`
        : scan.status === "pausing"
          ? "Finishing in-flight work"
          : readable(summary.stop_reason || scan.status)
      : "Waiting for a target";
    const current = phaseIndex(scan);
    document.querySelectorAll(".phase-stepper>div").forEach((node, index) => {
      node.classList.toggle("current", index === current);
      node.classList.toggle("done", index < current);
      node.setAttribute("aria-current", index === current ? "step" : "false");
    });
    const warnings = [
      ...(Array.isArray(summary.warnings) ? summary.warnings : []),
    ];
    if (scan?.error) warnings.push(scan.error);
    $("warnings-panel").hidden = !warnings.length;
    $("warnings-list").replaceChildren(
      ...warnings.map((warning) =>
        element(
          "li",
          "",
          typeof warning === "string" ? warning : JSON.stringify(warning),
        ),
      ),
    );
    const announcement = scan ? `${scan.status}:${verified}` : "";
    if (announcement !== state.lastAnnounced) {
      $("live-announcement").textContent = scan
        ? `${scan.target}: ${readable(scan.status)}, ${verified} verified hosts.`
        : "Ready to start a scan.";
      state.lastAnnounced = announcement;
    }
  }

  function renderWorkers() {
    const scan = state.scan;
    const active = Array.isArray(scan?.active) ? scan.active : [];
    $("worker-count").textContent = count(active.length);
    $("worker-state").textContent = active.length
      ? "LIVE QUERIES"
      : scan?.status === "running"
        ? "BETWEEN QUERIES"
        : "IDLE";
    const list = $("worker-list");
    const scroll = list.scrollTop;
    list.replaceChildren();
    for (const work of active.slice(0, 100)) {
      const row = element("div", "worker-row");
      const text = element("div");
      const host = element("div", "worker-host", work.hostname || "DNS lookup");
      host.title = work.hostname || "";
      const seconds = work.started_at
        ? Math.max(0, Math.floor(Date.now() / 1000 - number(work.started_at)))
        : 0;
      text.append(
        host,
        element(
          "div",
          "worker-meta",
          (work.probe ? "Wildcard calibration probe" : "Verifying candidate") +
            (seconds > 0 ? ` · ${seconds}s` : ""),
        ),
      );
      row.append(
        element("span", "worker-pulse"),
        text,
        element("span", "worker-type", work.query_type || "DNS"),
      );
      list.append(row);
    }
    if (!active.length) {
      const empty = element("div", "compact-empty");
      const phase = String(scan?.phase || "");
      let title = "Nothing in flight yet",
        caption = "Live DNS queries will appear here.";
      if (scan?.status === "running") {
        title = /passive|known|collect/.test(phase)
          ? "Collecting known hosts"
          : /predict|rank/i.test(phase)
            ? "Ranking discovery plans"
            : /model|ai/.test(phase)
              ? "Generating AI candidates"
              : "Preparing the next checks";
        caption = "No DNS queries are in flight at this instant.";
      } else if (scan?.status === "paused" || scan?.status === "pausing") {
        title =
          scan.status === "pausing"
            ? "Finishing current work"
            : "Discovery is paused";
        caption = "Resume when you’re ready to follow the next clues.";
      } else if (scan) {
        title = "All workers are idle";
        caption = "Review discoveries and patterns below.";
      }
      empty.append(
        element("span", "radar-symbol", "◎"),
        element("p", "", title),
        element("span", "", caption),
      );
      list.append(empty);
    }
    list.scrollTop = scroll;
    $("worker-footer-text").textContent = scan
      ? `Up to ${number(scan.options?.workers, 20)} workers · ${number(scan.options?.rate, 50)} DNS queries/sec shared`
      : "No synthetic activity. These are actual in-flight queries.";
  }

  function nodeShortName(value, target) {
    return value === target
      ? target
      : value.endsWith("." + target)
        ? value.slice(0, -target.length - 1)
        : value;
  }
  function graphNodes() {
    const all = Array.isArray(state.scan?.nodes) ? state.scan.nodes : [];
    const byName = new Map(all.map((node) => [node.value, node]));
    const active = new Set(
      (state.scan?.active || [])
        .map(
          (query) => query.candidate || (!query.probe ? query.hostname : null),
        )
        .filter(Boolean),
    );
    let candidates = $("graph-verified").checked
      ? all.filter((node) => node.status === "verified")
      : [...all];
    const priority = (node) =>
      node.value === state.selectedNode
        ? 4
        : active.has(node.value)
          ? 3
          : node.status === "verified"
            ? 2
            : node.status === "pending"
              ? 1
              : 0;
    candidates.sort((a, b) => priority(b) - priority(a));
    const chosen = new Set();
    for (const candidate of candidates) {
      if (chosen.size >= graphLimit) break;
      const chain = [];
      let current = candidate;
      const seen = new Set();
      while (
        current &&
        !chosen.has(current.value) &&
        !seen.has(current.value)
      ) {
        chain.unshift(current);
        seen.add(current.value);
        current = byName.get(current.parent);
      }
      if (chosen.size + chain.length > graphLimit) continue;
      for (const node of chain) chosen.add(node.value);
    }
    return {
      nodes: all.filter((node) => chosen.has(node.value)),
      total: candidates.length,
      active,
    };
  }
  function applyTransform() {
    const { x, y, scale } = state.graphTransform;
    $("graph-layer").setAttribute(
      "transform",
      `translate(${x} ${y}) scale(${scale})`,
    );
    $("zoom-level").textContent = Math.round(scale * 100) + "%";
  }
  function fitGraph() {
    if (!state.graphBounds) return;
    const surface = $("graph-surface");
    const { width, height } = state.graphBounds;
    const scale = Math.min(
      1.15,
      Math.max(
        0.04,
        Math.min(
          (surface.clientWidth - 38) / width,
          (surface.clientHeight - 45) / height,
        ),
      ),
    );
    state.graphTransform = {
      x: Math.max(19, (surface.clientWidth - width * scale) / 2),
      y: Math.max(16, (surface.clientHeight - height * scale) / 2),
      scale,
    };
    applyTransform();
  }
  function zoom(multiplier) {
    const surface = $("graph-surface"),
      previous = state.graphTransform;
    const scale = Math.min(2.5, Math.max(0.04, previous.scale * multiplier));
    const cx = surface.clientWidth / 2,
      cy = surface.clientHeight / 2;
    state.graphTransform = {
      x: cx - ((cx - previous.x) * scale) / previous.scale,
      y: cy - ((cy - previous.y) * scale) / previous.scale,
      scale,
    };
    applyTransform();
  }
  function focusWork(value) {
    const queries = state.scan?.active || [];
    const preferred =
      value ||
      queries
        .map((query) => query.candidate || query.hostname)
        .find((hostname) => state.graphPositions.has(hostname));
    const position =
      state.graphPositions.get(preferred) ||
      [...state.graphPositions.values()].find(
        (entry) => entry.node.value !== state.scan?.target,
      );
    if (!position) return;
    const surface = $("graph-surface"),
      scale = Math.max(0.75, Math.min(1, state.graphTransform.scale));
    state.graphTransform = {
      x:
        Math.max(15, surface.clientWidth / 2 - 101 * scale) -
        position.x * scale,
      y: Math.min(80, surface.clientHeight / 3) - position.y * scale,
      scale,
    };
    applyTransform();
  }
  function renderGraph(force = false) {
    const scan = state.scan;
    const { nodes, total, active } = graphNodes();
    const key = JSON.stringify([
      scan?.id,
      nodes.map((node) => [node.value, node.parent, node.status]),
      [...active],
      state.selectedNode,
      $("graph-verified").checked,
    ]);
    if (!force && key === state.graphKey) return;
    state.graphKey = key;
    const layer = $("graph-layer");
    const focusedValue = layer.contains(document.activeElement)
      ? document.activeElement?.dataset?.hostname
      : null;
    layer.replaceChildren();
    $("graph-empty").hidden = Boolean(scan && nodes.length);
    $("graph-count").textContent = count(nodes.length) + " NODES";
    $("graph-limit").hidden = total <= nodes.length;
    $("graph-limit").textContent =
      `Showing ${nodes.length} of ${count(total)} candidates, prioritizing active and verified branches. Search all candidates in Findings.`;
    if (!scan || !nodes.length) {
      state.graphBounds = null;
      state.graphPositions = new Map();
      if (scan) {
        $("graph-empty").querySelector("h3").textContent =
          scan.status === "running"
            ? "Listening for the first clues."
            : "No hosts in this view yet.";
        $("graph-empty").querySelector("p").textContent =
          scan.status === "running"
            ? "Your starting evidence is being prepared. Candidates will appear as they arrive."
            : "Try another filter or inspect scan notes below.";
      }
      return;
    }
    const nodeWidth = 202,
      nodeHeight = 43,
      columnGap = 64,
      rowGap = 16;
    const syntheticRoot = {
      value: scan.target,
      status: "root",
      source: "target",
    };
    const children = new Map([[scan.target, []]]),
      byName = new Map(nodes.map((node) => [node.value, node]));
    for (const node of nodes)
      if (!children.has(node.value)) children.set(node.value, []);
    for (const node of nodes) {
      if (node.value === scan.target) continue;
      const parent =
        node.parent && byName.has(node.parent) && node.parent !== node.value
          ? node.parent
          : scan.target;
      if (!children.has(parent)) children.set(parent, []);
      children.get(parent).push(node);
    }
    const positions = new Map(),
      visited = new Set();
    let nextRow = 0,
      maxDepth = 0;
    function place(node, depth) {
      if (visited.has(node.value) || depth > 12) return;
      visited.add(node.value);
      maxDepth = Math.max(maxDepth, depth);
      const descendants = (children.get(node.value) || []).filter(
        (child) => !visited.has(child.value),
      );
      let y;
      if (descendants.length) {
        const start = nextRow;
        descendants.forEach((child) => place(child, depth + 1));
        y =
          ((start + Math.max(start, nextRow - 1)) / 2) * (nodeHeight + rowGap);
      } else y = nextRow++ * (nodeHeight + rowGap);
      positions.set(node.value, {
        x: depth * (nodeWidth + columnGap),
        y,
        node,
      });
    }
    place(syntheticRoot, 0);
    for (const node of nodes) if (!visited.has(node.value)) place(node, 1);
    for (const [value, pos] of positions) {
      if (value === scan.target) continue;
      const parent =
        positions.get(pos.node.parent) || positions.get(scan.target);
      if (!parent) continue;
      const x1 = parent.x + nodeWidth,
        y1 = parent.y + nodeHeight / 2,
        x2 = pos.x,
        y2 = pos.y + nodeHeight / 2;
      layer.append(
        svg("path", {
          d: `M${x1},${y1} C${x1 + columnGap / 2},${y1} ${x2 - columnGap / 2},${y2} ${x2},${y2}`,
          class:
            "graph-edge " +
            (active.has(value)
              ? "active"
              : pos.node.status === "verified"
                ? "verified"
                : ""),
        }),
      );
    }
    for (const [value, pos] of positions) {
      const isRoot = value === scan.target,
        node = pos.node;
      const group = svg("g", {
        class:
          "graph-node " +
          (isRoot ? "root" : node.status || "pending") +
          (active.has(value) ? " active" : "") +
          (state.selectedNode === value ? " selected" : ""),
        transform: `translate(${pos.x} ${pos.y})`,
        role: "button",
        tabindex: 0,
        "data-hostname": value,
        "aria-label": `${value}, ${isRoot ? "target domain" : active.has(value) ? "currently checking DNS" : readable(node.status)}. Select for details.`,
      });
      group.append(
        svg("title", {}, value),
        svg("rect", {
          x: 0,
          y: 0,
          width: nodeWidth,
          height: nodeHeight,
          rx: 6,
        }),
        svg("circle", { cx: 13, cy: 17, r: 3 }),
      );
      const label = nodeShortName(value, scan.target);
      group.append(
        svg(
          "text",
          { x: 23, y: 20 },
          label.length > 25 ? label.slice(0, 23) + "…" : label,
        ),
        svg(
          "text",
          { x: 23, y: 33, class: "node-subtitle" },
          isRoot
            ? "TARGET / DISCOVERY ROOT"
            : active.has(value)
              ? "DNS check in progress"
              : readable(node.source || "candidate") +
                " · " +
                readable(node.status),
        ),
      );
      group.addEventListener("click", (event) => {
        if (drag?.moved) return;
        event.stopPropagation();
        selectNode(value);
      });
      group.addEventListener("keydown", (event) => {
        if (["Enter", " "].includes(event.key)) {
          event.preventDefault();
          selectNode(value);
        }
      });
      layer.append(group);
    }
    state.graphPositions = positions;
    state.graphBounds = {
      width: (maxDepth + 1) * nodeWidth + maxDepth * columnGap,
      height: Math.max(1, nextRow) * (nodeHeight + rowGap) - rowGap,
    };
    if (!state.graphInitialized) {
      fitGraph();
      if (state.graphTransform.scale < 0.5) focusWork();
      state.graphInitialized = true;
    } else applyTransform();
    if (focusedValue)
      [...layer.querySelectorAll("[data-hostname]")]
        .find((node) => node.dataset.hostname === focusedValue)
        ?.focus({ preventScroll: true });
  }
  function selectNode(value) {
    state.selectedNode = value;
    renderNode();
    renderGraph();
  }
  function renderNode() {
    const details = $("node-details");
    const node = state.scan?.nodes?.find(
      (candidate) => candidate.value === state.selectedNode,
    );
    details.replaceChildren();
    if (!node) {
      details.append(
        element(
          "p",
          "muted",
          state.selectedNode === state.scan?.target
            ? `${state.scan.target} is the discovery root. Known hosts are verified first; each verified host can inspire new pattern candidates.`
            : "Select a node on the map or a hostname in the findings list to see its origin, pattern, and DNS evidence.",
        ),
      );
      return;
    }
    details.append(
      element("div", "node-host", node.value),
      statusBadge(node.status),
    );
    const rows = [
      ["Origin", readable(node.source || "unknown")],
      ["Parent", node.parent || "Target / initial discovery"],
      ["Round", String(node.round ?? 0)],
      ["Pattern", node.pattern || "No inferred pattern"],
      [
        "DNS",
        Array.isArray(node.addresses) && node.addresses.length
          ? node.addresses.join(", ")
          : "No verified address",
      ],
    ];
    const list = element("dl");
    rows.forEach(([title, value]) =>
      list.append(element("dt", "", title), element("dd", "", value)),
    );
    details.append(list);
    if (node.reason)
      details.append(element("p", "node-detail-note", node.reason));
    if (node.error)
      details.append(element("p", "node-detail-note", node.error));
  }
  function renderFindings() {
    const all = Array.isArray(state.scan?.nodes) ? state.scan.nodes : [];
    $("finding-count").textContent = count(
      all.filter((node) => node.status === "verified").length,
    );
    const filter = $("findings-filter").value,
      search = $("findings-search").value.toLowerCase().trim();
    const nodes = all.filter(
      (node) =>
        (filter === "all" || node.status === filter) &&
        (!search ||
          String(node.value).toLowerCase().includes(search) ||
          (node.addresses || []).some((address) =>
            String(address).toLowerCase().includes(search),
          )),
    );
    const pages = Math.max(1, Math.ceil(nodes.length / pageSize));
    state.page = Math.min(state.page, pages - 1);
    const pageNodes = nodes.slice(
      state.page * pageSize,
      (state.page + 1) * pageSize,
    );
    const signature = JSON.stringify(
      pageNodes.map((node) => [
        node.value,
        node.status,
        node.source,
        node.addresses,
      ]),
    );
    const body = $("findings-body");
    const focusedValue = body.contains(document.activeElement)
      ? document.activeElement?.dataset?.hostname
      : null;
    if (state.findingsKey !== signature) {
      state.findingsKey = signature;
      body.replaceChildren();
      for (const node of pageNodes) {
        const row = element("tr"),
          host = element("td"),
          status = element("td");
        const button = element("button", "host-button", node.value);
        button.dataset.hostname = node.value;
        button.addEventListener("click", () => {
          selectNode(node.value);
          focusWork(node.value);
          $("node-details").scrollIntoView({
            behavior: matchMedia("(prefers-reduced-motion: reduce)").matches
              ? "instant"
              : "smooth",
            block: "nearest",
          });
        });
        host.append(button);
        status.append(statusBadge(node.status));
        row.append(
          host,
          status,
          element("td", "host-origin", readable(node.source || "unknown")),
          element(
            "td",
            "host-address",
            (node.addresses || []).join(", ") || "—",
          ),
        );
        body.append(row);
      }
      if (focusedValue)
        [...body.querySelectorAll("[data-hostname]")]
          .find((node) => node.dataset.hostname === focusedValue)
          ?.focus({ preventScroll: true });
    }
    $("findings-empty").hidden = nodes.length > 0;
    $("findings-empty").textContent = search
      ? "No hostnames or addresses match your search."
      : all.length
        ? "No candidates match this filter yet."
        : "Verified discoveries will appear here as the scan progresses.";
    $("findings-total").textContent = nodes.length
      ? `${count(state.page * pageSize + 1)}–${count(Math.min(nodes.length, (state.page + 1) * pageSize))} of ${count(nodes.length)} matching hosts`
      : "0 matching hosts";
    $("findings-page").textContent = `${state.page + 1} / ${pages}`;
    $("findings-prev").disabled = state.page <= 0;
    $("findings-next").disabled = state.page >= pages - 1;
  }
  function renderDiscoveryMethods() {
    const labels = {
      seed: "Starting names",
      passive: "Public-source names",
      bootstrap: "Common names",
      numeric: "Numeric patterns",
      pattern: "Other patterns",
      model: "AI proposals",
      other: "Other",
    };
    const methods = Array.isArray(state.scan?.summary?.discovery_methods)
      ? state.scan.summary.discovery_methods
      : [];
    const body = $("discovery-method-rows");
    body.replaceChildren();
    const seen = new Set();
    for (const method of methods.slice(0, 7)) {
      if (!Object.hasOwn(labels, method?.method) || seen.has(method.method))
        continue;
      seen.add(method.method);
      const row = element("tr");
      row.append(element("td", "", labels[method.method]));
      for (const field of ["pending", "checked", "verified", "miss", "inconclusive", "wildcard"])
        row.append(element("td", "", count(Math.max(0, number(method[field])))));
      body.append(row);
    }
    $("discovery-method-empty").hidden = seen.size > 0;
  }
  function renderPatterns() {
    renderDiscoveryMethods();
    const patterns = Array.isArray(state.scan?.patterns)
      ? state.scan.patterns
      : [];
    $("pattern-count").textContent = count(patterns.length);
    const list = $("patterns-list"),
      scroll = list.scrollTop;
    list.replaceChildren();
    for (const pattern of patterns.slice(0, 100)) {
      const card = element("article", "pattern-card");
      card.append(
        element(
          "h3",
          "pattern-name",
          pattern.pattern || pattern.template || "Inferred pattern",
        ),
        element(
          "p",
          "pattern-meta",
          `${count(pattern.evidence_count ?? pattern.support)} supporting hosts${pattern.confidence !== undefined ? " · " + (typeof pattern.confidence === "number" ? Math.round(pattern.confidence * 100) + "% heuristic confidence" : readable(pattern.confidence)) : ""}`,
        ),
      );
      if (pattern.examples?.length)
        card.append(
          element("div", "pattern-examples", pattern.examples.join(" · ")),
        );
      if (pattern.note) card.append(element("p", "pattern-note", pattern.note));
      list.append(card);
    }
    if (!patterns.length)
      list.append(
        element(
          "div",
          "table-empty",
          "Patterns will emerge from verified hosts.",
        ),
      );
    if (patterns.length > 100)
      list.append(
        element(
          "p",
          "muted",
          `Showing 100 of ${patterns.length} patterns. Export JSON for the complete report.`,
        ),
      );
    list.scrollTop = scroll;
  }
  function renderTimeline() {
    const events = Array.isArray(state.scan?.events) ? state.scan.events : [];
    const timeline = $("timeline"),
      scroll = timeline.scrollTop;
    timeline.replaceChildren();
    for (const event of events.slice(-150).reverse()) {
      const row = element("li", "timeline-row");
      let time = "—";
      if (event.time !== undefined) {
        const raw =
          typeof event.time === "number"
            ? event.time > 1e11
              ? event.time
              : event.time * 1000
            : event.time;
        const date = new Date(raw);
        if (!Number.isNaN(date.getTime()))
          time = date.toLocaleTimeString([], { hour12: false });
      }
      row.append(
        element("time", "timeline-time", time),
        element(
          "span",
          "timeline-dot " + String(event.kind || "").replace(/[^a-z]/g, ""),
        ),
        element(
          "span",
          "timeline-message",
          event.message || readable(event.kind),
        ),
      );
      timeline.append(row);
    }
    if (!events.length)
      timeline.append(element("li", "table-empty", "No scan activity yet."));
    timeline.scrollTop = scroll;
  }
  function renderPredictions() {
    const predictions = Array.isArray(state.scan?.summary?.predictions)
      ? state.scan.summary.predictions
      : [];
    $("prediction-count").textContent = count(predictions.length);
    const list = $("predictions-list"),
      scroll = list.scrollTop;
    list.replaceChildren();
    for (const prediction of predictions.slice(-60).reverse()) {
      const rawScore = prediction.score;
      const hasScore =
        prediction.status === "scored" &&
        rawScore !== null &&
        rawScore !== undefined &&
        Number.isFinite(Number(rawScore));
      const card = element(
        "article",
        "prediction-card" + (hasScore ? "" : " fallback"),
      );
      const header = element("div", "prediction-card-header");
      header.append(
        element("span", "prediction-round", "ROUND " + count(prediction.round)),
        element(
          "span",
          "prediction-score",
          hasScore
            ? "Priority " + Number(rawScore).toFixed(3)
            : "Not scored · fallback",
        ),
      );
      card.append(
        header,
        element("h3", "", prediction.description || "Discovery plan"),
        element(
          "p",
          "",
          `${prediction.plan_id || "Plan"} · ${count(prediction.count)} candidates${hasScore ? " · predicted priority, not DNS existence probability" : " · no Jev score; default priority used"}`,
        ),
      );
      list.append(card);
    }
    if (!predictions.length)
      list.append(
        element(
          "div",
          "table-empty",
          predictionProvider(state.scan)
            ? "Jev ranking is enabled. Plan priorities will appear when the scanner evaluates its next discovery plans."
            : "No prediction layer is enabled. Jev can be selected separately when starting a new scan.",
        ),
      );
    if (predictions.length > 60)
      list.append(
        element("p", "muted", "Showing the latest 60 prediction decisions."),
      );
    list.scrollTop = scroll;
  }
  function renderScan() {
    renderSummary();
    renderWorkers();
    renderGraph();
    renderNode();
    renderFindings();
    renderPatterns();
    renderPredictions();
    renderTimeline();
  }
  function chooseScan(id) {
    if (id === state.selectedId) return;
    state.selectedId = id;
    state.selectedNode = null;
    state.graphInitialized = false;
    state.graphKey = "";
    state.findingsKey = "";
    state.page = 0;
    state.scan = null;
  }

  async function refresh() {
    if (state.polling) return;
    state.polling = true;
    try {
      const overview = await json("/api/state");
      state.scans = Array.isArray(overview.scans) ? overview.scans : [];
      state.features = overview.features || {};
      $("vocabulary-breadth-option").hidden =
        state.features.vocabulary_breadth !== true;
      renderCredentials(overview.credentials || {});
      if (!state.selectedId && state.scans.length)
        chooseScan(state.scans[0].id);
      renderHistory();
      if (state.selectedId) {
        const requestedId = state.selectedId;
        const scan = await json(
          "/api/scans/" + encodeURIComponent(requestedId),
        );
        if (requestedId === state.selectedId) {
          state.scan = scan;
          renderScan();
        }
      }
      setConnection(true, "Connected · local");
    } catch (error) {
      setConnection(false, "Connection interrupted");
      notice(error.message);
    } finally {
      state.polling = false;
      schedulePoll();
    }
  }
  function schedulePoll() {
    clearTimeout(pollTimer);
    pollTimer = setTimeout(
      refresh,
      activeGlobal() || activeStatuses.has(state.scan?.status) ? 800 : 2000,
    );
  }
  async function action(callback) {
    if (state.busy) return;
    state.busy = true;
    renderHistory();
    renderSummary();
    try {
      await callback();
      await refresh();
    } catch (error) {
      notice(error.message);
    } finally {
      state.busy = false;
      renderHistory();
      renderSummary();
    }
  }
  $("provider").addEventListener("change", () => {
    const provider = providers[$("provider").value];
    $("ai-settings").hidden = !provider;
    $("mode-note").textContent = provider
      ? "Hosted AI explores different combinations of verified names in diverse batches. Python expands patterns; DNS checks every guess. Leave the finding goal empty to explore within your scan limits."
      : "Start from public evidence or your own hostnames. Verified discoveries guide the next round. No generation API key or generation charges; optional plan ranking is billed separately.";
    $("api-key").value = "";
    if (provider) {
      $("model-label").textContent = provider.name + " model";
      $("key-label").textContent = provider.name + " API key";
      $("api-key").placeholder = "Paste your " + provider.name + " API key";
      $("model").replaceChildren(
        ...provider.models.map(([value, title]) => {
          const option = element("option", "", title);
          option.value = value;
          return option;
        }),
      );
    }
    renderCredentials();
  });
  $("save-key").addEventListener("click", () =>
    action(async () => {
      const key = $("api-key").value.trim();
      $("api-key").value = "";
      const provider = $("provider").value;
      if (!providers[provider])
        throw new Error("Select a hosted AI provider first.");
      if (!key)
        throw new Error(
          "Paste your " + providers[provider].name + " API key first.",
        );
      const result = await json("/api/credentials", { provider, api_key: key });
      state.credentials[provider] = Boolean(result.configured);
      renderCredentials();
      notice(
        providers[provider].name +
          " key is available for this server session. It was not saved to disk.",
        true,
      );
    }),
  );
  $("clear-key").addEventListener("click", () =>
    action(async () => {
      const provider = $("provider").value;
      if (!providers[provider]) return;
      $("api-key").value = "";
      await json("/api/credentials", { provider, api_key: "" });
      state.credentials[provider] = false;
      renderCredentials();
      notice(
        providers[provider].name +
          " key cleared for future scans. An already-running scan may retain its existing key.",
        true,
      );
    }),
  );
  $("prediction-provider").addEventListener("change", () => {
    const enabled = $("prediction-provider").value === "jev";
    $("prediction-settings").hidden = !enabled;
    $("prediction-budget").required = enabled;
    $("prediction-budget").disabled = !enabled;
    $("prediction-api-key").value = "";
    renderCredentials();
  });
  $("save-prediction-key").addEventListener("click", () =>
    action(async () => {
      const key = $("prediction-api-key").value.trim();
      $("prediction-api-key").value = "";
      if (!key) throw new Error("Paste your separate TypeSafe API key first.");
      const result = await json("/api/credentials", {
        provider: "jev",
        api_key: key,
      });
      state.credentials.jev = Boolean(result.configured);
      renderCredentials();
      notice(
        "TypeSafe key is available for this server session. It was not saved to disk or browser storage.",
        true,
      );
    }),
  );
  $("clear-prediction-key").addEventListener("click", () =>
    action(async () => {
      $("prediction-api-key").value = "";
      await json("/api/credentials", { provider: "jev", api_key: "" });
      state.credentials.jev = false;
      renderCredentials();
      notice(
        "TypeSafe key cleared for future scans. An already-running scan may retain its existing key.",
        true,
      );
    }),
  );
  $("scan-form").addEventListener("submit", (event) => {
    event.preventDefault();
    if (!previewIsCurrent()) {
      scheduleResolve(0);
      notice(
        "Review the discovery scope preview, then choose Start discovery.",
      );
      return;
    }
    action(async () => {
      const provider = $("provider").value;
      if (provider !== "none" && !state.credentials[provider])
        throw new Error(
          "Save your " +
            (providers[provider]?.name || "provider") +
            " key for this session, or select free discovery mode.",
        );
      const target = $("target").value.trim();
      const predictionSettings = predictionStartSettings();
      const input = discoveryInput();
      const preview = await resolveInput();
      if (!preview) throw new Error("Enter a domain or known hostname first.");
      const options = {
        find: $("goal").value ? Number($("goal").value) : null,
        time_limit: Number($("duration").value) * 60,
        max_candidates: Number($("candidates").value),
        max_rounds: Number($("rounds").value),
        workers: Number($("workers").value),
        rate: Number($("rate").value),
        max_depth: Number($("depth").value),
        passive: resolvedPassive(preview),
        ...(state.features.vocabulary_breadth === true
          ? { vocabulary_breadth: provider !== "none" && $("vocabulary-breadth").checked }
          : {}),
      };
      const result = await json("/api/scans", {
        target,
        ...(input.scope ? { scope: input.scope } : {}),
        provider,
        model: provider !== "none" ? $("model").value : null,
        budget_usd: provider !== "none" ? Number($("budget").value) : null,
        options,
        seeds: input.seeds,
        ...predictionSettings,
      });
      chooseScan(result.id);
      $("notice").hidden = true;
    });
  });
  for (const id of ["target", "starting-seeds", "discovery-scope"]) {
    $(id).addEventListener("input", () => scheduleResolve());
    $(id).addEventListener("blur", () => {
      if (!previewIsCurrent()) scheduleResolve(0);
    });
  }
  $("passive").addEventListener("change", renderInputPreview);
  $("pause").addEventListener("click", () =>
    action(async () => {
      if (state.selectedId)
        await json(
          "/api/scans/" + encodeURIComponent(state.selectedId) + "/pause",
          {},
        );
    }),
  );
  function resumeRequest(payload) {
    return action(async () => {
      if (state.selectedId)
        await json(
          "/api/scans/" + encodeURIComponent(state.selectedId) + "/resume",
          payload,
        );
    });
  }
  $("resume").addEventListener("click", () => {
    if (!state.scan || state.busy || activeGlobal()) return;
    if (state.scan.status === "paused" && !predictionProvider(state.scan)) {
      resumeRequest({});
      return;
    }
    const scan = state.scan,
      options = scan.options || {},
      summary = scan.summary || {},
      ledger = summary.ledger;
    $("resume-title").textContent = "Continue " + scan.target;
    $("resume-evidence").textContent =
      `${count(summary.verified)} verified hosts · ${count(summary.attempted)} candidates already checked · ${count(summary.round)} rounds completed`;
    $("resume-candidates").value = number(options.max_candidates, 1000);
    $("resume-goal").value = options.find ?? "";
    $("resume-duration").value = number(options.time_limit, 300) / 60;
    $("resume-rounds").value = number(options.max_rounds, 10);
    $("resume-budget-row").hidden = providerName(scan) === "none";
    $("resume-budget").value = number(ledger?.limit_usd ?? ledger?.limit, 1);
    $("resume-budget").required = providerName(scan) !== "none";
    $("resume-budget-note").textContent =
      `${money(ledger?.spent_usd ?? ledger?.spent)} already spent${number(ledger?.reserved_usd ?? ledger?.reserved) > 0 ? " + " + money(ledger?.reserved_usd ?? ledger?.reserved) + " reserved" : ""}. Previous usage counts toward this total; the budget does not reset.`;
    const predictionLedger = summary.prediction_ledger;
    const predictionEnabled = Boolean(predictionProvider(scan));
    const predictionLimit = number(
      predictionLedger?.limit_usd ?? predictionLedger?.limit,
      0.05,
    );
    $("resume-prediction-budget-row").hidden = !predictionEnabled;
    $("resume-prediction-budget").required = predictionEnabled;
    $("resume-prediction-budget").value = predictionLimit;
    $("resume-prediction-budget").min = predictionLimit;
    $("resume-prediction-budget-note").textContent =
      `${money(predictionLedger?.spent_usd ?? predictionLedger?.spent)} already spent${number(predictionLedger?.reserved_usd ?? predictionLedger?.reserved) > 0 ? " + " + money(predictionLedger?.reserved_usd ?? predictionLedger?.reserved) + " reserved" : ""}. This additional, cumulative prediction cap can stay the same or increase; it is separate from generation charges.`;
    $("resume-error").hidden = true;
    $("resume-dialog").showModal();
  });
  for (const id of ["close-resume", "cancel-resume"])
    $(id).addEventListener("click", () => $("resume-dialog").close());
  $("resume-form").addEventListener("submit", (event) => {
    event.preventDefault();
    if (!state.scan) return;
    const options = {
      max_candidates: Number($("resume-candidates").value),
      find: $("resume-goal").value ? Number($("resume-goal").value) : null,
      time_limit: Number($("resume-duration").value) * 60,
      max_rounds: Number($("resume-rounds").value),
    };
    const summary = state.scan.summary || {},
      ledger = summary.ledger;
    let error = "";
    if (options.max_candidates <= number(summary.attempted))
      error = `Raise the candidate cap above ${count(summary.attempted)} already-checked candidates to continue.`;
    else if (options.find !== null && options.find <= number(summary.verified))
      error = `Raise the finding goal above ${count(summary.verified)} verified hosts, or leave it empty.`;
    else if (options.max_rounds <= number(summary.round))
      error = `Raise the round cap above ${count(summary.round)} completed rounds to continue.`;
    const payload = { options };
    if (providerName(state.scan) !== "none") {
      payload.budget_usd = Number($("resume-budget").value);
      if (
        payload.budget_usd <
        number(ledger?.spent_usd ?? ledger?.spent) +
          number(ledger?.reserved_usd ?? ledger?.reserved)
      )
        error =
          "The total AI budget must cover previous spending and any reserved usage.";
    }
    if (predictionProvider(state.scan)) {
      const predictionLedger = summary.prediction_ledger;
      payload.prediction_budget_usd = Number(
        $("resume-prediction-budget").value,
      );
      const minimum = Math.max(
        number(predictionLedger?.limit_usd ?? predictionLedger?.limit),
        number(predictionLedger?.spent_usd ?? predictionLedger?.spent) +
          number(predictionLedger?.reserved_usd ?? predictionLedger?.reserved),
      );
      if (
        !Number.isFinite(payload.prediction_budget_usd) ||
        payload.prediction_budget_usd < minimum
      )
        error =
          "The additional prediction budget cannot decrease and must cover previous prediction spending and reserved usage.";
    }
    if (error) {
      $("resume-error").textContent = error;
      $("resume-error").hidden = false;
      return;
    }
    $("resume-dialog").close();
    resumeRequest(payload);
  });
  $("scan-history").addEventListener("change", () => {
    chooseScan($("scan-history").value);
    refresh();
  });
  $("dismiss-notice").addEventListener("click", () => {
    $("notice").hidden = true;
  });
  $("graph-verified").addEventListener("change", () => {
    state.graphInitialized = false;
    renderGraph(true);
  });
  $("fit-graph").addEventListener("click", fitGraph);
  $("focus-active").addEventListener("click", () => focusWork());
  $("zoom-in").addEventListener("click", () => zoom(1.3));
  $("zoom-out").addEventListener("click", () => zoom(1 / 1.3));
  const surface = $("graph-surface");
  surface.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    drag = {
      id: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      originX: state.graphTransform.x,
      originY: state.graphTransform.y,
      moved: false,
    };
  });
  surface.addEventListener("pointermove", (event) => {
    if (!drag || drag.id !== event.pointerId) return;
    const dx = event.clientX - drag.x,
      dy = event.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 4) {
      drag.moved = true;
      surface.setPointerCapture(event.pointerId);
      state.graphTransform.x = drag.originX + dx;
      state.graphTransform.y = drag.originY + dy;
      applyTransform();
    }
  });
  surface.addEventListener("pointerup", () => {
    setTimeout(() => {
      drag = null;
    }, 0);
  });
  surface.addEventListener("pointercancel", () => {
    drag = null;
  });
  surface.addEventListener(
    "wheel",
    (event) => {
      if (!event.ctrlKey && !event.metaKey) return;
      event.preventDefault();
      zoom(event.deltaY > 0 ? 0.9 : 1.1);
    },
    { passive: false },
  );
  $("findings-search").addEventListener("input", () => {
    state.page = 0;
    renderFindings();
  });
  $("findings-filter").addEventListener("change", () => {
    state.page = 0;
    renderFindings();
  });
  $("findings-prev").addEventListener("click", () => {
    state.page--;
    renderFindings();
  });
  $("findings-next").addEventListener("click", () => {
    state.page++;
    renderFindings();
  });
  const tabs = [...document.querySelectorAll("[data-tab]")];
  function activateTab(tab) {
    tabs.forEach((item) => {
      const selected = item === tab;
      item.classList.toggle("active", selected);
      item.setAttribute("aria-selected", String(selected));
      item.tabIndex = selected ? 0 : -1;
      $(item.dataset.tab + "-panel").hidden = !selected;
    });
  }
  $("show-predictions").addEventListener("click", () => {
    activateTab($("tab-predictions"));
    $("predictions-panel").scrollIntoView({
      behavior: matchMedia("(prefers-reduced-motion: reduce)").matches
        ? "instant"
        : "smooth",
      block: "nearest",
    });
  });
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => activateTab(tab));
    tab.addEventListener("keydown", (event) => {
      if (["ArrowRight", "ArrowLeft", "Home", "End"].includes(event.key)) {
        event.preventDefault();
        const next =
          event.key === "Home"
            ? 0
            : event.key === "End"
              ? tabs.length - 1
              : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) %
                tabs.length;
        activateTab(tabs[next]);
        tabs[next].focus();
      }
    });
  });
  $("export").addEventListener("click", () =>
    action(async () => {
      if (!state.selectedId) return;
      const format = $("export-format").value;
      const response = await api(
        "/api/scans/" +
          encodeURIComponent(state.selectedId) +
          "/export?format=" +
          encodeURIComponent(format),
      );
      const url = URL.createObjectURL(await response.blob());
      const link = element("a");
      link.href = url;
      link.download = `${(state.scan?.target || "enumeraite").replace(/[^a-z0-9.-]/gi, "_")}-discovery.${format}`;
      document.body.append(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }),
  );
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) refresh();
  });
  renderCredentials({});
  renderHistory();
  renderScan();
  refresh();
  if ($("target").value.trim()) scheduleResolve(0);
})();
