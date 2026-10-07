"use strict";

// No browser, server, network, or real credentials are accessed by these tests.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const assets = path.resolve(__dirname, "../../src/enumeraite/web/assets");
const html = fs.readFileSync(path.join(assets, "index.html"), "utf8");
const source = fs.readFileSync(path.join(assets, "app.js"), "utf8");

class FakeElement {
  constructor(tag = "div") {
    this.tagName = tag;
    this.children = [];
    this.dataset = {};
    this.listeners = {};
    this.attributes = {};
    this.value = "";
    this.hidden = false;
    this.checked = false;
    this.disabled = false;
    this.scrollTop = 0;
    this.clientWidth = 650;
    this.clientHeight = 350;
    this.classList = { toggle() {} };
    this._text = "";
  }
  set textContent(value) {
    this._text = String(value);
    this.children = [];
  }
  set innerHTML(_) {
    throw new Error("Dashboard rendering must not parse HTML from data");
  }
  get textContent() {
    return (
      this._text +
      this.children.map((child) => child.textContent || "").join("")
    );
  }
  get lastElementChild() {
    return (
      this.children.at(-1) ||
      this._last ||
      (this._last = new FakeElement("span"))
    );
  }
  append(...children) {
    this.children.push(...children);
  }
  replaceChildren(...children) {
    this._text = "";
    this.children = children;
  }
  setAttribute(name, value) {
    this.attributes[name] = String(value);
  }
  addEventListener(name, callback) {
    (this.listeners[name] ||= []).push(callback);
  }
  querySelector(selector) {
    return (
      this.children.find((child) => child.tagName === selector) ||
      new FakeElement(selector)
    );
  }
  querySelectorAll() {
    return [];
  }
  contains() {
    return false;
  }
  focus() {}
  scrollIntoView() {
    this.scrolledIntoView = true;
  }
  showModal() {
    this.open = true;
  }
  close() {
    this.open = false;
  }
  remove() {}
  setPointerCapture() {}
}

async function fixture({ features } = {}) {
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
  assert.equal(new Set(ids).size, ids.length, "HTML IDs must be unique");
  const nodes = new Map(ids.map((id) => [id, new FakeElement()]));
  const node = (id) => {
    assert(nodes.has(id), `Missing HTML ID: ${id}`);
    return nodes.get(id);
  };
  const tabs = ["findings", "patterns", "predictions", "timeline"].map(
    (name) => {
      const item = node("tab-" + name);
      item.dataset.tab = name;
      return item;
    },
  );
  for (const [id, value] of Object.entries({
    provider: "none",
    "prediction-provider": "none",
    "prediction-budget": "0.05",
    passive: "auto",
    "findings-filter": "verified",
    duration: "5",
    candidates: "1000",
    rounds: "10",
    workers: "20",
    rate: "50",
    depth: "4",
  }))
    node(id).value = value;
  const requests = [],
    storageRemovals = [];
  const credentials = {
    minimax: false,
    openai: false,
    anthropic: false,
    jev: false,
  };
  let selectedScan = null;
  const sandbox = {
    console,
    URLSearchParams,
    URL,
    Date,
    Math,
    Number,
    String,
    Boolean,
    Array,
    Object,
    JSON,
    Map,
    Set,
    Error,
    Promise,
    location: { hash: "", pathname: "/", search: "" },
    history: { replaceState() {} },
    sessionStorage: {
      removeItem(key) {
        storageRemovals.push(key);
      },
      getItem() {
        throw new Error("Storage reads prohibited");
      },
      setItem() {
        throw new Error("Storage writes prohibited");
      },
    },
    setTimeout() {
      return 1;
    },
    clearTimeout() {},
    matchMedia() {
      return { matches: true };
    },
    document: {
      getElementById: node,
      createElement: (tag) => new FakeElement(tag),
      createElementNS: (_, tag) => new FakeElement(tag),
      createTextNode: (value) => {
        const text = new FakeElement("text");
        text.textContent = value;
        return text;
      },
      querySelectorAll: (selector) => (selector === "[data-tab]" ? tabs : []),
      addEventListener() {},
      activeElement: null,
      body: new FakeElement("body"),
    },
    fetch: async (url, options = {}) => {
      assert(
        url.startsWith("/api/"),
        "Only the fake same-origin API may be used",
      );
      assert.equal(options.headers?.Authorization, undefined);
      const body = options.body ? JSON.parse(options.body) : undefined;
      requests.push({ url, body });
      let result = {};
      if (url === "/api/state")
        result = {
          credentials,
          scans: selectedScan ? [selectedScan] : [],
          ...(features === undefined ? {} : { features }),
        };
      else if (url === "/api/credentials") {
        credentials[body.provider] = Boolean(body.api_key);
        result = { configured: credentials[body.provider] };
      } else if (url === "/api/resolve")
        result = {
          input: body.target,
          target: "example.com",
          seeds: body.target === "example.com" ? [] : [body.target],
          mode: "seeded",
          passive_default: false,
        };
      else if (url === "/api/scans") {
        selectedScan = {
          id: "fake-scan",
          target: "example.com",
          status: "running",
          phase: "Verifying DNS",
          options: body.options,
          provider: body.provider,
          summary: { verified: 0, attempted: 0, queued: 0, round: 0 },
          nodes: [],
          active: [],
          events: [],
          patterns: [],
        };
        result = { id: selectedScan.id };
      } else if (url === "/api/scans/fake-scan") result = selectedScan;
      return { ok: true, status: 200, json: async () => result };
    },
  };
  vm.createContext(sandbox);
  const instrumented = source.replace(
    /\}\)\(\);\s*$/,
    "globalThis.__dashboardTest={state,renderSummary,renderPredictions,renderDiscoveryMethods,predictionStartSettings,resolveInput,renderCredentials,renderFindings,renderNode,renderGraph,renderScan,selectNode};\n})();",
  );
  assert.notEqual(
    instrumented,
    source,
    "Test instrumentation must find the module boundary",
  );
  vm.runInContext(instrumented, sandbox, { filename: "app.js" });
  async function flush() {
    for (let i = 0; i < 35; i++) await Promise.resolve();
  }
  async function trigger(id, event) {
    for (const callback of node(id).listeners[event] || [])
      await callback({ preventDefault() {} });
    await flush();
  }
  await flush();
  return {
    node,
    requests,
    storageRemovals,
    credentials,
    state: sandbox.__dashboardTest.state,
    api: sandbox.__dashboardTest,
    trigger,
    flush,
  };
}

test("plain-localhost initialization stays tokenless and touches only legacy cleanup", async () => {
  const ui = await fixture();
  assert(ui.requests.some((request) => request.url === "/api/state"));
  assert.deepEqual(ui.storageRemovals, ["enumeraite.ui.token"]);
  assert.equal(ui.node("prediction-provider").value, "none");
  assert.equal(ui.node("prediction-spend-panel").hidden, true);
});

function showDnsNodes(ui, nodes) {
  ui.state.scan = {
    id: "dns-fixture", target: "example.com", status: "paused", provider: "none",
    nodes, active: [], events: [], patterns: [], options: {},
    summary: {verified: nodes.filter((node) => node.status === "verified").length},
  };
  ui.state.selectedNode = nodes[0]?.value;
  ui.node("findings-filter").value = "all";
  ui.api.renderScan();
}

function dnsNode(overrides = {}) {
  return {
    value: "api.example.com", status: "verified", source: "model", attempts: 1,
    addresses: ["192.0.2.10"],
    dns_evidence: {
      checked_at: "2026-10-03T12:34:56.000Z", duration_ms: 12.5,
      records: {
        A: {status: "answered", addresses: ["192.0.2.10"], error: ""},
        AAAA: {status: "no_answer", addresses: [], error: ""},
      },
      wildcard: {status: "not_detected", parent: "example.com"},
    },
    ...overrides,
  };
}

function detailRows(ui) {
  const children = ui.node("node-details").children.find((child) => child.tagName === "dl").children;
  const rows = {};
  for (let i = 0; i < children.length; i += 2)
    rows[children[i].textContent] = children[i + 1].textContent;
  return rows;
}

function descendants(node) {
  return [node, ...node.children.flatMap(descendants)];
}

test("DNS findings separate address evidence from neutral unchecked web status", async () => {
  const ui = await fixture();
  const requestCount = ui.requests.length;
  showDnsNodes(ui, [dnsNode()]);
  const row = ui.node("findings-body").children[0];
  assert.deepEqual(row.children.map((cell) => cell.textContent),
    ["api.example.com", "DNS resolved", "Web not checked", "AI proposal", "192.0.2.10"]);
  assert.equal(row.children[2].className, "web-status");
  const details = detailRows(ui);
  assert.equal(details["A (IPv4)"], "Address answer · 192.0.2.10");
  assert.equal(details["AAAA (IPv6)"], "No AAAA answer");
  assert.equal(details["Last DNS check"], "2026-10-03 12:34:56.000 UTC");
  assert.equal(details["Lookup duration"], "12.5 ms");
  assert.equal(details["Completed DNS attempts"], "1");
  assert.match(details["Wildcard check"], /No wildcard detected.*Parent: example.com/);
  assert.match(ui.node("node-details").textContent, /two random sibling names per parent and reuse/);
  assert.match(ui.node("node-details").textContent, /not live monitoring/);
  assert.match(ui.node("node-details").textContent, /does not prove a working website/);
  assert.match(ui.node("scan-description").textContent, /Saved DNS observations/);
  assert.match(ui.node("live-announcement").textContent, /DNS-resolved hosts. Web not checked/);
  assert.equal(ui.requests.length, requestCount, "Rendering does not check DNS or web availability");
  assert.match(html, /<th scope="col">DNS status<\/th>/);
  assert.match(html, /<th scope="col">Web<\/th>/);
  assert.match(html, /<option value="verified">DNS resolved<\/option>/);
});

test("legacy findings retain saved IP families without inventing lookup history", async () => {
  const ui = await fixture();
  showDnsNodes(ui, [dnsNode({
    dns_evidence: undefined, attempts: undefined,
    addresses: ["192.0.2.10", "2001:db8::10"],
  })]);
  let details = detailRows(ui);
  assert.equal(details["A (IPv4)"], "Check: Not recorded · Saved addresses: 192.0.2.10");
  assert.equal(details["AAAA (IPv6)"], "Check: Not recorded · Saved addresses: 2001:db8::10");
  for (const field of ["Last DNS check", "Lookup duration", "Completed DNS attempts", "Wildcard check"])
    assert.equal(details[field], "Not recorded");
  ui.state.scan.nodes[0].addresses = ["192.0.2.10"];
  ui.api.renderNode();
  details = detailRows(ui);
  assert.equal(details["AAAA (IPv6)"], "Check: Not recorded · No saved IPv6 address");
  assert(!details["AAAA (IPv6)"].includes("No AAAA answer"));
  assert(!details["AAAA (IPv6)"].includes("Not checked"));
});

test("partial-positive DNS explains failed record types and renders hostile evidence as text", async () => {
  const ui = await fixture();
  const hostile = "<img src=x onerror=alert(1)>";
  const node = dnsNode({reason: hostile, error: hostile, pattern: hostile, parent: hostile});
  node.dns_evidence.records.AAAA = {status: "error", addresses: [], error: hostile};
  node.dns_evidence.wildcard = {status: "different_answer", parent: hostile};
  showDnsNodes(ui, [node]);
  const details = detailRows(ui);
  assert.match(ui.node("findings-body").children[0].children[1].textContent, /DNS resolvedPartial evidence/);
  assert.match(ui.node("node-details").textContent, /Partial DNS evidence.*AAAA lookup failed/);
  assert.match(details["AAAA (IPv6)"], /Lookup error/);
  assert(details["AAAA (IPv6)"].includes(hostile));
  assert.match(details["Wildcard check"], /differ from the sampled wildcard response/);
  assert(details["Wildcard check"].includes(hostile));
  assert(descendants(ui.node("node-details")).every((item) => !["img", "script"].includes(item.tagName)));
  assert.equal(ui.node("findings-body").children[0].children[2].textContent, "Web not checked");
});

test("record failures, skipped checks and wildcard outcomes remain distinguishable", async () => {
  const ui = await fixture();
  const cases = [
    ["pending", "DNS queued", "not_checked", /Not checked for this observation/],
    ["miss", "No DNS address", "not_detected", /No wildcard detected/],
    ["wildcard", "Wildcard filtered", "matched", /matching or rotating sibling answers/],
    ["inconclusive", "DNS inconclusive", "inconclusive", /calibration was inconclusive/],
  ];
  for (const [status, label, wildcard, explanation] of cases) {
    const node = dnsNode({status, addresses: []});
    node.dns_evidence.records = {
      A: {status: "nxdomain", addresses: [], error: ""},
      AAAA: {status: "not_checked", addresses: [], error: ""},
    };
    node.dns_evidence.wildcard.status = wildcard;
    showDnsNodes(ui, [node]);
    assert.equal(ui.node("findings-body").children[0].children[1].textContent, label);
    assert.equal(ui.node("findings-body").children[0].children[2].textContent, "Web not checked");
    const details = detailRows(ui);
    assert.equal(details["A (IPv4)"], "Name not found (NXDOMAIN)");
    assert.equal(details["AAAA (IPv6)"], "Not checked");
    assert.match(details["Wildcard check"], explanation);
  }
});

test("positive DNS with skipped or contradictory sibling record checks warns without treating no-answer as a failure", async () => {
  const ui = await fixture();
  for (const [status, warning] of [
    ["not_checked", "AAAA was not checked"],
    ["nxdomain", "AAAA reported NXDOMAIN"],
  ]) {
    const node = dnsNode();
    node.dns_evidence.records.AAAA.status = status;
    showDnsNodes(ui, [node]);
    assert(ui.node("node-details").textContent.includes(warning));
    assert.match(ui.node("findings-body").textContent, /Partial evidence/);
  }
  showDnsNodes(ui, [dnsNode()]);
  assert(!ui.node("node-details").textContent.includes("Partial DNS evidence"));
  assert(!ui.node("findings-body").textContent.includes("Partial evidence"));
});

test("finding clicks open current evidence and graph accessibility uses DNS-specific labels", async () => {
  const ui = await fixture();
  const requestCount = ui.requests.length;
  showDnsNodes(ui, [dnsNode(), dnsNode({value: "second.example.com", source: "passive"})]);
  const button = ui.node("findings-body").children[1].children[0].children[0];
  assert.match(button.attributes["aria-label"], /DNS resolved.*Show DNS evidence; web not checked/);
  await button.listeners.click[0]();
  assert.equal(ui.state.selectedNode, "second.example.com");
  assert.equal(ui.node("node-details").children[0].textContent, "second.example.com");
  assert.equal(ui.node("node-details").scrolledIntoView, true);
  const graphNode = ui.node("graph-layer").children.find((item) => item.attributes["data-hostname"] === "second.example.com");
  assert.match(graphNode.attributes["aria-label"], /DNS resolved. Web not checked/);
  assert.match(graphNode.textContent, /DNS resolved · Web not checked/);
  assert.equal(ui.requests.length, requestCount, "Inspecting a saved node makes no network request");
});

test("evidence-only snapshot updates refresh finding warnings and selected details", async () => {
  const ui = await fixture();
  showDnsNodes(ui, [dnsNode()]);
  const firstRow = ui.node("findings-body").children[0];
  const changed = dnsNode();
  changed.dns_evidence.checked_at = "2026-10-03T13:00:00Z";
  changed.dns_evidence.records.AAAA = {status: "error", addresses: [], error: "Resolver timeout"};
  ui.state.scan.nodes = [changed];
  ui.api.renderScan();
  assert.notEqual(ui.node("findings-body").children[0], firstRow);
  assert.match(ui.node("findings-body").textContent, /Partial evidence/);
  assert.equal(detailRows(ui)["Last DNS check"], "2026-10-03 13:00:00.000 UTC");
  assert.match(detailRows(ui)["AAAA (IPv6)"], /Resolver timeout/);
});

function originPanel(ui) {
  return ui.node("node-details").children.find((child) => child.className === "origin-section");
}

function originRows(ui) {
  const children = originPanel(ui).children.find((child) => child.tagName === "dl").children;
  const rows = {};
  for (let i = 0; i < children.length; i += 2)
    rows[children[i].textContent] = children[i + 1].textContent;
  return rows;
}

test("legacy passive memberships expose all known collectors without inventing record history", async () => {
  const ui = await fixture();
  const node = dnsNode({value: "podcast.example.com", source: "passive", reason: "Public discovery"});
  showDnsNodes(ui, [node]);
  ui.state.scan.created_at = "2026-01-01T00:00:00Z";
  ui.state.scan.summary.passive_sources = {
    "crt.sh": [node.value], merklemap: [node.value], subfinder: [node.value],
    unrelated: ["other.example.com"],
  };
  const requests = ui.requests.length;
  ui.api.renderScan();
  const panel = originPanel(ui);
  const cards = panel.children.filter((child) => child.tagName === "article");
  assert.deepEqual(cards.map((card) => card.children[0].textContent), ["crt.sh", "merklemap", "subfinder"]);
  assert(cards.every((card) => card.textContent.includes("details were not recorded")));
  assert.match(panel.textContent, /silent output does not report the upstream provider/);
  assert.equal(originRows(ui)["Queued at"], "Not recorded");
  assert(!panel.textContent.includes("2026-01-01"));
  const links = descendants(panel).filter((child) => child.tagName === "a");
  assert.deepEqual(links.map((link) => new URL(link.attributes.href).hostname), ["crt.sh"]);
  assert(links.every((link) => link.textContent.includes("Live lookup")));
  assert.equal(ui.requests.length, requests, "Rendering links does not fetch source data");
});

test("saved passive records distinguish certificate dates, collection time and historical IP from DNS", async () => {
  const ui = await fixture();
  const node = dnsNode({
    source: "seed", reason: "Supplied starting name",
    origin_evidence: {
      queued_at: "2026-10-03T11:00:00Z",
      passive: [
        {source: "crt.sh", query: "%.example.com", matched_name: "*.API.example.com",
          collected_at: "2026-10-03T11:01:00Z", certificate_id: "123456789",
          issuer: "Example CA", not_before: "2024-01-01", not_after: "2025-01-01",
          entry_timestamp: "2024-01-02"},
        {source: "merklemap", query: "*.example.com", matched_name: "api.example.com",
          collected_at: "2026-10-03T11:02:00Z", subject_common_name: "*.example.com", first_seen: "2024-02-03T04:05:06Z"},
      ],
    },
  });
  showDnsNodes(ui, [node]);
  const panel = originPanel(ui);
  assert.equal(originRows(ui)["First queued by"], "Starting name");
  assert.equal(originRows(ui)["Why proposed"], "Supplied starting name");
  assert.equal(originRows(ui)["Queued at"], "2026-10-03 11:00:00.000 UTC");
  for (const value of ["%.example.com", "*.API.example.com", "Example CA", "2024-01-01", "2025-01-01", "2024-01-02", "*.example.com", "2024-02-03T04:05:06Z"])
    assert(panel.textContent.includes(value));
  assert.match(panel.textContent, /Source collected at2026-10-03 11:01:00.000 UTC/);
  assert.match(panel.textContent, /Provider first seen \(not this scan's DNS check\)2024-02-03T04:05:06Z/);
  assert.match(panel.textContent, /at most 3 per collector/);
  assert(!panel.textContent.includes("192.0.2.10"));
  assert.match(detailRows(ui)["A (IPv4)"], /192.0.2.10/);
  assert.equal(detailRows(ui)["Last DNS check"], "2026-10-03 12:34:56.000 UTC");
  const link = descendants(panel).find((child) => child.tagName === "a");
  assert.equal(link.attributes.href, "https://crt.sh/?id=123456789");
  assert(!link.textContent.includes("Live lookup"));
  assert.equal(link.attributes.rel, "noopener noreferrer");
});

test("proposal metadata explains generation and its evidence cohort can select existing nodes", async () => {
  const ui = await fixture();
  const candidate = dnsNode({
    source: "pattern", reason: "Extend the observed numeric family", parent: "api1.example.com",
    round: 3, pattern: "api{n}.example.com",
    origin_evidence: {
      evidence: ["api1.example.com", "missing.example.com"], confidence: 0.8,
      provider: "example-provider", model: "example-model", focus: "numeric services", batch: 2,
      numeric_ceiling: 20, plan_id: "numeric-1", plan_description: "Continue the family",
      layout: "{word}.example.com", word: "api",
    },
  });
  showDnsNodes(ui, [candidate, dnsNode({value: "api1.example.com", source: "seed"})]);
  const panel = originPanel(ui);
  assert.equal(originRows(ui)["Why proposed"], candidate.reason);
  assert.equal(originRows(ui).Parent, candidate.parent);
  for (const value of ["example-provider", "example-model", "numeric services", "numeric-1", "Continue the family", "{word}.example.com"])
    assert(panel.textContent.includes(value));
  assert.match(panel.textContent, /Numeric ceiling20/);
  assert.match(panel.textContent, /0.8 \/ 1 \(not DNS likelihood\)/);
  assert.match(panel.textContent, /Generated hypothesis, not an observed hostname/);
  const buttons = descendants(panel).filter((child) => child.tagName === "button");
  assert.deepEqual(buttons.map((button) => button.textContent), ["api1.example.com"]);
  assert.match(panel.textContent, /missing.example.com \(not in saved node list\)/);
  await buttons[0].listeners.click[0]();
  assert.equal(ui.state.selectedNode, "api1.example.com");
  assert.equal(ui.node("node-details").children[0].textContent, "api1.example.com");
});

test("legacy AI configuration is not attributed to a request and metadata-only updates refresh origins", async () => {
  const ui = await fixture();
  showDnsNodes(ui, [dnsNode({reason: "Saved reason", parent: "parent.example.com"})]);
  ui.state.scan.provider = "openai";
  ui.state.scan.model = "saved-model";
  ui.api.renderScan();
  assert.match(originPanel(ui).textContent, /openai \/ saved-model \(not recorded request metadata\)/);
  assert.match(originPanel(ui).textContent, /cohort was not recorded/);
  assert.equal(originRows(ui)["Why proposed"], "Saved reason");
  ui.state.scan.nodes[0].origin_evidence = {provider: "request-provider", model: "request-model", batch: 7};
  ui.state.scan.summary.passive_sources = {"crt.sh": ["api.example.com"]};
  ui.api.renderScan();
  const text = originPanel(ui).textContent;
  assert(text.includes("request-provider") && text.includes("request-model"));
  assert(!text.includes("saved-model"));
  assert(text.includes("crt.sh"));
  assert.equal(originRows(ui)["First queued by"], "AI proposal");
});

test("malicious passive fields remain text and links can only target official source hosts", async () => {
  const ui = await fixture();
  const attack = '<img src=x onerror="alert(1)">';
  const node = dnsNode({
    value: 'evil.example.com&redirect=https://evil.test/#<script>',
    origin_evidence: {
      passive: [
        {source: "crt.sh", certificate_id: "123&redirect=https://evil.test",
          issuer: attack, matched_name: attack, query: attack, href: "javascript:alert(1)"},
        {source: "merklemap", subject_common_name: attack, href: "https://evil.test/"},
        {source: attack, certificate_id: "123", href: "javascript:alert(1)"},
      ],
    },
  });
  const requests = ui.requests.length;
  showDnsNodes(ui, [node]);
  const panel = originPanel(ui);
  assert(panel.textContent.includes(attack));
  const children = descendants(panel);
  assert(!children.some((child) => ["img", "script", "iframe"].includes(child.tagName)));
  const links = children.filter((child) => child.tagName === "a");
  assert.equal(links.length, 1);
  for (const link of links) {
    const url = new URL(link.attributes.href);
    assert.equal(url.protocol, "https:");
    assert.equal(url.hostname, "crt.sh");
    assert.equal(url.searchParams.get("q"), node.value);
    assert.equal(url.searchParams.get("redirect"), null);
    assert.equal(url.searchParams.get("id"), null);
    assert.equal(link.attributes.rel, "noopener noreferrer");
  }
  assert.equal(ui.requests.length, requests);
});

test("missing individual evidence and invalid dates never become invented check results", async () => {
  const ui = await fixture();
  showDnsNodes(ui, [dnsNode({dns_evidence: {
    checked_at: "not-a-date", duration_ms: "12", records: {}, wildcard: {},
  }})]);
  const details = detailRows(ui);
  assert.equal(details["Last DNS check"], "Not recorded");
  assert.equal(details["Lookup duration"], "Not recorded");
  assert.equal(details["Wildcard check"], "Not recorded");
  assert.match(details["AAAA (IPv6)"], /^Check: Not recorded/);
});

test("old servers hide vocabulary breadth and never receive its option", async () => {
  const ui = await fixture();
  assert.equal(ui.node("vocabulary-breadth-option").hidden, true);
  ui.credentials.minimax = true;
  ui.state.credentials = { ...ui.credentials };
  ui.node("provider").value = "minimax";
  ui.node("model").value = "MiniMax-M2.7";
  ui.node("budget").value = "1";
  // A retained checkbox value must not add an unsupported field to old APIs.
  ui.node("vocabulary-breadth").checked = true;
  ui.node("target").value = "api.example.com";
  await ui.api.resolveInput();
  await ui.trigger("scan-form", "submit");
  const request = ui.requests.find((item) => item.url === "/api/scans");
  assert(request);
  assert.equal(Object.hasOwn(request.body.options, "vocabulary_breadth"), false);
});

test("old servers ignore retained open-ended choices and keep finite limits", async () => {
  const ui = await fixture();
  assert.equal(ui.node("open-ended-option").hidden, true);
  assert.equal(ui.node("resume-open-ended-option").hidden, true);
  ui.node("run-until-paused").checked = true;
  await ui.trigger("run-until-paused", "change");
  assert.equal(ui.node("candidates").disabled, false);
  ui.node("target").value = "example.com";
  await ui.api.resolveInput();
  await ui.trigger("scan-form", "submit");
  const options = ui.requests.find((item) => item.url === "/api/scans").body.options;
  assert.equal(options.max_candidates, 1000);
  assert.equal(options.max_rounds, 10);
  assert.equal(options.time_limit, 300);
  assert.equal(Object.hasOwn(options, "stale_rounds"), false);
  ui.state.scans = [];
  ui.state.scan = {
    id: "fake-scan", target: "example.com", status: "completed", provider: "none",
    options, summary: { attempted: 10, round: 1, verified: 1 },
  };
  await ui.trigger("resume", "click");
  ui.node("resume-run-until-paused").checked = true;
  await ui.trigger("resume-run-until-paused", "change");
  await ui.trigger("resume-form", "submit");
  assert.deepEqual(ui.requests.find((item) => item.url.endsWith("/resume")).body.options, {
    max_candidates: 1000, max_rounds: 10, time_limit: 300, find: null,
  });
});

test("open-ended start preserves finite choices, finding goal, and separate budgets", async () => {
  const ui = await fixture({ features: { open_ended_scans: true } });
  ui.credentials.minimax = true;
  ui.credentials.jev = true;
  ui.state.credentials = { ...ui.credentials };
  ui.node("provider").value = "minimax";
  ui.node("budget").value = "2000";
  ui.node("prediction-provider").value = "jev";
  ui.node("prediction-budget").value = "1500";
  ui.node("candidates").value = "60000";
  ui.node("rounds").value = "200";
  ui.node("duration").value = "2000";
  ui.node("goal").value = "55000";
  ui.node("run-until-paused").checked = true;
  await ui.trigger("run-until-paused", "change");
  for (const id of ["candidates", "rounds", "duration"])
    assert.equal(ui.node(id).disabled, true);
  assert.equal(ui.node("goal").disabled, false);
  ui.node("run-until-paused").checked = false;
  await ui.trigger("run-until-paused", "change");
  assert.equal(ui.node("candidates").value, "60000");
  assert.equal(ui.node("rounds").value, "200");
  assert.equal(ui.node("duration").value, "2000");
  ui.node("run-until-paused").checked = true;
  await ui.trigger("run-until-paused", "change");
  ui.node("target").value = "example.com";
  await ui.api.resolveInput();
  await ui.trigger("scan-form", "submit");
  const request = ui.requests.find((item) => item.url === "/api/scans");
  assert.equal(request.body.options.max_candidates, 0);
  assert.equal(request.body.options.max_rounds, 0);
  assert.equal(request.body.options.time_limit, 0);
  assert.equal(request.body.options.stale_rounds, 0);
  assert.equal(request.body.options.find, 55000);
  assert.equal(request.body.budget_usd, 2000);
  assert.equal(request.body.prediction_budget_usd, 1500);
  assert.equal(ui.node("metric-stop").textContent, "No time limit");
});

for (const [provider, checked, expected] of [
  ["minimax", false, false],
  ["minimax", true, true],
  ["none", false, false],
  ["none", true, false],
]) {
  test(`vocabulary capability: provider=${provider}, checked=${checked} sends ${expected}`, async () => {
    const ui = await fixture({ features: { vocabulary_breadth: true } });
    assert.equal(ui.node("vocabulary-breadth-option").hidden, false);
    assert.equal(ui.node("vocabulary-breadth").checked, false);
    ui.credentials.minimax = true;
    ui.state.credentials = { ...ui.credentials };
    ui.node("provider").value = provider;
    ui.node("model").value = "MiniMax-M2.7";
    ui.node("budget").value = "1";
    ui.node("vocabulary-breadth").checked = checked;
    ui.node("target").value = "api.example.com";
    await ui.api.resolveInput();
    await ui.trigger("scan-form", "submit");
    const request = ui.requests.find((item) => item.url === "/api/scans");
    assert(request);
    assert.equal(request.body.options.vocabulary_breadth, expected);
  });
}

test("method accounting renders safely, separates unresolved, and supports old snapshots", async () => {
  const ui = await fixture();
  ui.state.scan = {summary: {discovery_methods: [
    {method: "numeric", pending: 3, checked: 12, verified: 7, miss: 2, inconclusive: 2, wildcard: 1},
    {method: "model", pending: 1, checked: 4, verified: 1, miss: 3},
    {method: "<img src=x onerror=alert(1)>", verified: 999},
    {method: "numeric", verified: 999},
  ]}};
  ui.api.renderDiscoveryMethods();
  const rows = ui.node("discovery-method-rows").children;
  assert.equal(rows.length, 2);
  assert.deepEqual(rows[0].children.map(x => x.textContent),
    ["Numeric patterns", "3", "12", "7", "2", "2", "1"]);
  assert.deepEqual(rows[1].children.map(x => x.textContent),
    ["AI proposals", "1", "4", "1", "3", "0", "0"]);
  assert.equal(ui.node("discovery-method-empty").hidden, true);
  ui.state.scan = {summary: {verified: 3}};
  ui.api.renderDiscoveryMethods();
  assert.equal(ui.node("discovery-method-rows").children.length, 0);
  assert.equal(ui.node("discovery-method-empty").hidden, false);
});

test("Jev is opt-in with a separate key and additional budget", async () => {
  const ui = await fixture();
  assert.deepEqual(
    JSON.parse(JSON.stringify(ui.api.predictionStartSettings())),
    { prediction_provider: "none" },
  );
  ui.node("prediction-provider").value = "jev";
  await ui.trigger("prediction-provider", "change");
  assert.equal(ui.node("prediction-settings").hidden, false);
  assert.throws(
    () => ui.api.predictionStartSettings(),
    /separate TypeSafe API key/,
  );
  ui.state.credentials.jev = true;
  assert.deepEqual(
    JSON.parse(JSON.stringify(ui.api.predictionStartSettings())),
    { prediction_provider: "jev", prediction_budget_usd: 0.05 },
  );
  ui.node("prediction-budget").value = "-1";
  assert.throws(
    () => ui.api.predictionStartSettings(),
    /additional Jev prediction budget/,
  );
  ui.node("prediction-provider").value = "none";
  await ui.trigger("prediction-provider", "change");
  assert.equal(ui.node("prediction-budget").disabled, true);
  assert.equal(ui.node("prediction-budget").required, false);
});

test("saving a synthetic TypeSafe key clears the field and never changes generation provider", async () => {
  const ui = await fixture();
  ui.node("provider").value = "minimax";
  ui.node("prediction-api-key").value = "TEST_ONLY_TYPESAFE_CREDENTIAL";
  await ui.trigger("save-prediction-key", "click");
  assert.equal(ui.node("prediction-api-key").value, "");
  const request = ui.requests.find((item) => item.url === "/api/credentials");
  assert.deepEqual(request.body, {
    provider: "jev",
    api_key: "TEST_ONLY_TYPESAFE_CREDENTIAL",
  });
  assert.equal(ui.node("provider").value, "minimax");
  assert.equal(ui.node("prediction-credential-status").textContent, "READY");
  await ui.trigger("clear-prediction-key", "click");
  assert.deepEqual(
    ui.requests.filter((item) => item.url === "/api/credentials").at(-1).body,
    { provider: "jev", api_key: "" },
  );
});

test("start sends independent generation and prediction budgets", async () => {
  const ui = await fixture();
  ui.credentials.minimax = true;
  ui.credentials.jev = true;
  ui.state.credentials = { ...ui.credentials };
  ui.node("provider").value = "minimax";
  ui.node("model").value = "MiniMax-M2.7";
  ui.node("budget").value = "1.25";
  ui.node("prediction-provider").value = "jev";
  ui.node("prediction-budget").value = "0.07";
  ui.node("target").value = "api.example.com";
  await ui.api.resolveInput();
  await ui.trigger("scan-form", "submit");
  const request = ui.requests.find((item) => item.url === "/api/scans");
  assert(request);
  assert.equal(request.body.provider, "minimax");
  assert.equal(request.body.model, "MiniMax-M2.7");
  assert.equal(request.body.budget_usd, 1.25);
  assert.equal(request.body.prediction_budget_usd, 0.07);
  assert.equal(request.body.prediction_provider, "jev");
});

test("prediction rendering separates costs, preserves plain text, and labels fallback", async () => {
  const ui = await fixture();
  ui.state.scan = {
    id: "fake-scan",
    target: "example.com",
    status: "paused",
    provider: "none",
    options: {},
    summary: {
      predictor: { name: "jev", model: "jev-1.13.0" },
      ledger: { spent_usd: "0.20", limit_usd: "1" },
      prediction_ledger: {
        spent_usd: "0.003",
        limit_usd: "0.05",
        reserved_usd: "0.001",
      },
      predictions: [
        {
          round: 1,
          plan_id: "plan-a",
          description: "<script>not executable</script>",
          score: 0.8,
          status: "scored",
          count: 12,
        },
        {
          round: 1,
          plan_id: "plan-b",
          description: "Other plan",
          score: null,
          status: "fallback",
          count: 3,
        },
      ],
    },
  };
  ui.api.renderSummary();
  ui.api.renderPredictions();
  assert.equal(ui.node("metric-spent").textContent, "$0.20");
  assert.equal(ui.node("prediction-spent").textContent, "$0.0030");
  assert.equal(
    ui.node("prediction-cap").textContent,
    "of $0.05 prediction cap",
  );
  assert.equal(ui.node("prediction-reserved").textContent, "$0.0010 reserved");
  assert.match(ui.node("predictions-list").textContent, /Priority 0\.800/);
  assert.match(
    ui.node("predictions-list").textContent,
    /not DNS existence probability/,
  );
  assert.match(
    ui.node("predictions-list").textContent,
    /Not scored · fallback/,
  );
  assert.match(
    ui.node("predictions-list").textContent,
    /<script>not executable<\/script>/,
  );
  assert(!ui.node("predictions-list").textContent.includes("80%"));
  ui.state.scan.summary.predictions = Array.from({ length: 90 }, (_, i) => ({
    round: i,
    score: 0.2,
    status: "scored",
    count: 1,
    description: "Plan",
  }));
  ui.api.renderPredictions();
  assert.equal(
    ui
      .node("predictions-list")
      .children.filter((child) => child.tagName === "article").length,
    60,
  );
});

test("Jev resume edits cumulative prediction budget without changing predictor identity", async () => {
  const ui = await fixture();
  ui.state.selectedId = "fake-scan";
  ui.state.scan = {
    id: "fake-scan",
    target: "example.com",
    status: "paused",
    provider: "none",
    options: {
      max_candidates: 100,
      find: null,
      max_rounds: 10,
      time_limit: 60,
    },
    summary: {
      attempted: 20,
      verified: 3,
      round: 1,
      predictor: { name: "jev", model: "jev-1.13.0" },
      prediction_ledger: {
        spent_usd: "0.02",
        reserved_usd: "0",
        limit_usd: "0.05",
      },
    },
  };
  await ui.trigger("resume", "click");
  assert.equal(ui.node("resume-dialog").open, true);
  assert.equal(ui.node("resume-prediction-budget").value, 0.05);
  ui.node("resume-prediction-budget").value = "0.01";
  await ui.trigger("resume-form", "submit");
  assert.match(ui.node("resume-error").textContent, /cannot decrease/);
  assert(!ui.requests.some((item) => item.url.endsWith("/resume")));
  ui.node("resume-prediction-budget").value = "0.08";
  await ui.trigger("resume-form", "submit");
  const request = ui.requests.find((item) => item.url.endsWith("/resume"));
  assert.equal(request.body.prediction_budget_usd, 0.08);
  assert.equal(request.body.prediction_provider, undefined);
  assert.equal(request.body.provider, undefined);
  assert.equal(request.body.budget_usd, undefined);
});

test("paused scans can remove reached work caps but must cover cumulative spend and goals", async () => {
  const ui = await fixture({ features: { open_ended_scans: true } });
  ui.state.selectedId = "fake-scan";
  ui.state.scan = {
    id: "fake-scan", target: "example.com", status: "paused", provider: "minimax",
    options: { max_candidates: 1000, max_rounds: 10, time_limit: 300, find: 5, stale_rounds: 2 },
    summary: {
      attempted: 1000, round: 10, verified: 5,
      ledger: { limit_usd: 1, spent_usd: 0.8, reserved_usd: 0.1 },
    },
  };
  await ui.trigger("resume", "click");
  assert.equal(ui.node("resume-dialog").open, true);
  assert(!ui.requests.some((item) => item.url.endsWith("/resume")));
  ui.node("resume-run-until-paused").checked = true;
  await ui.trigger("resume-run-until-paused", "change");
  await ui.trigger("resume-form", "submit");
  assert.match(ui.node("resume-error").textContent, /finding goal/);
  ui.node("resume-goal").value = "";
  ui.node("resume-budget").value = "0.5";
  await ui.trigger("resume-form", "submit");
  assert.match(ui.node("resume-error").textContent, /previous spending/);
  assert(!ui.requests.some((item) => item.url.endsWith("/resume")));
  ui.node("resume-budget").value = "2000";
  await ui.trigger("resume-form", "submit");
  const request = ui.requests.find((item) => item.url.endsWith("/resume"));
  assert.deepEqual(request.body.options, {
    max_candidates: 0, max_rounds: 0, time_limit: 0, stale_rounds: 0, find: null,
  });
  assert.equal(request.body.budget_usd, 2000);
});

test("saved open-ended sessions resume unchanged and offer usable finite defaults", async () => {
  const ui = await fixture({ features: { open_ended_scans: true } });
  ui.state.selectedId = "fake-scan";
  const scan = {
    id: "fake-scan", target: "example.com", status: "completed", provider: "none",
    options: { max_candidates: 0, max_rounds: 0, time_limit: 0, stale_rounds: 0, find: null },
    summary: { attempted: 60000, round: 200, verified: 5 },
  };
  ui.state.scan = scan;
  await ui.trigger("resume", "click");
  assert.equal(ui.node("resume-run-until-paused").checked, true);
  assert.equal(ui.node("resume-candidates").disabled, true);
  await ui.trigger("resume-form", "submit");
  assert.deepEqual(ui.requests.find((item) => item.url.endsWith("/resume")).body.options, scan.options);
  ui.state.scan = scan;
  await ui.trigger("resume", "click");
  ui.node("resume-run-until-paused").checked = false;
  await ui.trigger("resume-run-until-paused", "change");
  assert.equal(ui.node("resume-candidates").disabled, false);
  assert(Number(ui.node("resume-candidates").value) > scan.summary.attempted);
  assert(Number(ui.node("resume-rounds").value) > scan.summary.round);
  assert(Number(ui.node("resume-duration").value) > 0);
  await ui.trigger("resume-form", "submit");
  const options = ui.requests.filter((item) => item.url.endsWith("/resume")).at(-1).body.options;
  assert(options.max_candidates > scan.summary.attempted);
  assert(options.max_rounds > scan.summary.round);
  assert(options.time_limit > 0);
});

test("mixed zero limits survive resume while positive cumulative caps still apply", async () => {
  const ui = await fixture({ features: { open_ended_scans: true } });
  ui.state.selectedId = "fake-scan";
  ui.state.scan = {
    id: "fake-scan", target: "example.com", status: "paused", provider: "none",
    options: { max_candidates: 0, max_rounds: 200, time_limit: 0, find: null },
    summary: { attempted: 60000, round: 200, verified: 5 },
  };
  await ui.trigger("resume", "click");
  assert.equal(ui.node("resume-run-until-paused").checked, false);
  await ui.trigger("resume-form", "submit");
  assert.match(ui.node("resume-error").textContent, /round cap/);
  assert(!ui.requests.some((item) => item.url.endsWith("/resume")));
  ui.node("resume-rounds").value = "300";
  await ui.trigger("resume-form", "submit");
  assert.deepEqual(ui.requests.find((item) => item.url.endsWith("/resume")).body.options, {
    max_candidates: 0, max_rounds: 300, time_limit: 0, find: null,
  });
});
