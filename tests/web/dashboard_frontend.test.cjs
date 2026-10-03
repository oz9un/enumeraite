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
  scrollIntoView() {}
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
    "globalThis.__dashboardTest={state,renderSummary,renderPredictions,renderDiscoveryMethods,predictionStartSettings,resolveInput,renderCredentials};\n})();",
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
