// Vinted Hunter UI. Vanilla JS, no build step.
// All server-provided text is inserted with textContent or as attribute values after
// scheme checks. Nothing is written with innerHTML.

(() => {
  "use strict";

  const STAGES = [
    { id: "plan", label: "Plan" },
    { id: "search", label: "Search" },
    { id: "triage", label: "Triage" },
    { id: "details", label: "Details" },
    { id: "extract", label: "Vision" },
    { id: "specs", label: "Specs" },
    { id: "rank", label: "Rank" },
  ];
  const VERDICT_LABEL = { top_pick: "Top pick", good: "Good", ok: "OK", avoid: "Avoid" };
  // Hard-requirement status, computed by the server. Older saved hunts may lack it.
  const REQUIREMENT_BADGE = {
    verified: { label: "\u2713 v\u00e9rifi\u00e9", title: "Every hard requirement is confirmed" },
    unverified: { label: "? non v\u00e9rifi\u00e9", title: "Nothing is ruled out, but some requirement is unconfirmed" },
    failed: { label: "\u2717 \u00e9choue", title: "A hard requirement is contradicted" },
  };
  const eur = new Intl.NumberFormat("fr-FR", { style: "currency", currency: "EUR" });
  const dateFmt = new Intl.DateTimeFormat("en-GB", { dateStyle: "medium", timeStyle: "short" });

  const $ = (id) => document.getElementById(id);
  const form = $("hunt-form");
  const submitBtn = $("submit");
  const progressBox = $("progress");
  const stepper = $("stepper");
  const progressNote = $("progress-note");
  const logList = $("log");
  const resultsBox = $("results");
  const cardsBox = $("cards");
  let stream = null;
  let running = false;

  // --- DOM helpers ----------------------------------------------------------

  function el(tag, props = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (value === undefined || value === null || value === false) continue;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key === "style") Object.assign(node.style, value);
      else node.setAttribute(key, value === true ? "" : String(value));
    }
    for (const child of children.flat()) {
      if (child === null || child === undefined || child === false) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  function safeUrl(url) {
    if (typeof url !== "string") return null;
    try {
      const parsed = new URL(url);
      return parsed.protocol === "https:" ? parsed.href : null;
    } catch {
      return null;
    }
  }

  function percent(value) {
    return value === null || value === undefined ? "n/a" : `${Math.round(value * 100)}%`;
  }

  function money(value) {
    return value === null || value === undefined ? "n/a" : eur.format(value);
  }

  // Shipping as Vinted's default option shows it, with the landed total when the shipping cost is known.
  function shippingText(listing) {
    if (listing.pickup_only) return "retrait uniquement";
    if (listing.shipping_is_free) return "livraison gratuite";
    if (listing.shipping_price_eur === null || listing.shipping_price_eur === undefined) return "livraison inconnue";
    const landed =
      listing.landed_price_eur === null || listing.landed_price_eur === undefined
        ? ""
        : ` · total livré ≈ ${money(listing.landed_price_eur)}`;
    return `≈ ${money(listing.shipping_price_eur)} livraison (estimation Vinted)${landed}`;
  }

  // --- Progress -------------------------------------------------------------

  function resetProgress() {
    stepper.replaceChildren();
    for (const stage of STAGES) {
      stepper.append(
        el("li", { "data-stage": stage.id }, el("span", { class: "name", text: stage.label }), el("span", { class: "count" }))
      );
    }
    logList.replaceChildren();
    progressNote.textContent = "";
    progressBox.hidden = false;
  }

  function stageNode(id) {
    return stepper.querySelector(`li[data-stage="${id}"]`);
  }

  function setStage(id, status, detail) {
    const node = stageNode(id);
    if (!node) return;
    node.className = status === "start" ? "active" : status === "done" ? "done" : "";
    if (detail) node.querySelector(".count").textContent = detail;
  }

  function addLog(message, isError = false) {
    logList.append(el("li", { class: isError ? "error" : "", text: message }));
    while (logList.children.length > 200) logList.firstElementChild.remove();
    logList.scrollTop = logList.scrollHeight;
  }

  function handleEvent(event) {
    switch (event.type) {
      case "stage":
        setStage(event.stage, event.status, event.detail || "");
        if (event.status === "start" && event.detail) progressNote.textContent = event.detail;
        break;
      case "progress": {
        const node = stageNode(event.stage);
        if (node) node.querySelector(".count").textContent = `${event.done}/${event.total}`;
        break;
      }
      case "log":
        addLog(event.message);
        break;
      case "result":
        finish();
        renderResult(event.result);
        loadHistory();
        break;
      case "error":
        addLog(event.message, true);
        progressNote.replaceChildren(el("span", { class: "error-box", text: event.message }));
        document.querySelector("li.active")?.classList.add("error");
        finish();
        break;
      default:
        break;
    }
  }

  // While a hunt runs, the form and the past-hunt list are locked so a live stream is never replaced.
  function setRunning(value) {
    running = value;
    submitBtn.disabled = value;
    submitBtn.textContent = value ? "Hunting..." : "Hunt";
    for (const button of document.querySelectorAll("#history button")) {
      button.disabled = value;
    }
  }

  function finish() {
    if (stream) stream.close();
    stream = null;
    setRunning(false);
  }

  function openStream(huntId) {
    if (stream) stream.close();
    stream = new EventSource(`/api/hunts/${huntId}/events`);
    stream.onopen = () => {
      // Each (re)connection replays the full event history, so rebuild the view from scratch.
      resetProgress();
    };
    stream.onmessage = (message) => {
      let event;
      try {
        event = JSON.parse(message.data);
      } catch {
        return;
      }
      handleEvent(event);
    };
    stream.onerror = () => {
      // Browsers reconnect automatically. Only give up once the server has sent a terminal event.
      if (stream && stream.readyState === EventSource.CLOSED) {
        finish();
        addLog("Lost connection to the server.", true);
      }
    };
  }

  // --- Results --------------------------------------------------------------

  function renderResult(result) {
    resultsBox.hidden = false;
    renderPlan(result.plan);
    renderWarnings(result.warnings || []);
    renderSummary(result);
    renderCards(result.ranked || []);
    renderRejected(result.rejected || []);
    renderCost(result.cost);
    resultsBox.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderSummary(result) {
    const ranked = (result.ranked || []).length;
    const rejected = (result.rejected || []).length;
    $("results-summary").textContent =
      `${ranked} ranked, ${rejected} rejected, budget ${money(result.budget_eur)}, hunt ${dateFmt.format(new Date(result.created_at))}`;
    $("no-results").hidden = ranked > 0;
  }

  function renderPlan(plan) {
    const box = $("plan");
    box.replaceChildren();
    if (!plan) {
      box.hidden = true;
      return;
    }
    const list = (items) =>
      items && items.length ? el("ul", {}, items.map((i) => el("li", { text: i }))) : el("span", { text: "none" });
    // append() would write null as the text "null", so the optional rows are filtered out first.
    box.append(
      ...[
        el("div", {}, el("strong", { text: "Plan: " }), plan.product_summary || ""),
        plan.product_type ? el("div", {}, el("strong", { text: "Product type: " }), plan.product_type) : null,
        plan.not_product_types && plan.not_product_types.length
          ? el("div", {}, el("strong", { text: "Not this type: " }), plan.not_product_types.join(" \u00b7 "))
          : null,
        el("div", {}, el("strong", { text: "Hard requirements:" }), list((plan.hard_requirements || []).map((r) => r.text))),
        el("div", {}, el("strong", { text: "Specs researched: " }), (plan.spec_keys || []).join(" \u00b7 ") || "none"),
        el("div", {}, el("strong", { text: "Searches: " }), (plan.queries || []).join(" \u00b7 ")),
      ].filter(Boolean)
    );
    box.hidden = false;
  }

  function renderWarnings(warnings) {
    const box = $("warnings");
    box.replaceChildren(...warnings.map((w) => el("p", { text: w })));
    box.hidden = warnings.length === 0;
  }

  function renderCards(ranked) {
    cardsBox.replaceChildren(...ranked.map(renderCard));
  }

  function renderFoundSpecs(specs) {
    if (!specs) {
      return el("p", { class: "muted specs-missing", text: "Specs non trouvées en ligne." });
    }
    const rows = [];
    const addRow = (label, value) => {
      if (value !== null && value !== undefined && value !== "") rows.push(el("dt", { text: label }), el("dd", { text: String(value) }));
    };
    if (specs.variant_match === "base_model") addRow("Portée", "specs du modèle de base, pas cette variante");
    // Spec values are shown only for web specs of the exact variant. The backend withholds them from the
    // ranker for other specs too, so a base model's value is never shown as this variant's.
    const exactWeb = specs.lookup_mode === "web" && specs.variant_match === "exact";
    if (exactWeb && specs.specs) {
      for (const [name, value] of Object.entries(specs.specs)) addRow(name, value);
    }
    addRow("Notes prix", specs.price_notes || null);
    if (specs.lookup_mode === "model_knowledge") addRow("Source", "connaissances du modèle, non vérifié");
    const sources = (specs.sources || []).map(safeUrl).filter(Boolean);
    return el(
      "section",
      { class: "specs-found" },
      el("h4", { text: "Specs trouvées en ligne" }),
      rows.length ? el("dl", { class: "specs" }, rows) : el("p", { class: "muted", text: "Aucune donnée exploitable." }),
      sources.length
        ? el(
            "ul",
            { class: "sources" },
            sources.map((url) => el("li", {}, el("a", { href: url, target: "_blank", rel: "noopener noreferrer", text: new URL(url).hostname })))
          )
        : null
    );
  }

  // Prices of the card: the new price (web specs only, labelled for a base model), the typical used price, and how
  // much cheaper this listing is than the new price. Each part is shown only when the server has it.
  function priceLine(entry) {
    const parts = [];
    const newPrice = entry.new_price;
    if (newPrice && newPrice.eur !== null && newPrice.eur !== undefined) {
      parts.push(`Neuf \u2248 ${money(newPrice.eur)}${newPrice.base_model ? " (mod\u00e8le de base)" : ""}`);
    }
    const used = entry.specs && entry.specs.typical_used_price_eur;
    if (used) parts.push(`occasion typique \u2248 ${money(used)}`);
    if (newPrice && newPrice.discount_pct !== null && newPrice.discount_pct !== undefined) {
      const pct = newPrice.discount_pct;
      if (pct > 0) parts.push(`cette annonce \u2212${pct} % vs neuf`);
      else if (pct < 0) parts.push(`cette annonce +${-pct} % vs neuf`);
      else parts.push("cette annonce au prix du neuf");
    }
    return parts.length ? parts.join(" \u00b7 ") : null;
  }

  // Defect and product-type notes from the photo check. Shown only when there is something to say.
  function defectNotes(extraction) {
    if (!extraction) return [];
    const notes = [];
    if (extraction.is_requested_product_type === false) {
      const evidence = extraction.product_type_evidence ? ` : ${extraction.product_type_evidence}` : "";
      notes.push(el("p", { class: "defect unusable", text: `Pas le type de produit demand\u00e9${evidence}` }));
    }
    const severity = extraction.defect_severity;
    if (severity === "functional" || severity === "unusable") {
      const label = severity === "unusable" ? "Article inutilisable" : "D\u00e9faut fonctionnel";
      notes.push(el("p", { class: `defect ${severity}`, text: extraction.defect_note ? `${label} : ${extraction.defect_note}` : label }));
    }
    return notes;
  }

  function renderCard(entry) {
    const { listing, ranking, specs, extraction } = entry;
    const img = safeUrl(listing.photo_url);
    const link = safeUrl(listing.url);
    const identified = extraction && extraction.model;
    const ident = identified
      ? el(
          "p",
          { class: "ident" },
          `${extraction.brand ? `${extraction.brand} ` : ""}${extraction.model} `,
          el("span", { class: "muted", text: `(confidence ${percent(extraction.model_confidence)})` })
        )
      : el("p", { class: "ident" }, el("span", { class: "unknown", text: "Model not identified" }));

    // Specs researched online for the identified model. Never derived from the listing text.
    const specsBlock = renderFoundSpecs(specs);
    const checks = (extraction && extraction.requirement_checks) || [];
    // "none" means the buyer stated no hard requirements: no badge at all.
    const status = REQUIREMENT_BADGE[entry.requirements_status] ? entry.requirements_status : "unverified";
    const badge = entry.requirements_status === "none" ? null : REQUIREMENT_BADGE[status];
    const sellerLine = [
      listing.condition,
      listing.seller_rating !== null && listing.seller_rating !== undefined
        ? `seller ${percent(listing.seller_rating)}${listing.seller_review_count ? ` (${listing.seller_review_count} reviews)` : ""}`
        : null,
      listing.upload_info,
    ]
      .filter(Boolean)
      .join(" · ");

    return el(
      "article",
      { class: "card" },
      el(
        "div",
        { class: "thumb" },
        img ? el("img", { src: img, alt: "", loading: "lazy", referrerpolicy: "no-referrer" }) : "no photo"
      ),
      el(
        "div",
        { class: "body" },
        el(
          "div",
          { class: "top" },
          el("span", { class: `badge ${ranking.verdict}`, text: VERDICT_LABEL[ranking.verdict] || ranking.verdict }),
          badge ? el("span", { class: `req ${status}`, title: badge.title, text: badge.label }) : null,
          el("span", { class: "rank", text: `#${entry.rank}` })
        ),
        el("h3", { text: listing.title }),
        el(
          "p",
          { class: "price" },
          money(listing.total_price_eur),
          el("small", { text: `  incl. fees (item ${money(listing.price_eur)})` })
        ),
        el("p", { class: "shipping", text: shippingText(listing) }),
        ident,
        ...defectNotes(extraction),
        priceLine(entry) ? el("p", { class: "price-line", text: priceLine(entry) }) : null,
        el(
          "div",
          { class: "score" },
          el("div", { class: "track" }, el("div", { class: "bar", style: { width: `${Math.max(0, Math.min(100, ranking.score))}%` } })),
          el("span", { text: `${ranking.score}` })
        ),
        el("p", { class: "headline", text: ranking.headline }),
        ranking.model_verdict ? el("p", { class: "model-verdict", text: ranking.model_verdict }) : null,
        el("p", { class: "rationale muted", text: ranking.value_rationale }),
        specsBlock,
        el(
          "div",
          { class: "pc" },
          ranking.pros && ranking.pros.length
            ? el("div", {}, el("h4", { text: "Pros" }), el("ul", { class: "pros" }, ranking.pros.map((p) => el("li", { text: p }))))
            : null,
          ranking.cons && ranking.cons.length
            ? el("div", {}, el("h4", { text: "Cons" }), el("ul", { class: "cons" }, ranking.cons.map((c) => el("li", { text: c }))))
            : null
        ),
        extraction && extraction.red_flags && extraction.red_flags.length
          ? el("ul", { class: "flags" }, extraction.red_flags.map((f) => el("li", { text: f })))
          : null,
        checks.length
          ? el(
              "ul",
              { class: "checks", "aria-label": "Requirement checks" },
              checks.map((c) => el("li", { class: c.status, title: c.evidence || "", text: `${c.requirement}: ${c.status}` }))
            )
          : null,
        el("p", { class: "meta", text: sellerLine }),
        link ? el("a", { class: "view", href: link, target: "_blank", rel: "noopener noreferrer", text: "Voir sur Vinted" }) : null
      )
    );
  }

  function renderRejected(rejected) {
    $("rejected-count").textContent = `(${rejected.length})`;
    $("rejected-list").replaceChildren(
      ...rejected.map((item) => {
        const link = safeUrl(item.url);
        return el(
          "li",
          {},
          link
            ? el("a", { href: link, target: "_blank", rel: "noopener noreferrer", text: item.title })
            : el("span", { text: item.title }),
          el("span", { class: "stage", text: item.stage }),
          el("div", { class: "muted", text: item.reason })
        );
      })
    );
  }

  function renderCost(cost) {
    if (!cost) return;
    const perStage = Object.entries(cost.stages || {})
      .map(([name, s]) => `${name} ${s.priced ? `$${s.usd.toFixed(4)}` : s.usd > 0 ? `$${s.usd.toFixed(4)}+ (partly unpriced)` : "unpriced"}`)
      .join(", ");
    const web = cost.total_web_search_requests ? `, ${cost.total_web_search_requests} web searches (fees not included)` : "";
    // The Claude CLI backend reports what the same calls would cost at API prices. The
    // subscription covers that, so it is not money spent.
    const label =
      cost.basis === "cli_reported"
        ? "Equivalent API cost, covered by your subscription"
        : "Estimated API cost";
    // Some calls reported no cost: the total is then a lower bound, not a complete figure.
    const total = cost.priced === false ? `at least $${cost.total_usd.toFixed(4)}` : `$${cost.total_usd.toFixed(4)}`;
    $("cost").textContent =
      `${label}: ${total} over ${cost.total_calls} calls` +
      (cost.priced === false ? " (some calls reported no cost)" : "") +
      (perStage ? ` (${perStage})` : "") +
      web +
      ".";
  }

  // --- Form and history -----------------------------------------------------

  async function submitHunt(event) {
    event.preventDefault();
    const request = $("request").value.trim();
    const budget = Number($("budget").value);
    if (!request || !(budget > 0)) return;
    const options = {};
    const maxDetailed = $("max-detailed").value;
    const pages = $("search-pages").value;
    if (maxDetailed) options.max_detailed = Number(maxDetailed);
    if (pages) options.search_pages = Number(pages);

    if (running) return;
    setRunning(true);
    resultsBox.hidden = true;
    resetProgress();
    try {
      const response = await fetch("/api/hunts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ request, budget, options: Object.keys(options).length ? options : null }),
      });
      if (!response.ok) {
        let detail = `Server returned ${response.status}`;
        try {
          const body = await response.json();
          if (typeof body.detail === "string") detail = body.detail;
        } catch {
          /* keep the generic message */
        }
        throw new Error(detail);
      }
      const { hunt_id: huntId } = await response.json();
      openStream(huntId);
      loadHistory();
    } catch (err) {
      addLog(`Could not start the hunt: ${err.message}`, true);
      progressBox.hidden = false;
      finish();
    }
  }

  async function loadHistory() {
    const list = $("history");
    try {
      const response = await fetch("/api/hunts");
      const hunts = await response.json();
      $("history-empty").hidden = hunts.length > 0;
      list.replaceChildren(
        ...hunts.map((h) =>
          el(
            "li",
            {},
            el(
              "button",
              { type: "button", "data-hunt": h.hunt_id, disabled: running },
              el("span", { class: "title", text: h.request || "(untitled)" }),
              el("span", {
                class: "sub",
                text: `${money(h.budget_eur)} · ${h.ranked_count} ranked${h.top_title ? ` · top: ${h.top_title}` : ""}`,
              }),
              el("span", { class: "sub", text: h.created_at ? dateFmt.format(new Date(h.created_at)) : "" })
            )
          )
        )
      );
    } catch {
      $("history-empty").hidden = false;
    }
  }

  async function openPastHunt(huntId) {
    if (running) return;
    progressBox.hidden = true;
    try {
      const response = await fetch(`/api/hunts/${encodeURIComponent(huntId)}`);
      if (!response.ok) throw new Error(`Server returned ${response.status}`);
      renderResult(await response.json());
    } catch (err) {
      alert(`Could not open that hunt: ${err.message}`);
    }
  }

  // Shows a setup hint when the chosen backend cannot run. The API backend needs the key;
  // the Claude CLI backend needs the binary and a login.
  function setupProblem(data) {
    if (data.backend === "claude-cli") {
      if (!data.cli || !data.cli.found) {
        return "Claude Code CLI not found. Install Claude Code and run `claude` once to log in.";
      }
      if (data.cli.logged_in === false) {
        return "Not logged in to Claude Code. Run `claude` once to log in.";
      }
      return null;
    }
    return data.api_key_configured ? null : "ANTHROPIC_API_KEY is not set. Add it to a .env file in the project root and restart the server.";
  }

  async function checkHealth() {
    try {
      const response = await fetch("/api/health");
      const data = await response.json();
      const problem = setupProblem(data);
      $("key-banner").textContent = problem || "";
      $("key-banner").hidden = !problem;
    } catch {
      /* ignore */
    }
  }

  form.addEventListener("submit", submitHunt);
  $("history").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-hunt]");
    if (button) openPastHunt(button.getAttribute("data-hunt"));
  });

  checkHealth();
  loadHistory();
})();
