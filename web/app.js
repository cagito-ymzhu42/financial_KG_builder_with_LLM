import {
  buildDataset,
  filterRelations,
  connectedGroups,
  safeSourceUrl,
} from "./model.mjs";

const $ = (selector) => document.querySelector(selector);
const colors = ["#749c68", "#b89459", "#6e91ad", "#a388a0", "#7e9b99"];
const state = {
  data: null,
  cluster: "all",
  selected: null,
  article: 0,
  view: "overview",
  dataset: "example",
  loading: false,
};
const format = (value) => String(value ?? "").replaceAll("_", " ");
const escape = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (char) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        char
      ],
  );
const colorFor = (cluster) =>
  cluster === null
    ? "#89998b"
    : colors[Math.max(0, state.data.clusters.indexOf(cluster)) % colors.length];
const visibleRelations = () =>
  state.cluster === "all"
    ? state.data.relations
    : state.data.relations.filter((row) => row.cluster === state.cluster);

async function fetchText(url, optional = false) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    if (optional && response.status === 404) return "";
    throw new Error(`Could not load ${url} (HTTP ${response.status})`);
  }
  return response.text();
}

async function loadDataset(value) {
  if (state.loading) return;
  state.loading = true;
  $("#dataset").disabled = true;
  $("#error").hidden = true;
  try {
    const directory =
      value === "example" ? "../examples/expected" : "../results";
    const jsonl = await fetchText(`${directory}/all_output.jsonl`);
    const csv = await fetchText(`${directory}/relation_clusters.csv`, true);
    const data = buildDataset(jsonl, csv);
    $("#export").href = `${directory}/relations.json`;
    state.data = data;
    state.dataset = value;
    state.cluster = "all";
    state.article = 0;
    state.selected = data.entities[0] || null;
    $("#search").value = "";
    $("#notice").innerHTML = `<span>ⓘ</span><span>${
      value === "example"
        ? "Example dataset · Real archived excerpts, manually prepared relationships, computed clusters."
        : "Local results · Displaying exported files from results/. No model calls or database writes."
    }${!csv ? " No cluster file available." : ""}</span>`;
    $("#article-count").textContent = data.articles.length;
    $("#entity-count").textContent = data.entities.length;
    $("#relation-count").textContent = data.relations.length;
    $("#cluster-count").textContent = data.clusters.length;
    renderFilters();
    renderGraph();
    renderInspector();
    renderArticles();
    renderRelations();
  } catch (error) {
    $("#error").textContent =
      `${error.message}.${value === "results" ? "Run the Python pipeline to create results/all_output.jsonl, or explore the examples." : "Start an HTTP server from the project root, then open /web/."}`;
    $("#error").hidden = false;
    $("#dataset").value = state.dataset;
    if (!state.data)
      $("#notice").innerHTML =
        "<span>ⓘ</span><span>No data loaded. See the error below.</span>";
  } finally {
    state.loading = false;
    $("#dataset").disabled = false;
  }
}

function setView(view) {
  state.view = view;
  const titles = {
    overview: [
      "Overview",
      "Financial knowledge, connected",
      "Explore entities, trace sources, and see how financial concepts connect.",
    ],
    articles: [
      "Source articles",
      "Back to the source",
      "Read summaries alongside source text. Give every relationship context.",
    ],
    relations: [
      "Relationships",
      "From concepts to connections",
      "Search entities and relationship types, with clusters and sources in view.",
    ],
  };
  document.querySelectorAll(".view").forEach((viewElement) => {
    viewElement.hidden = viewElement.id !== `${view}-view`;
  });
  document.querySelectorAll("[data-view]").forEach((button) => {
    const active = button.dataset.view === view;
    button.classList.toggle("active", active);
    if (active) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  $("#breadcrumb").textContent = titles[view][0];
  $("#page-title").innerHTML = `${titles[view][1]}<span>.</span>`;
  $("#page-subtitle").textContent = titles[view][2];
  window.scrollTo({ top: 0, left: 0 });
}

function renderFilters() {
  $("#cluster-filters").innerHTML =
    `<button class="chip ${state.cluster === "all" ? "active" : ""}" data-cluster="all" aria-pressed="${state.cluster === "all"}">All relationships</button>` +
    state.data.clusters
      .map(
        (cluster) =>
          `<button class="chip ${state.cluster === cluster ? "active" : ""}" data-cluster="${escape(cluster)}" aria-pressed="${state.cluster === cluster}"><i class="dot" style="--cluster-color:${colorFor(cluster)}"></i>Cluster ${escape(cluster)}</button>`,
      )
      .join("");
  $("#cluster-filters")
    .querySelectorAll("button")
    .forEach((button) =>
      button.addEventListener("click", () => {
        state.cluster = button.dataset.cluster;
        const rows = visibleRelations();
        if (
          !rows.some(
            (row) =>
              row.source === state.selected || row.target === state.selected,
          )
        )
          state.selected = rows[0]?.source || null;
        renderFilters();
        renderGraph();
        renderInspector();
      }),
    );
}

function wrapName(name) {
  const words = format(name).split(" "),
    lines = [];
  let line = "";
  for (const word of words) {
    if (line && (line + " " + word).length > 21) {
      lines.push(line);
      line = word;
    } else line += (line ? " " : "") + word;
  }
  if (line) lines.push(line);
  return lines;
}

function renderGraph() {
  const rows = visibleRelations(),
    groups = connectedGroups(rows),
    positions = new Map();
  const radiusFor = (name) =>
    Math.min(
      28,
      10 +
        rows.filter((row) => row.source === name || row.target === name)
          .length *
          4,
    );
  const columns = Math.min(3, Math.max(1, groups.length)),
    cellWidth = 1000 / columns;
  const height = Math.max(460, Math.ceil(groups.length / columns) * 460);
  const graph = $("#graph");
  graph.setAttribute("viewBox", `0 0 1000 ${height}`);
  graph.style.height =
    groups.length > 3 ? `${Math.ceil(groups.length / 3) * 360}px` : "";
  $("#graph-count").textContent =
    `${new Set(rows.flatMap((row) => [row.source, row.target])).size} entities · ${rows.length} relationships`;
  let markup = `<defs>${colors
    .concat("#89998b")
    .map(
      (color, index) =>
        `<marker id="arrow-${index}" markerWidth="7" markerHeight="7" refX="7" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7" fill="${color}"/></marker>`,
    )
    .join("")}</defs>`;
  groups.forEach((group, index) => {
    const cx = (index % columns) * cellWidth + cellWidth / 2,
      cy = Math.floor(index / columns) * 460 + 215;
    const cluster =
      rows.find((row) => row.source === group[0] || row.target === group[0])
        ?.cluster ?? null;
    const color = colorFor(cluster);
    markup += `<circle cx="${cx}" cy="${cy}" r="148" fill="${color}" opacity=".035"/><text x="${cx}" y="${cy - 180}" text-anchor="middle" class="group-label">CONCEPT GROUP ${String(index + 1).padStart(2, "0")}</text>`;
    positions.set(group[0], {
      x: cx,
      y: cy,
      radius: radiusFor(group[0]),
      color,
    });
    group.slice(1).forEach((name, nodeIndex) => {
      const angle =
        -Math.PI / 2 + (nodeIndex * 2 * Math.PI) / (group.length - 1);
      positions.set(name, {
        x: cx + Math.cos(angle) * 118,
        y: cy + Math.sin(angle) * 135,
        radius: radiusFor(name),
        color,
      });
    });
  });
  rows.forEach((row) => {
    const source = positions.get(row.source),
      target = positions.get(row.target);
    const color = colorFor(row.cluster),
      markerIndex = colors.concat("#89998b").indexOf(color);
    const dx = target.x - source.x,
      dy = target.y - source.y,
      distance = Math.hypot(dx, dy) || 1;
    if (row.source === row.target) {
      markup += `<path d="M${source.x + 15},${source.y - 15} c60,-55 60,60 5,25" class="network-edge" stroke="${color}" marker-end="url(#arrow-${markerIndex})"/>`;
    } else {
      const x1 = source.x + (dx / distance) * (source.radius + 4),
        y1 = source.y + (dy / distance) * (source.radius + 4);
      const x2 = target.x - (dx / distance) * (target.radius + 7),
        y2 = target.y - (dy / distance) * (target.radius + 7);
      markup += `<path d="M${x1},${y1} L${x2},${y2}" class="network-edge" stroke="${color}" marker-end="url(#arrow-${markerIndex})"/><text x="${(x1 + x2) / 2 + (Math.abs(dx) < 20 ? 10 : 0)}" y="${(y1 + y2) / 2 - 7}" text-anchor="${Math.abs(dx) < 20 ? "start" : "middle"}" class="edge-label">${escape(row.type)}</text>`;
    }
  });
  for (const [name, point] of positions) {
    const selected = state.selected === name;
    markup += `<g class="node" tabindex="0" role="button" aria-label="Inspect entity ${escape(name)}" data-entity="${escape(name)}">${selected ? `<circle cx="${point.x}" cy="${point.y}" r="${point.radius + 9}" fill="none" stroke="${point.color}" stroke-opacity=".3"/>` : ""}<circle cx="${point.x}" cy="${point.y}" r="${point.radius}" fill="${point.color}" fill-opacity="${point.radius > 20 ? ".95" : ".16"}" stroke="${point.color}" stroke-width="1.5"/>${point.radius > 20 ? `<circle cx="${point.x}" cy="${point.y}" r="4" fill="white"/>` : ""}<text x="${point.x}" y="${point.y + point.radius + 20}" text-anchor="middle">${wrapName(
      name,
    )
      .map(
        (line, index) =>
          `<tspan x="${point.x}" dy="${index ? 16 : 0}">${escape(line)}</tspan>`,
      )
      .join("")}</text><title>${escape(name)}</title></g>`;
  }
  if (!rows.length)
    markup +=
      '<text x="500" y="230" text-anchor="middle" fill="#8a9982" font-size="16">No relationships to display</text>';
  graph.innerHTML = markup;
  graph.querySelectorAll("[data-entity]").forEach((node) => {
    const select = () => {
      state.selected = node.dataset.entity;
      renderGraph();
      renderInspector();
    };
    node.addEventListener("click", select);
    node.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        select();
      }
    });
  });
}

function openArticle(index) {
  state.article = index;
  renderArticles();
  setView("articles");
}
function attachArticleLinks(container) {
  container
    .querySelectorAll("[data-article]")
    .forEach((button) =>
      button.addEventListener("click", () =>
        openArticle(Number(button.dataset.article)),
      ),
    );
}

function renderInspector() {
  const name = state.selected,
    inspector = $("#inspector");
  if (!name) {
    inspector.innerHTML =
      '<div class="empty">Select a node to explore its connections.</div>';
    return;
  }
  const rows = visibleRelations().filter(
    (row) => row.source === name || row.target === name,
  );
  const articleIds = [...new Set(rows.map((row) => row.articleIndex))];
  inspector.innerHTML = `<div class="eyebrow">ENTITY INSPECTOR</div><div class="entity-symbol">${escape(name.charAt(0).toUpperCase())}</div><h3>${escape(format(name))}</h3><div class="entity-tag">Entity · ${rows.length} connections</div><div class="detail-label">CONNECTIONS</div>${rows.map((row) => `<div class="relation-item"><small>${row.source === name ? "↗ " : "↙ "}${escape(row.type)}</small><br>${escape(format(row.source === name ? row.target : row.source))}</div>`).join("")}<div class="detail-label">SOURCE ARTICLES</div>${articleIds.map((index) => `<button class="source-button" data-article="${index}">${escape(format(state.data.articles[index].Entity))}<span>↗</span></button>`).join("")}<div class="inspector-foot">Follow a source to read the article<br>behind these connections.</div>`;
  attachArticleLinks(inspector);
}

function renderArticles() {
  const articles = state.data.articles;
  $("#article-list").innerHTML = articles
    .map(
      (article, index) =>
        `<button class="article-select ${index === state.article ? "active" : ""}" data-article="${index}" aria-pressed="${index === state.article}"><small>${String(index + 1).padStart(2, "0")} / ${escape(article.Source || "ARTICLE")}</small><strong>${escape(format(article.Entity))}</strong><span>${article.Relationships.length} relationships · ${escape(article.Source === "wiki" ? "Wikipedia" : article.Source || "Source")}</span></button>`,
    )
    .join("");
  attachArticleLinks($("#article-list"));
  const article = articles[state.article];
  if (!article) {
    $("#article-detail").innerHTML =
      '<div class="empty">No articles in this dataset.</div>';
    return;
  }
  const url = safeSourceUrl(article.URL);
  $("#article-detail").innerHTML =
    `<div class="article-title-row"><div><div class="eyebrow">SOURCE ARTICLE</div><h2>${escape(format(article.Entity))}</h2><p class="meta">${escape(article.Source === "wiki" ? "Wikipedia" : article.Source || "Source not recorded")} <span class="slash">/</span> ${article.Relationships.length} relationships${article.ExampleNote ? " / Archived excerpt" : ""}</p></div>${url ? `<a class="source-link" href="${escape(url)}" target="_blank" rel="noopener noreferrer">Open original page ↗</a>` : ""}</div><div class="summary-block"><div class="detail-label">${article.ExampleNote ? "EXAMPLE SUMMARY / MANUALLY PREPARED" : "EXTRACTED SUMMARY"}</div><p>${escape(article.Summary || "No summary available.")}</p></div><div class="detail-label">${article.ExampleNote ? "ARCHIVED EXCERPT" : "ARTICLE"} / ORIGINAL TEXT</div><p class="article-content">${escape(article.Content || article.Article || "No article text available.")}</p><div class="detail-label">${article.ExampleNote ? "EXAMPLE" : "EXTRACTED"} / RELATIONSHIPS</div>${article.Relationships.map((triple) => `<div class="triple"><code>${escape(format(triple[0]))}</code><span class="relation-pill">${escape(triple[1])} →</span><code>${escape(format(triple[2]))}</code></div>`).join("") || '<p class="subtle">No relationships available.</p>'}`;
}

function renderRelations() {
  if (!state.data) return;
  const rows = filterRelations(state.data.relations, $("#search").value);
  $("#relation-rows").innerHTML =
    rows
      .map(
        (row) =>
          `<tr><td>${escape(format(row.source))}</td><td><span class="relation-pill">${escape(row.type)} →</span></td><td>${escape(format(row.target))}</td><td>${row.cluster === null ? "Unassigned" : `Cluster ${escape(row.cluster)}`}</td><td><button data-article="${row.articleIndex}">${escape(format(state.data.articles[row.articleIndex].Entity))} ↗</button></td></tr>`,
      )
      .join("") ||
    '<tr><td colspan="5" class="empty">No matching relationships. Try another search.</td></tr>';
  $("#table-count").textContent =
    `Showing ${rows.length} of ${state.data.relations.length} relationships`;
  attachArticleLinks($("#relation-rows"));
}

document
  .querySelectorAll("[data-view]")
  .forEach((button) =>
    button.addEventListener("click", () => setView(button.dataset.view)),
  );
$("[data-open-articles]").addEventListener("click", () => setView("articles"));
$("#dataset").addEventListener("change", (event) =>
  loadDataset(event.target.value),
);
$("#search").addEventListener("input", renderRelations);
$("#reset").addEventListener("click", () => {
  if (!state.data) return;
  state.cluster = "all";
  state.selected = state.data.entities[0] || null;
  renderFilters();
  renderGraph();
  renderInspector();
});
loadDataset("example");
