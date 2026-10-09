// Read the existing pipeline exports; no new extraction or clustering logic.
export function parseCsv(text) {
  const rows = [];
  let row = [],
    field = "",
    quoted = false;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (char === '"') {
      if (quoted && text[i + 1] === '"') {
        field += '"';
        i++;
      } else quoted = !quoted;
    } else if (char === "," && !quoted) {
      row.push(field);
      field = "";
    } else if (char === "\n" && !quoted) {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      field = "";
    } else field += char;
  }
  if (field || row.length) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  const header = rows.shift() || [];
  return rows
    .filter((row) => row.some(Boolean))
    .map((row) =>
      Object.fromEntries(header.map((key, i) => [key, row[i] || ""])),
    );
}

export function buildDataset(jsonl, clusterCsv = "") {
  const articles = jsonl
    .split(/\r?\n/)
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line));
  if (
    !articles.every(
      (article) =>
        typeof article.Entity === "string" &&
        Array.isArray(article.Relationships),
    )
  ) {
    throw new Error(
      "Use all_output.jsonl from the refactored pipeline, with Entity and Relationships fields.",
    );
  }
  const clusterMap = new Map(
    parseCsv(clusterCsv).map((row) => [row.relations, row.cluster_id]),
  );
  const relations = articles.flatMap((article, articleIndex) =>
    article.Relationships.map((triple) => {
      if (
        !Array.isArray(triple) ||
        triple.length !== 3 ||
        !triple.every((value) => typeof value === "string" && value.trim())
      ) {
        throw new Error("Relationships contains an incomplete triple.");
      }
      return {
        source: triple[0],
        type: triple[1],
        target: triple[2],
        articleIndex,
        cluster: clusterMap.get(triple.join(" ")) ?? null,
      };
    }),
  );
  return {
    articles,
    relations,
    entities: [
      ...new Set(relations.flatMap((row) => [row.source, row.target])),
    ],
    clusters: [
      ...new Set(
        relations.map((row) => row.cluster).filter((value) => value !== null),
      ),
    ].sort((a, b) => Number(a) - Number(b)),
  };
}

export function filterRelations(relations, query) {
  const needle = query.trim().toLowerCase();
  return relations.filter((row) =>
    [row.source, row.type, row.target].some(
      (value) =>
        value.toLowerCase().includes(needle) ||
        value.replaceAll("_", " ").toLowerCase().includes(needle),
    ),
  );
}

export function connectedGroups(relations) {
  const neighbors = new Map();
  for (const row of relations) {
    if (!neighbors.has(row.source)) neighbors.set(row.source, new Set());
    if (!neighbors.has(row.target)) neighbors.set(row.target, new Set());
    neighbors.get(row.source).add(row.target);
    neighbors.get(row.target).add(row.source);
  }
  const seen = new Set(),
    groups = [];
  for (const name of neighbors.keys()) {
    if (seen.has(name)) continue;
    const group = [],
      queue = [name];
    seen.add(name);
    while (queue.length) {
      const current = queue.shift();
      group.push(current);
      for (const next of neighbors.get(current)) {
        if (!seen.has(next)) {
          seen.add(next);
          queue.push(next);
        }
      }
    }
    group.sort((a, b) => neighbors.get(b).size - neighbors.get(a).size);
    groups.push(group);
  }
  return groups;
}

export function safeSourceUrl(value) {
  try {
    const url = new URL(value);
    return ["https:", "http:"].includes(url.protocol) ? url.href : null;
  } catch {
    return null;
  }
}
