import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import {
  buildDataset,
  parseCsv,
  filterRelations,
  connectedGroups,
  safeSourceUrl,
} from "../web/model.mjs";

test("reads actual Python exports into the viewer", async () => {
  const jsonl = await readFile(
    new URL("../examples/expected/all_output.jsonl", import.meta.url),
    "utf8",
  );
  const csv = await readFile(
    new URL("../examples/expected/relation_clusters.csv", import.meta.url),
    "utf8",
  );
  const data = buildDataset(jsonl, csv);
  assert.equal(data.articles.length, 3);
  assert.equal(data.entities.length, 12);
  assert.equal(data.relations.length, 9);
  assert.equal(data.clusters.length, 3);
  assert.equal(connectedGroups(data.relations).length, 3);
  assert.equal(filterRelations(data.relations, "annual percentage").length, 3);
  assert.equal(filterRelations(data.relations, "DEPENDS_ON").length, 3);
  assert.equal(filterRelations(data.relations, "not-present").length, 0);
  assert.equal(buildDataset(jsonl).clusters.length, 0);
});

test("handles quoted CSV, commas, newlines and Windows line endings", () => {
  assert.deepEqual(
    parseCsv('cluster_id,relations\r\n0,"a, b said ""hello""\nnext"\r\n'),
    [{ cluster_id: "0", relations: 'a, b said "hello"\nnext' }],
  );
});

test("empty exports work; malformed data produces an actionable error", () => {
  assert.equal(buildDataset("").relations.length, 0);
  assert.throws(() => buildDataset('{"Entity":"x"}'), /all_output.jsonl/);
  assert.throws(
    () => buildDataset('{"Entity":"x","Relationships":[["x","y"]]}'),
    /triple/,
  );
});

test("source links only use web URLs", () => {
  assert.equal(
    safeSourceUrl("https://en.wikipedia.org/wiki/Currency"),
    "https://en.wikipedia.org/wiki/Currency",
  );
  assert.equal(safeSourceUrl("javascript:alert(1)"), null);
  assert.equal(safeSourceUrl("file:///etc/passwd"), null);
});
