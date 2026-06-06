import { readFile } from "node:fs/promises";
import { test } from "node:test";
import assert from "node:assert/strict";

test("extension does not spawn platform-specific system URL openers", async () => {
  const source = await readFile(new URL("./extension.ts", import.meta.url), "utf8");

  assert.doesNotMatch(source, /node:child_process/);
  assert.doesNotMatch(source, /execFile\s*\(/);
});

test("extension never falls back to Unix-only temp paths", async () => {
  const source = await readFile(new URL("./runtime.ts", import.meta.url), "utf8");

  assert.doesNotMatch(source, /\/tmp/);
  assert.match(source, /os\.tmpdir\(\)/);
});

test("context menu titles rely on Live's automatic extension prefix", async () => {
  const source = await readFile(new URL("./extension.ts", import.meta.url), "utf8");

  assert.match(source, /"Process Selection…"/);
  assert.match(source, /"Process Clip…"/);
  assert.doesNotMatch(source, /"SampleBrain: Process/);
});

test("extension is visible from ordinary audio clip context menus", async () => {
  const source = await readFile(new URL("./extension.ts", import.meta.url), "utf8");

  assert.match(source, /registerContextMenuAction\(\s*"AudioClip"/);
  assert.match(source, /"Process Clip…"/);
});

test("production entry stays as a lightweight activation shim", async () => {
  const source = await readFile(new URL("./extension.ts", import.meta.url), "utf8");

  assert.doesNotMatch(source, /node:http|node:fs\/promises|node:net|\.\/dialog\.html|\.\/brain\.js|\.\/wav\.js/);
  assert.match(source, /import\("\.\/runtime\.js"\)/);
});

test("package exposes a separate typecheck gate", async () => {
  const packageJson = JSON.parse(
    await readFile(new URL("../package.json", import.meta.url), "utf8"),
  ) as { scripts?: Record<string, string> };

  assert.equal(packageJson.scripts?.typecheck, "tsc --noEmit");
});

test("manifest declares production permissions for filesystem and loopback UI", async () => {
  const manifest = JSON.parse(
    await readFile(new URL("../manifest.json", import.meta.url), "utf8"),
  ) as { permissions?: string[] };

  assert.deepEqual(manifest.permissions?.sort(), ["filesystem", "network"]);
});
