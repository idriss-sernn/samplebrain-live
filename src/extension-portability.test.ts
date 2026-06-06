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

test("installed runtime does not require a loopback HTTP server", async () => {
  const runtime = await readFile(new URL("./runtime.ts", import.meta.url), "utf8");
  const dialog = await readFile(new URL("./dialog.html", import.meta.url), "utf8");

  assert.doesNotMatch(runtime, /node:http|createServer|127\.0\.0\.1|ServerResponse/);
  assert.doesNotMatch(dialog, /\/preview|\/upload-file/);
});

test("context menu titles are explicit for installed Live Beta menus", async () => {
  const source = await readFile(new URL("./extension.ts", import.meta.url), "utf8");

  assert.match(source, /"SampleBrain: Process Selection\.\.\."/);
  assert.match(source, /"SampleBrain: Process Clip\.\.\."/);
});

test("extension is visible from ordinary audio clip context menus", async () => {
  const source = await readFile(new URL("./extension.ts", import.meta.url), "utf8");

  assert.match(source, /registerContextMenuAction\(\s*"AudioClip"/);
  assert.match(source, /"SampleBrain: Process Clip\.\.\."/);
});

test("production entry keeps heavy work behind the command handler", async () => {
  const source = await readFile(new URL("./extension.ts", import.meta.url), "utf8");

  assert.doesNotMatch(source, /node:http|node:fs\/promises|node:net|\.\/dialog\.html|\.\/brain\.js|\.\/wav\.js/);
  assert.match(source, /void run\(ctx, arg\)\.catch\(console\.error\)/);
});

test("packaging uses a single production entry file", async () => {
  const packageJson = JSON.parse(
    await readFile(new URL("../package.json", import.meta.url), "utf8"),
  ) as { scripts?: Record<string, string> };
  const buildScript = await readFile(new URL("../build.ts", import.meta.url), "utf8");

  assert.doesNotMatch(buildScript, /dist\/runtime\.cjs/);
  assert.doesNotMatch(packageJson.scripts?.package ?? "", /-i dist\/runtime\.cjs/);
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
