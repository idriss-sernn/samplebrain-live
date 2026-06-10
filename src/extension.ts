import * as http from "node:http";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import * as crypto from "node:crypto";
import { execFile } from "node:child_process";
import * as net from "node:net";
import * as https from "node:https";
import * as os from "node:os";
import { URL } from "node:url";
import {
  initialize,
  AudioTrack,
  DataModelObject,
  type ActivationContext,
  type ArrangementSelection,
} from "@ableton-extensions/sdk";
import dialogHtml from "./dialog.html";
import { buildBrain, buildTarget, synthesize, type BrainParams, type MatchParams } from "./brain.js";
import { decodeWav, encodeWav, resampleLinear } from "./wav.js";
import { generateSiblings, detectOnsetFractions } from "./siblings.js";

// ⚠️ Keep in sync with manifest.json / package.json on every release.
const APP_VERSION = "1.2.0";
const REPO = "idriss-sernn/samplebrain-live";

// Largest block size (samples) that still yields ≥1 grain from both the target and at
// least one brain source. The requested size is clamped to this instead of erroring out.
function maxValidBlockSize(targetLen: number, brainPcms: Float64Array[]): number {
  const longestBrain = brainPcms.reduce((m, p) => Math.max(m, p.length), 0);
  return Math.max(1, Math.min(targetLen, longestBrain));
}

// Reads an audio file, working around a macOS firmlink mismatch: Node's permission model
// (only enforced in the installed .ablx, not in `extensions-cli run`) can deny a
// "/var/folders/…" path when the grant is on the canonical "/private/var/folders/…" (or
// vice-versa). On a restriction error we retry with the /private prefix toggled.
async function decodeAudioRobust(p: string) {
  try {
    return await decodeWav(p);
  } catch (e) {
    const msg = String((e as { message?: string })?.message ?? e);
    if (!/restricted|ERR_ACCESS_DENIED|allow-fs/i.test(msg)) throw e;
    const alt = p.startsWith("/private/") ? p.slice(8) : "/private" + p;
    console.warn(`SampleBrain: fs read restricted on "${p}" — retrying "${alt}"`);
    return await decodeWav(alt);
  }
}

// Returns true when `latest` is a strictly newer semver than `current` (tolerates a "v" prefix).
function isNewerVersion(latest: string, current: string): boolean {
  const norm = (s: string) => s.replace(/^v/i, "").split(".").map((n) => parseInt(n, 10) || 0);
  const a = norm(latest), b = norm(current);
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    const x = a[i] ?? 0, y = b[i] ?? 0;
    if (x !== y) return x > y;
  }
  return false;
}

// Fetch the latest GitHub release tag. Resolves null on any error/timeout (offline-safe).
function fetchLatestRelease(): Promise<{ tag: string; url: string } | null> {
  return new Promise((resolve) => {
    const req = https.get(
      {
        hostname: "api.github.com",
        path: `/repos/${REPO}/releases/latest`,
        headers: { "User-Agent": "SampleBrain-Extension", Accept: "application/vnd.github+json" },
      },
      (res) => {
        if (res.statusCode !== 200) { res.resume(); resolve(null); return; }
        const chunks: Buffer[] = [];
        res.on("data", (c: Buffer) => chunks.push(c));
        res.on("end", () => {
          try {
            const j = JSON.parse(Buffer.concat(chunks).toString());
            if (typeof j.tag_name === "string") {
              const url = typeof j.html_url === "string" ? j.html_url : `https://github.com/${REPO}/releases`;
              resolve({ tag: j.tag_name, url });
            } else resolve(null);
          } catch { resolve(null); }
        });
      },
    );
    req.on("error", () => resolve(null));
    req.setTimeout(4000, () => { req.destroy(); resolve(null); });
  });
}

export function activate(activation: ActivationContext) {
  const ctx = initialize(activation, "1.0.0");

  ctx.commands.registerCommand("samplebrain.process", (arg: unknown) =>
    void run(ctx, arg as ArrangementSelection).catch(console.error),
  );

  ctx.ui.registerContextMenuAction(
    "AudioTrack.ArrangementSelection",
    "SampleBrain…",
    "samplebrain.process",
  );
}

type Ctx = ReturnType<typeof initialize>;

async function run(ctx: Ctx, selection: ArrangementSelection): Promise<void> {
  const song = ctx.application.song;
  if (!song) return;

  const selectedTracks = selection.selected_lanes
    .map((h) => ctx.getObjectFromHandle(h, DataModelObject))
    .filter((o): o is AudioTrack<"1.0.0"> => o instanceof AudioTrack);

  if (selectedTracks.length === 0) return;
  const targetTrack = selectedTracks[0]!;

  // Capture selection bounds immediately — the ArrangementSelection proxy becomes
  // stale once showModalDialog suspends the command handler, returning 0 for both values.
  const selStart = selection.time_selection_start;
  const selEnd = selection.time_selection_end;

  if (selEnd - selStart < 0.01) {
    console.error("SampleBrain: selection is empty — drag a time range in the Arrangement View before right-clicking");
    return;
  }

  // All AudioTracks in the project — candidates for both target and brain sources
  const allAudio = song.tracks.filter((t): t is AudioTrack<"1.0.0"> => t instanceof AudioTrack);

  // Default target = the right-clicked track; the dialog can switch to any other track
  const defaultTargetIndex = Math.max(0, allAudio.findIndex(t => t === targetTrack || t.name === targetTrack.name));

  // Random token — required on all API requests so other local processes can't call the endpoints
  const sessionToken = crypto.randomBytes(16).toString("hex");

  const initData = {
    tracks: allAudio.map((t, i) => ({ index: i, name: t.name })),
    defaultTargetIndex,
    sessionToken,
  };

  let pendingPreviewId = 0;

  // PCM cache — renderPreFxAudio is called only once per source/target change, not on every slider move
  let cachedTargetPcm: { samples: Float64Array; sampleRate: number } | null = null;
  let cachedTargetIndex = -1;
  let cachedBrainKey = "";
  let cachedBrainPcms: Float64Array[] = [];

  // Sample Design source = the concatenative-synthesis output. runPreview caches the exact
  // PCM it last produced here, so "Generate samples" works on the same audio the user saw/selected.
  let lastSdOutput: { pcm: Float64Array; sampleRate: number } | null = null;

  // Progress state — polled by /preview-progress while a render is in flight
  const renderProgress = { done: 0, total: 0 };

  // Serve the dialog HTML with injected data over a local HTTP server
  const server = http.createServer((req, res) => {
    const reqUrl = new URL(req.url ?? "/", `http://127.0.0.1`);

    // Auth: everything except the initial page load requires the session token
    const isPageLoad = req.method === "GET" && reqUrl.pathname === "/";
    if (!isPageLoad && (req.headers["x-sb-token"] as string) !== sessionToken) {
      res.writeHead(403).end();
      return;
    }

    if (req.method === "GET" && reqUrl.pathname === "/") {
      // Escape JSON before embedding in <script>: neutralises </script> breakout via track names (XSS #1)
      const safeInit = JSON.stringify(initData)
        .replace(/</g, "\\u003c").replace(/>/g, "\\u003e").replace(/&/g, "\\u0026");
      const page = dialogHtml.replace(
        "</head>",
        `<script>window.__SB_INIT__ = ${safeInit};window.__SB_TOKEN__ = ${JSON.stringify(sessionToken)};</script></head>`,
      );
      res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      res.end(page);
    } else if (req.method === "GET" && reqUrl.pathname === "/open") {
      const url = reqUrl.searchParams.get("url") ?? "";
      if (url.startsWith("https://")) execFile("/usr/bin/open", [url], (err) => { if (err) console.error("SampleBrain /open failed:", err); });
      res.writeHead(204).end();
    } else if (req.method === "GET" && reqUrl.pathname === "/check-update") {
      fetchLatestRelease().then((rel) => {
        const latest = rel?.tag ?? null;
        const updateAvailable = !!(latest && isNewerVersion(latest, APP_VERSION));
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ current: APP_VERSION, latest, updateAvailable, url: rel?.url ?? `https://github.com/${REPO}/releases` }));
      }).catch(() => {
        res.writeHead(200, { "Content-Type": "application/json" }).end(JSON.stringify({ current: APP_VERSION, latest: null, updateAvailable: false }));
      });
    } else if (req.method === "POST" && reqUrl.pathname === "/upload-file") {
      const chunks: Buffer[] = [];
      req.on("data", (c: Buffer) => chunks.push(c));
      req.on("end", async () => {
        try {
          const buf = Buffer.concat(chunks);
          const filename = (req.headers["x-filename"] as string) ?? "sample.wav";
          // Whitelist extension — never trust X-Filename (null-byte / arbitrary ext injection)
          const rawExt = path.extname(filename).toLowerCase();
          const ext = [".wav", ".aif", ".aiff"].includes(rawExt) ? rawExt : ".wav";
          // Installed .ablx: tempDirectory is the only writable dir (permission model).
          // Dev (extensions-cli run): it can be undefined → fall back to the OS temp.
          const tmpDir = ctx.environment.tempDirectory ?? os.tmpdir();
          const outPath = path.join(tmpDir, `sb_drop_${Date.now()}${ext}`);
          await fs.writeFile(outPath, buf);
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ ok: true, path: outPath }));
        } catch {
          res.writeHead(500).end();
        }
      });
    } else if (req.method === "GET" && reqUrl.pathname === "/preview-progress") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify(renderProgress));
    } else if (req.method === "GET" && reqUrl.pathname === "/sd-onsets") {
      // Onset markers for the Sample Design waveform (computed on the last shown output)
      const onsets = lastSdOutput ? detectOnsetFractions(lastSdOutput.pcm, lastSdOutput.sampleRate) : [];
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ onsets }));
    } else if (req.method === "POST" && reqUrl.pathname === "/preview") {
      const chunks: Buffer[] = [];
      req.on("data", (c: Buffer) => chunks.push(c));
      req.on("end", () => void runPreview(
        ++pendingPreviewId,
        () => pendingPreviewId,
        JSON.parse(Buffer.concat(chunks).toString()) as PreviewRequest,
        res,
      ).catch((err) => { if (!res.writableEnded) res.writeHead(500).end(String(err)); }));
    } else {
      res.writeHead(404).end();
    }
  });

  async function runPreview(
    myId: number,
    currentId: () => number,
    p: PreviewRequest,
    res: http.ServerResponse,
  ): Promise<void> {
    const stale = () => myId !== currentId();
    if (stale()) { res.writeHead(409).end(); return; }

    // selStart/selEnd captured in outer run() scope — always valid
    if (selEnd - selStart < 0.01) { console.error("SampleBrain preview: selection too short", selStart, selEnd); res.writeHead(400, { "Content-Type": "text/plain" }).end("No target selected — highlight a time range on an audio track first"); return; }

    // Target PCM — re-rendered when the chosen target track changes
    const targetIdx = p.targetTrackIndex;
    const targetTrk = allAudio[targetIdx];
    if (!targetTrk) { res.writeHead(400, { "Content-Type": "text/plain" }).end("Invalid target track"); return; }
    // Never use the target as its own brain source (same-track concatenation is pointless)
    const brainIndices = p.brainTrackIndices.filter(i => i !== targetIdx);
    const brainKey = brainIndices.join(",") + "|" + p.droppedFiles.join(",");
    const needsTarget = !cachedTargetPcm || cachedTargetIndex !== targetIdx;
    const needsBrain = brainKey !== cachedBrainKey;

    if (needsTarget || needsBrain) {
      // Count items that will actually be rendered to show accurate progress
      const tracksWithClips = needsBrain
        ? brainIndices.map(i => allAudio[i]).filter(t => t && t.arrangementClips.length > 0)
        : [];
      renderProgress.done = 0;
      renderProgress.total = (needsTarget ? 1 : 0) + tracksWithClips.length + (needsBrain ? p.droppedFiles.length : 0);
    } else {
      renderProgress.done = 0; renderProgress.total = 0;
    }

    if (needsTarget) {
      const targetWavPath = await ctx.resources.renderPreFxAudio(targetTrk, selStart, selEnd);
      console.log(`SampleBrain: render path="${targetWavPath}" tempDir="${ctx.environment.tempDirectory}"`);
      if (stale()) { res.writeHead(409).end(); return; }
      const decoded = await decodeAudioRobust(targetWavPath);
      if (stale()) { res.writeHead(409).end(); return; }
      cachedTargetPcm = decoded;
      cachedTargetIndex = targetIdx;
      renderProgress.done++;
    }
    const targetData = cachedTargetPcm!;

    // Brain PCMs — re-rendered only when the source selection changes, not on slider moves
    if (needsBrain) {
      const newPcms: Float64Array[] = [];
      for (const idx of brainIndices) {
        if (stale()) { res.writeHead(409).end(); return; }
        const track = allAudio[idx];
        if (!track || track.arrangementClips.length === 0) continue;
        const brainEnd = track.arrangementClips.reduce((m, c) => Math.max(m, c.endTime), 0);
        const brainWavPath = await ctx.resources.renderPreFxAudio(track, 0, brainEnd);
        if (stale()) { res.writeHead(409).end(); return; }
        const brainData = await decodeAudioRobust(brainWavPath);
        newPcms.push(brainData.samples);
        renderProgress.done++;
      }
      for (const filePath of p.droppedFiles) {
        if (stale()) { res.writeHead(409).end(); return; }
        if (![".wav", ".aif", ".aiff"].includes(path.extname(filePath).toLowerCase())) continue; // #7: only ever read audio paths
        try { newPcms.push((await decodeAudioRobust(filePath)).samples); renderProgress.done++; } catch { /* skip */ }
      }
      if (!stale()) { cachedBrainKey = brainKey; cachedBrainPcms = newPcms; }
    }
    const brainPcms = cachedBrainPcms;

    if (brainPcms.length === 0) { console.error("SampleBrain preview: all brain sources empty"); res.writeHead(400, { "Content-Type": "text/plain" }).end("No usable brain source — check a track with audio clips, or drop a WAV/AIFF file (MP3/FLAC not supported)"); return; }
    if (stale()) { res.writeHead(409).end(); return; }

    const { sampleRate } = targetData;
    // Clamp the requested block size to what the selection can actually produce (no error)
    const requestedBlock = Math.max(64, Math.round((p.blockSizeMs / 1000) * sampleRate));
    const blockSizeSamples = Math.min(requestedBlock, maxValidBlockSize(targetData.samples.length, brainPcms));
    const brainParams: BrainParams = {
      blockSizeSamples,
      overlapRatio: p.overlapRatio,
      windowType: p.windowType as BrainParams["windowType"],
      descriptorType: p.descriptorType as BrainParams["descriptorType"],
      mfccRatio: p.mfccRatio,
      sampleRate,
    };
    const brainBlocks = buildBrain(brainPcms, brainParams);
    const targetBlocks = buildTarget(targetData.samples, brainParams);
    if (brainBlocks.length === 0 || targetBlocks.length === 0) { console.error("SampleBrain preview: 0 blocks after clamp", brainBlocks.length, targetBlocks.length); res.writeHead(400, { "Content-Type": "text/plain" }).end("Selection too short to build any grain — select a longer range"); return; }
    if (stale()) { res.writeHead(409).end(); return; }

    const hopSize = Math.max(1, Math.round(blockSizeSamples * (1 - p.overlapRatio)));
    const matchParams: MatchParams = { novelty: p.novelty, boredom: p.boredom, stickiness: p.stickiness, voices: p.voices, pitchShift: p.pitchShift, pitchShiftVar: p.pitchShiftVar, reverse: p.reverse, density: p.density, matchMode: p.matchMode };
    const output = synthesize(targetBlocks, brainBlocks, hopSize, matchParams);
    const wavBuf = encodeWav(output, sampleRate, p.ampWeight);
    lastSdOutput = { pcm: output, sampleRate }; // Sample Design works on this exact output

    if (stale()) { res.writeHead(409).end(); return; }
    res.writeHead(200, { "Content-Type": "audio/wav", "Content-Length": String(wavBuf.length) });
    res.end(wavBuf);
  }

  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", () => resolve()));
  const { port } = server.address() as net.AddressInfo;

  let rawResult: string;
  try {
    rawResult = await ctx.ui.showModalDialog(`http://127.0.0.1:${port}/`, 576, 780);
  } finally {
    server.closeAllConnections?.();
    server.close();
  }

  let params: DialogResult;
  try {
    params = JSON.parse(rawResult) as DialogResult;
  } catch {
    console.error("SampleBrain: failed to parse dialog result:", rawResult);
    return;
  }
  if (params.cancelled) return;

  // ── Sample Design: generate pitched sibling samples/loops from the sources ──
  if (params.mode === "sample-design") {
    await ctx.ui.withinProgressDialog("Sample Design", {}, async (update, signal) => {
      try {
        const secPerBeat = 60 / song.tempo;

        // Source = the concatenative-synthesis output last shown in the Sample Design waveform
        if (!lastSdOutput) throw new Error("Open the Sample Design tab and let the waveform load first.");
        const concat = lastSdOutput.pcm;
        const sr = lastSdOutput.sampleRate;
        const baseName = allAudio[params.targetTrackIndex]?.name ?? "synth";
        let inputSources: { pcm: Float64Array; sampleRate: number; name: string }[] =
          [{ pcm: concat, sampleRate: sr, name: baseName }];
        // If the user selected a zone on the waveform, transform only that region
        const s0 = params.sibSelStart, s1 = params.sibSelEnd;
        if (s0 >= 0 && s1 > s0 && concat.length > 0) {
          const a = Math.max(0, Math.floor(s0 * concat.length));
          const b = Math.min(concat.length, Math.ceil(s1 * concat.length));
          if (b - a > 64) inputSources = [{ pcm: concat.slice(a, b), sampleRate: sr, name: "selection" }];
        }

        update("Generating siblings…", 45);
        const sibs = generateSiblings(inputSources, {
          minSt: params.sibMinSt,
          maxSt: params.sibMaxSt,
          strategy: params.sibStrategy,
          mode: params.sibMode,
          layers: params.sibLayers,
          autoAttenuation: params.sibAutoAtten,
          generations: params.sibGenerations,
        });
        if (sibs.length === 0) throw new Error("No siblings generated.");

        // Export folder: ~/Music/SampleBrain/<timestamp>/
        // Permission model: only storage/temp dirs are writable — not ~/Music. The imported
        // clip copies also land in the Live project folder (findable, correctly named).
        // Installed .ablx: storage/temp dirs (permission model). Dev: fall back to OS temp.
        const base = ctx.environment.storageDirectory ?? ctx.environment.tempDirectory ?? os.tmpdir();
        const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
        const dir = path.join(base, "SampleBrain", stamp);
        await fs.mkdir(dir, { recursive: true });
        console.log(`SampleBrain Sample Design: ${sibs.length} siblings → ${dir}`);

        const outTrack = await song.createAudioTrack();
        outTrack.name = "SB Siblings";
        let t = selStart;
        const sanitize = (s: string) => s.replace(/[/\\:*?"<>|]/g, "_").trim() || "sample";

        let written = 0, placed = 0;
        for (let i = 0; i < sibs.length; i++) {
          if (signal.aborted) return;
          const sib = sibs[i]!;
          update(`Writing sample ${i + 1}/${sibs.length}…`, 55 + i * (40 / sibs.length));
          const tag = sib.sourceName === "layered" ? "layered" : `${sanitize(sib.sourceName)} ${sib.label}`;
          const fname = `SB ${tag} ${i + 1}.wav`;
          const outPath = path.join(dir, fname);
          // Write the WAV first so the file exists on disk even if clip placement fails.
          // Use the proven float32 encoder (like Synth) — 24-bit PCM WAV breaks importIntoProject.
          try {
            const wav = encodeWav(resampleLinear(sib.pcm, sib.sampleRate, 44100), 44100, 1);
            await fs.writeFile(outPath, wav);
            written++;
          } catch (e) {
            console.error(`SampleBrain: failed to write "${fname}":`, e);
            continue;
          }
          // Import + place a clip — failure here is non-fatal (the WAV is already saved)
          try {
            const imported = await ctx.resources.importIntoProject(outPath);
            const durBeats = (sib.pcm.length / sib.sampleRate) / secPerBeat;
            await outTrack.createAudioClip({ filePath: imported, startTime: t, duration: durBeats, isWarped: false });
            t += durBeats;
            placed++;
          } catch (e) {
            console.error(`SampleBrain: failed to place clip for "${fname}":`, e);
          }
        }

        // Always reveal the export folder, even if some clips failed
        execFile("/usr/bin/open", [dir], (err) => { if (err) console.error("SampleBrain reveal folder failed:", err); });
        console.log(`SampleBrain Sample Design: wrote ${written}/${sibs.length} WAVs, placed ${placed} clips → ${dir}`);
        update(`Done — ${written} samples in ${dir}`, 100);
      } catch (err) {
        if (signal.aborted) return;
        console.error("SampleBrain Sample Design error:", err);
        throw err;
      }
    });
    return;
  }

  // Target chosen in the dialog (may differ from the right-clicked track)
  const chosenTarget = allAudio[params.targetTrackIndex] ?? targetTrack;

  const brainTracks = params.brainTrackIndices
    .filter((i) => i !== params.targetTrackIndex) // never concatenate the target with itself
    .map((i) => allAudio[i])
    .filter((t): t is AudioTrack<"1.0.0"> => t !== undefined);

  if (brainTracks.length === 0 && params.droppedFiles.length === 0) {
    console.error("SampleBrain: no brain sources selected");
    return;
  }

  await ctx.ui.withinProgressDialog("SampleBrain", {}, async (update, signal) => {
    try {
      const tempo = song.tempo;
      const secPerBeat = 60 / tempo;
      // selStart/selEnd captured in outer run() scope (before showModalDialog) — proxy is stale here

      // --- Render target ---
      update(`Rendering target… (${selStart.toFixed(2)}–${selEnd.toFixed(2)} beats)`, 5);
      const targetWavPath = await ctx.resources.renderPreFxAudio(chosenTarget, selStart, selEnd);
      if (signal.aborted) return;
      const targetData = await decodeAudioRobust(targetWavPath);
      if (signal.aborted) return;

      // Guard: target must have audio content
      const targetPeak = targetData.samples.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
      if (targetPeak < 1e-4) {
        throw new Error(`Target track "${chosenTarget.name}" has no audio in the selected range — select a range that contains audio clips.`);
      }

      // --- Render each brain track ---
      const brainPcms: Float64Array[] = [];
      for (let i = 0; i < brainTracks.length; i++) {
        if (signal.aborted) return;
        const track = brainTracks[i]!;
        const progressBase = 10 + i * (30 / brainTracks.length);
        update(`Rendering brain: ${track.name} (${i + 1}/${brainTracks.length})…`, progressBase);

        const clips = track.arrangementClips;
        if (clips.length === 0) {
          console.warn(`SampleBrain: brain track "${track.name}" has no clips — skipping`);
          continue;
        }
        const brainEnd = clips.reduce((m, c) => Math.max(m, c.endTime), 0);
        const brainWavPath = await ctx.resources.renderPreFxAudio(track, 0, brainEnd);
        if (signal.aborted) return;
        const brainData = await decodeAudioRobust(brainWavPath);
        brainPcms.push(brainData.samples);
      }

      // --- Decode dropped files ---
      for (let i = 0; i < params.droppedFiles.length; i++) {
        if (signal.aborted) return;
        const filePath = params.droppedFiles[i]!;
        if (![".wav", ".aif", ".aiff"].includes(path.extname(filePath).toLowerCase())) { console.warn(`SampleBrain: ignoring non-audio path "${filePath}"`); continue; } // #7
        update(`Loading file ${i + 1}/${params.droppedFiles.length}…`, 40 + i * (8 / Math.max(1, params.droppedFiles.length)));
        try {
          const brainData = await decodeAudioRobust(filePath);
          brainPcms.push(brainData.samples);
        } catch (err) {
          console.warn(`SampleBrain: skipping dropped file "${filePath}": ${err}`);
        }
      }

      if (brainPcms.length === 0) {
        throw new Error("No usable brain source — check a track with audio clips, or drop a WAV/AIFF file (MP3/FLAC not supported).");
      }

      const { sampleRate } = targetData;
      // Clamp the requested block size to what the selection can actually produce (no error)
      const requestedBlock = Math.max(64, Math.round((params.blockSizeMs / 1000) * sampleRate));
      const blockSizeSamples = Math.min(requestedBlock, maxValidBlockSize(targetData.samples.length, brainPcms));
      console.log(`SampleBrain: sampleRate=${sampleRate}, blockSize=${blockSizeSamples} samples (requested ${requestedBlock}), target=${targetData.numSamples} samples`);

      const brainParams: BrainParams = {
        blockSizeSamples,
        overlapRatio: params.overlapRatio,
        windowType: params.windowType as BrainParams["windowType"],
        descriptorType: params.descriptorType as BrainParams["descriptorType"],
        mfccRatio: params.mfccRatio,
        sampleRate,
      };

      // --- Build brain blocks ---
      update("Building brain index…", 50);
      if (signal.aborted) return;
      const brainBlocks = buildBrain(brainPcms, brainParams);
      console.log(`SampleBrain: ${brainBlocks.length} brain blocks built`);

      if (brainBlocks.length === 0) {
        throw new Error("Brain produced 0 grains — lower the block size.");
      }

      // --- Analyse target ---
      update("Analysing target…", 60);
      if (signal.aborted) return;
      const targetBlocks = buildTarget(targetData.samples, brainParams);
      console.log(`SampleBrain: ${targetBlocks.length} target blocks`);

      if (targetBlocks.length === 0) {
        throw new Error("Target produced 0 grains — selection too short.");
      }

      // --- Synthesise ---
      update(`Synthesising (${targetBlocks.length} target × ${brainBlocks.length} brain cells)…`, 68);
      if (signal.aborted) return;
      const hopSize = Math.max(1, Math.round(blockSizeSamples * (1 - params.overlapRatio)));
      const matchParams: MatchParams = {
        novelty: params.novelty,
        boredom: params.boredom,
        stickiness: params.stickiness,
        voices: params.voices,
        pitchShift: params.pitchShift,
        pitchShiftVar: params.pitchShiftVar,
        reverse: params.reverse,
        density: params.density,
        matchMode: params.matchMode,
      };
      const output = synthesize(targetBlocks, brainBlocks, hopSize, matchParams);
      console.log(`SampleBrain: output ${output.length} samples`);

      // --- Write WAV ---
      update("Writing output…", 85);
      if (signal.aborted) return;
      const wavBuf = encodeWav(output, sampleRate, params.ampWeight);
      const tmpDir = ctx.environment.tempDirectory ?? os.tmpdir(); // dev fallback; installed = host temp
      const outPath = path.join(tmpDir, `samplebrain_${Date.now()}.wav`);
      await fs.writeFile(outPath, wavBuf);
      console.log(`SampleBrain: wrote ${wavBuf.length} bytes → ${outPath}`);

      // --- Import + place clip on a new track ---
      update("Importing into project…", 93);
      const importedPath = await ctx.resources.importIntoProject(outPath);
      console.log(`SampleBrain: imported → ${importedPath}`);
      if (signal.aborted) return;

      const outputDurSec = output.length / sampleRate;
      const outputDurBeats = outputDurSec / secPerBeat;

      const outputTrack = await song.createAudioTrack();
      outputTrack.name = `SB: ${chosenTarget.name}`;
      await outputTrack.createAudioClip({
        filePath: importedPath,
        startTime: selStart,
        duration: outputDurBeats,
        isWarped: false,
      });

      update("Done.", 100);
      console.log("SampleBrain: clip placed successfully");
    } catch (err) {
      if (signal.aborted) return;
      console.error("SampleBrain error:", err);
      throw err;
    }
  });
}

interface DialogResult {
  cancelled: boolean;
  mode: "synth" | "sample-design";
  // Sample Design params (Audio Siblings — only meaningful when mode === "sample-design")
  sibMinSt: number;
  sibMaxSt: number;
  sibStrategy: "chromatic" | "major" | "minor" | "pentatonic";
  sibMode: "pitch" | "layered" | "slice" | "automate";
  sibLayers: number;
  sibAutoAtten: boolean;
  sibGenerations: number;
  sibSelStart: number; // waveform selection start (fraction 0–1), -1 = none
  sibSelEnd: number;   // waveform selection end (fraction 0–1)
  targetTrackIndex: number;
  brainTrackIndices: number[];
  droppedFiles: string[];
  matchMode: "greedy" | "smooth";
  blockSizeMs: number;
  overlapRatio: number;
  windowType: string;
  descriptorType: string;
  mfccRatio: number;
  novelty: number;
  boredom: number;
  stickiness: number;
  ampWeight: number;
  voices: number;
  pitchShift: number;
  pitchShiftVar: number;
  reverse: number;
  density: number;
}

interface PreviewRequest {
  targetTrackIndex: number;
  brainTrackIndices: number[];
  droppedFiles: string[];
  matchMode: "greedy" | "smooth";
  blockSizeMs: number;
  overlapRatio: number;
  windowType: string;
  descriptorType: string;
  mfccRatio: number;
  novelty: number;
  boredom: number;
  stickiness: number;
  ampWeight: number;
  voices: number;
  pitchShift: number;
  pitchShiftVar: number;
  reverse: number;
  density: number;
}
