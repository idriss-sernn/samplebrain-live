import * as http from "node:http";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import { execFile } from "node:child_process";
import * as net from "node:net";
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
import { decodeWav, encodeWav } from "./wav.js";

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

  // All AudioTracks in the project, as brain source candidates
  const allAudio = song.tracks.filter((t): t is AudioTrack<"1.0.0"> => t instanceof AudioTrack);

  // Exclude the target track from brain sources (same-track concatenation is pointless)
  const targetAudioIdx = allAudio.findIndex(t => t === targetTrack || t.name === targetTrack.name);

  const initData = {
    targetName: targetTrack.name,
    availableTracks: allAudio
      .map((t, i) => ({ index: i, name: t.name }))
      .filter(({ index }) => index !== targetAudioIdx),
  };

  let pendingPreviewId = 0;

  // Serve the dialog HTML with injected data over a local HTTP server
  const server = http.createServer((req, res) => {
    const reqUrl = new URL(req.url ?? "/", `http://127.0.0.1`);

    if (req.method === "GET" && reqUrl.pathname === "/") {
      const page = dialogHtml.replace(
        "</head>",
        `<script>window.__SB_INIT__ = ${JSON.stringify(initData)};</script></head>`,
      );
      res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      res.end(page);
    } else if (req.method === "GET" && reqUrl.pathname === "/open") {
      const url = reqUrl.searchParams.get("url") ?? "";
      if (url.startsWith("https://")) execFile("open", [url]);
      res.writeHead(204).end();
    } else if (req.method === "POST" && reqUrl.pathname === "/upload-file") {
      const chunks: Buffer[] = [];
      req.on("data", (c: Buffer) => chunks.push(c));
      req.on("end", async () => {
        try {
          const buf = Buffer.concat(chunks);
          const filename = (req.headers["x-filename"] as string) ?? "sample.wav";
          const ext = path.extname(filename) || ".wav";
          const tmpDir = ctx.environment.tempDirectory ?? "/tmp";
          const outPath = path.join(tmpDir, `sb_drop_${Date.now()}${ext}`);
          await fs.writeFile(outPath, buf);
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ ok: true, path: outPath }));
        } catch {
          res.writeHead(500).end();
        }
      });
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

    const selStart = selection.time_selection_start;
    const selEnd = selection.time_selection_end;
    if (selEnd - selStart < 0.01) { res.writeHead(400).end(); return; }

    const targetWavPath = await ctx.resources.renderPreFxAudio(targetTrack, selStart, selEnd);
    if (stale()) { res.writeHead(409).end(); return; }
    const targetData = await decodeWav(targetWavPath);
    if (stale()) { res.writeHead(409).end(); return; }

    const brainPcms: Float64Array[] = [];
    for (const idx of p.brainTrackIndices) {
      if (stale()) { res.writeHead(409).end(); return; }
      const track = allAudio[idx];
      if (!track || track.arrangementClips.length === 0) continue;
      const brainEnd = Math.max(...track.arrangementClips.map((c) => c.endTime));
      const brainWavPath = await ctx.resources.renderPreFxAudio(track, 0, brainEnd);
      if (stale()) { res.writeHead(409).end(); return; }
      const brainData = await decodeWav(brainWavPath);
      brainPcms.push(brainData.samples);
    }
    for (const filePath of p.droppedFiles) {
      if (stale()) { res.writeHead(409).end(); return; }
      try { brainPcms.push((await decodeWav(filePath)).samples); } catch { /* skip */ }
    }

    if (brainPcms.length === 0) { res.writeHead(400).end(); return; }
    if (stale()) { res.writeHead(409).end(); return; }

    const { sampleRate } = targetData;
    const blockSizeSamples = Math.max(64, Math.round((p.blockSizeMs / 1000) * sampleRate));
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
    if (brainBlocks.length === 0 || targetBlocks.length === 0) { res.writeHead(400).end(); return; }
    if (stale()) { res.writeHead(409).end(); return; }

    const hopSize = Math.max(1, Math.round(blockSizeSamples * (1 - p.overlapRatio)));
    const matchParams: MatchParams = { novelty: p.novelty, boredom: p.boredom, stickiness: p.stickiness, voices: p.voices };
    const output = synthesize(targetBlocks, brainBlocks, hopSize, matchParams);
    const wavBuf = encodeWav(output, sampleRate, p.ampWeight);

    if (stale()) { res.writeHead(409).end(); return; }
    res.writeHead(200, { "Content-Type": "audio/wav", "Content-Length": String(wavBuf.length) });
    res.end(wavBuf);
  }

  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", () => resolve()));
  const { port } = server.address() as net.AddressInfo;

  let rawResult: string;
  try {
    rawResult = await ctx.ui.showModalDialog(`http://127.0.0.1:${port}/`, 576, 768);
  } finally {
    server.closeAllConnections?.();
    server.close();
  }

  let params: DialogResult;
  try {
    params = JSON.parse(rawResult) as DialogResult;
  } catch {
    return;
  }
  if (params.cancelled) return;

  const brainTracks = params.brainTrackIndices
    .map((i) => allAudio[i])
    .filter((t): t is AudioTrack<"1.0.0"> => t !== undefined);

  if (brainTracks.length === 0) return;

  await ctx.ui.withinProgressDialog("SampleBrain", {}, async (update, signal) => {
    try {
      const tempo = song.tempo;
      const secPerBeat = 60 / tempo;
      const selStart = selection.time_selection_start;
      const selEnd = selection.time_selection_end;

      if (selEnd - selStart < 0.01) {
        console.error("SampleBrain: selection is empty — drag a time range in the Arrangement View before right-clicking");
        return;
      }

      // --- Render target ---
      update("Rendering target…", 5);
      const targetWavPath = await ctx.resources.renderPreFxAudio(targetTrack, selStart, selEnd);
      if (signal.aborted) return;
      const targetData = await decodeWav(targetWavPath);
      if (signal.aborted) return;

      // Guard: target must have audio content
      const targetPeak = targetData.samples.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
      if (targetPeak < 1e-4) {
        console.error(`SampleBrain: target track "${targetTrack.name}" has no audio in the selected range. Right-click on a track that has clips.`);
        return;
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
        const brainEnd = Math.max(...clips.map((c) => c.endTime));
        const brainWavPath = await ctx.resources.renderPreFxAudio(track, 0, brainEnd);
        if (signal.aborted) return;
        const brainData = await decodeWav(brainWavPath);
        brainPcms.push(brainData.samples);
      }

      // --- Decode dropped files ---
      for (let i = 0; i < params.droppedFiles.length; i++) {
        if (signal.aborted) return;
        const filePath = params.droppedFiles[i]!;
        update(`Loading file ${i + 1}/${params.droppedFiles.length}…`, 40 + i * (8 / Math.max(1, params.droppedFiles.length)));
        try {
          const brainData = await decodeWav(filePath);
          brainPcms.push(brainData.samples);
        } catch (err) {
          console.warn(`SampleBrain: skipping dropped file "${filePath}": ${err}`);
        }
      }

      if (brainPcms.length === 0) {
        console.error("SampleBrain: all selected brain tracks are empty");
        return;
      }

      const { sampleRate } = targetData;
      const blockSizeSamples = Math.max(64, Math.round((params.blockSizeMs / 1000) * sampleRate));
      console.log(`SampleBrain: sampleRate=${sampleRate}, blockSize=${blockSizeSamples} samples, target=${targetData.numSamples} samples`);

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
        console.error("SampleBrain: brain produced 0 blocks — try a smaller block size");
        return;
      }

      // --- Analyse target ---
      update("Analysing target…", 60);
      if (signal.aborted) return;
      const targetBlocks = buildTarget(targetData.samples, brainParams);
      console.log(`SampleBrain: ${targetBlocks.length} target blocks`);

      if (targetBlocks.length === 0) {
        console.error("SampleBrain: target produced 0 blocks — selection too short or block size too large");
        return;
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
      };
      const output = synthesize(targetBlocks, brainBlocks, hopSize, matchParams);
      console.log(`SampleBrain: output ${output.length} samples`);

      // --- Write WAV ---
      update("Writing output…", 85);
      if (signal.aborted) return;
      const wavBuf = encodeWav(output, sampleRate, params.ampWeight);
      const tmpDir = ctx.environment.tempDirectory ?? "/tmp";
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
      outputTrack.name = `SB: ${targetTrack.name}`;
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
  brainTrackIndices: number[];
  droppedFiles: string[];
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
}

interface PreviewRequest {
  brainTrackIndices: number[];
  droppedFiles: string[];
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
}
