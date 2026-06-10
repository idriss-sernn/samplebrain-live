import {
  applyWindow,
  computeDescriptor,
  euclidean,
  type DescriptorType,
  type WindowType,
} from "./dsp.js";

// Tukey (tapered cosine) window for OLA: flat centre + raised-cosine edges of length
// `taper`. With taper = overlap region (blockSize - hopSize), adjacent grains crossfade
// exactly over their overlap while their centres play at full amplitude — no seam clicks
// (low overlap) and no scalloping/pumping of transients. At 50% overlap it equals a Hann.
function tukeyWindow(size: number, taper: number): Float64Array {
  const w = new Float64Array(size);
  const t = Math.max(1, Math.min(taper, size >> 1));
  for (let i = 0; i < size; i++) {
    if (i < t) w[i] = 0.5 * (1 - Math.cos((Math.PI * i) / t));
    else if (i >= size - t) w[i] = 0.5 * (1 - Math.cos((Math.PI * (size - 1 - i)) / t));
    else w[i] = 1;
  }
  return w;
}

export interface Block {
  pcm: Float64Array;
  descriptor: Float64Array;
}

export interface BrainParams {
  blockSizeSamples: number;
  overlapRatio: number; // 0.0–0.75
  windowType: WindowType;
  descriptorType: DescriptorType;
  mfccRatio: number; // 0.0–1.0, only used when descriptorType === "mix"
  sampleRate: number;
}

export interface MatchParams {
  novelty: number;      // 0.0–1.0 — bias against already-used blocks
  boredom: number;      // 0.0–1.0 — decay rate of usage counts
  stickiness: number;   // 0.0–1.0 — greedy: follow adjacent blocks / smooth: continuity weight λ
  voices: number;       // 1–4 — simultaneous brain blocks per target grain (greedy only)
  pitchShift: number;   // semitones center offset (negative = down)
  pitchShiftVar: number;// semitones random variance ≥ 0
  reverse: number;      // 0.0–1.0 — probability grain is played backwards
  density: number;      // 0.0–1.0 — probability a grain is created at all
  matchMode: "greedy" | "smooth"; // greedy = original per-frame kNN; smooth = Viterbi unit-selection
}

function segmentPcm(pcm: Float64Array, params: BrainParams): Block[] {
  const { blockSizeSamples, overlapRatio, windowType, descriptorType, mfccRatio, sampleRate } = params;
  const hopSize = Math.max(1, Math.round(blockSizeSamples * (1 - overlapRatio)));
  const blocks: Block[] = [];

  for (let start = 0; start + blockSizeSamples <= pcm.length; start += hopSize) {
    const slice = new Float64Array(blockSizeSamples);
    for (let i = 0; i < blockSizeSamples; i++) slice[i] = pcm[start + i]!;
    applyWindow(slice, windowType);
    const descriptor = computeDescriptor(slice, sampleRate, descriptorType, mfccRatio);
    // Store unwindowed pcm for resynthesis
    const rawSlice = new Float64Array(blockSizeSamples);
    for (let i = 0; i < blockSizeSamples; i++) rawSlice[i] = pcm[start + i]!;
    blocks.push({ pcm: rawSlice, descriptor });
  }

  return blocks;
}

export function buildBrain(sources: Float64Array[], params: BrainParams): Block[] {
  const all: Block[] = [];
  for (const src of sources) all.push(...segmentPcm(src, params));
  return all;
}

export function buildTarget(pcm: Float64Array, params: BrainParams): Block[] {
  return segmentPcm(pcm, params);
}

export function synthesize(
  targetBlocks: Block[],
  brainBlocks: Block[],
  hopSize: number,
  matchParams: MatchParams,
): Float64Array {
  if (brainBlocks.length === 0 || targetBlocks.length === 0) return new Float64Array(0);

  if (matchParams.matchMode === "smooth") return synthesizeSmooth(targetBlocks, brainBlocks, hopSize, matchParams);

  const { novelty, boredom, stickiness, voices, pitchShift, pitchShiftVar, reverse, density } = matchParams;
  const numVoices = Math.max(1, Math.min(4, Math.round(voices)));
  const blockSize = brainBlocks[0]!.pcm.length;
  const outputLen = hopSize * targetBlocks.length + blockSize;
  const output = new Float64Array(outputLen);
  const normSum = new Float64Array(outputLen);

  // Taper = overlap region so the cosine edges line up with the crossfade zone
  const taper = Math.max(1, Math.min(blockSize - hopSize, blockSize >> 1));
  const win = tukeyWindow(blockSize, taper);
  const usageCount = new Float64Array(brainBlocks.length);
  let lastIdx = -1;
  let stickyRemaining = 0;

  const stickinessThreshold = 0.5 * stickiness;

  for (let t = 0; t < targetBlocks.length; t++) {
    // density — skip grain probabilistically
    if (density < 1 && Math.random() > density) continue;

    const targetDesc = targetBlocks[t]!.descriptor;

    let primaryIdx: number;
    if (stickyRemaining > 0 && lastIdx >= 0) {
      const nextIdx = (lastIdx + 1) % brainBlocks.length;
      const d = euclidean(targetDesc, brainBlocks[nextIdx]!.descriptor);
      if (d < stickinessThreshold) {
        primaryIdx = nextIdx;
        stickyRemaining--;
      } else {
        stickyRemaining = 0;
        primaryIdx = knnSearch(targetDesc, brainBlocks, usageCount, novelty);
        stickyRemaining = Math.round(stickiness * 8);
      }
    } else {
      primaryIdx = knnSearch(targetDesc, brainBlocks, usageCount, novelty);
      stickyRemaining = Math.round(stickiness * 8);
    }
    lastIdx = primaryIdx;

    for (let i = 0; i < usageCount.length; i++) usageCount[i]! *= (1 - boredom * 0.1);

    const outPos = t * hopSize;

    if (numVoices === 1) {
      usageCount[primaryIdx]! += 1;
      olaGrain(brainBlocks[primaryIdx]!.pcm, win, output, normSum, outPos, outputLen, pitchShift, pitchShiftVar, reverse, 1);
    } else {
      const topN = knnTopN(targetDesc, brainBlocks, usageCount, novelty, numVoices);
      for (const { idx, weight } of topN) {
        usageCount[idx]! += 1;
        olaGrain(brainBlocks[idx]!.pcm, win, output, normSum, outPos, outputLen, pitchShift, pitchShiftVar, reverse, weight);
      }
    }
  }

  for (let i = 0; i < outputLen; i++) {
    if (normSum[i]! > 1e-9) output[i]! /= normSum[i]!;
  }

  return output;
}

// ── SMOOTH mode: unit-selection via Viterbi ────────────────────────────────
// Globally optimal grain sequence minimising Σ targetCost + λ·Σ concatCost, instead
// of greedy per-frame picks. Continuity is rewarded (a grain that follows the previous
// one in the source costs 0 to join), so the output stitches together coherent runs.

const VITERBI_K = 16; // candidate grains pre-selected per target frame

// K brain blocks with lowest target cost (descriptor distance + reuse penalty), ascending.
function topKCandidates(targetDesc: Float64Array, brain: Block[], K: number, penalty: Float64Array): number[] {
  const k = Math.min(K, brain.length);
  const idxs = new Array<number>(k).fill(-1);
  const dists = new Array<number>(k).fill(Infinity);
  for (let i = 0; i < brain.length; i++) {
    const d = euclidean(targetDesc, brain[i]!.descriptor) + penalty[i]!;
    if (d < dists[k - 1]!) {
      let p = k - 1;
      while (p > 0 && dists[p - 1]! > d) { dists[p] = dists[p - 1]!; idxs[p] = idxs[p - 1]!; p--; }
      dists[p] = d; idxs[p] = i;
    }
  }
  return idxs.filter((x) => x >= 0);
}

// `penalty[i]` is added to grain i's target cost — used to push successive voices onto
// different material so polyphonic layers don't collapse onto the same path.
function chooseSequenceViterbi(
  targetBlocks: Block[],
  brainBlocks: Block[],
  lambda: number,
  novelty: number,
  penalty: Float64Array,
): Int32Array {
  const T = targetBlocks.length;
  const cand: number[][] = new Array(T);
  for (let t = 0; t < T; t++) cand[t] = topKCandidates(targetBlocks[t]!.descriptor, brainBlocks, VITERBI_K, penalty);

  const cost: Float64Array[] = new Array(T);
  const back: Int32Array[] = new Array(T);

  cost[0] = new Float64Array(cand[0]!.length);
  for (let k = 0; k < cand[0]!.length; k++) {
    const j0 = cand[0]![k]!;
    cost[0]![k] = euclidean(targetBlocks[0]!.descriptor, brainBlocks[j0]!.descriptor) + penalty[j0]!;
  }

  for (let t = 1; t < T; t++) {
    const ct = cand[t]!, cp = cand[t - 1]!;
    cost[t] = new Float64Array(ct.length);
    back[t] = new Int32Array(ct.length);
    for (let k = 0; k < ct.length; k++) {
      const j = ct[k]!;
      const tc = euclidean(targetBlocks[t]!.descriptor, brainBlocks[j]!.descriptor) + penalty[j]!;
      let best = Infinity, bestp = 0;
      for (let p = 0; p < cp.length; p++) {
        const i = cp[p]!;
        // concat cost: free if grain j follows i in the source; anti-stutter on repeats;
        // otherwise the spectral jump between the two grains
        const cc = j === i + 1 ? 0 : j === i ? novelty : euclidean(brainBlocks[i]!.descriptor, brainBlocks[j]!.descriptor);
        const v = cost[t - 1]![p]! + lambda * cc;
        if (v < best) { best = v; bestp = p; }
      }
      cost[t]![k] = best + tc;
      back[t]![k] = bestp;
    }
  }

  const seq = new Int32Array(T);
  let bk = 0, bestc = Infinity;
  const last = cand[T - 1]!;
  for (let k = 0; k < last.length; k++) if (cost[T - 1]![k]! < bestc) { bestc = cost[T - 1]![k]!; bk = k; }
  seq[T - 1] = last[bk]!;
  for (let t = T - 1; t > 0; t--) { bk = back[t]![bk]!; seq[t - 1] = cand[t - 1]![bk]!; }
  return seq;
}

function synthesizeSmooth(
  targetBlocks: Block[],
  brainBlocks: Block[],
  hopSize: number,
  matchParams: MatchParams,
): Float64Array {
  const { stickiness, novelty, voices, pitchShift, pitchShiftVar, reverse, density } = matchParams;
  const numVoices = Math.max(1, Math.min(4, Math.round(voices)));
  const blockSize = brainBlocks[0]!.pcm.length;
  const outputLen = hopSize * targetBlocks.length + blockSize;
  const output = new Float64Array(outputLen);
  const normSum = new Float64Array(outputLen);

  const taper = Math.max(1, Math.min(blockSize - hopSize, blockSize >> 1));
  const win = tukeyWindow(blockSize, taper);

  // One coherent Viterbi path per voice; grains used by earlier voices are strongly
  // penalised so later voices fall back to distinct material (another coherent run),
  // guaranteeing Voices always thickens the texture rather than duplicating a path.
  const lambda = stickiness * 3; // λ: 0 = pure target match, higher = longer source runs
  const EXCLUDE = 1e3;           // far above any descriptor distance → effectively excludes
  const penalty = new Float64Array(brainBlocks.length);
  const sequences: Int32Array[] = [];
  for (let v = 0; v < numVoices; v++) {
    const seq = chooseSequenceViterbi(targetBlocks, brainBlocks, lambda, novelty, penalty);
    sequences.push(seq);
    if (v < numVoices - 1) for (let t = 0; t < seq.length; t++) penalty[seq[t]!]! += EXCLUDE;
  }

  const weight = 1 / numVoices;
  for (let t = 0; t < targetBlocks.length; t++) {
    if (density < 1 && Math.random() > density) continue;
    for (const seq of sequences) {
      olaGrain(brainBlocks[seq[t]!]!.pcm, win, output, normSum, t * hopSize, outputLen, pitchShift, pitchShiftVar, reverse, weight);
    }
  }

  for (let i = 0; i < outputLen; i++) {
    if (normSum[i]! > 1e-9) output[i]! /= normSum[i]!;
  }
  return output;
}

function olaGrain(
  src: Float64Array,
  win: Float64Array,
  output: Float64Array,
  normSum: Float64Array,
  outPos: number,
  outputLen: number,
  pitchShift: number,
  pitchShiftVar: number,
  reverse: number,
  weight: number,
): void {
  const blockSize = src.length;
  const semitones = pitchShift + (pitchShiftVar > 0 ? (Math.random() * 2 - 1) * pitchShiftVar : 0);
  const pitched = Math.abs(semitones) > 0.01 ? resampleGrain(src, semitones) : src;
  const rev = reverse > 0 && Math.random() < reverse;

  for (let i = 0; i < blockSize && outPos + i < outputLen; i++) {
    const s = pitched[rev ? blockSize - 1 - i : i]!;
    output[outPos + i]! += s * win[i]! * weight;
    normSum[outPos + i]! += win[i]! * weight;
  }
}

function resampleGrain(pcm: Float64Array, semitones: number): Float64Array {
  const ratio = Math.pow(2, semitones / 12);
  const out = new Float64Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) {
    const pos = i * ratio;
    const idx = Math.floor(pos);
    const frac = pos - idx;
    const a = idx < pcm.length ? pcm[idx]! : 0;
    const b = idx + 1 < pcm.length ? pcm[idx + 1]! : 0;
    out[i] = a + frac * (b - a);
  }
  return out;
}

function knnSearch(
  target: Float64Array,
  brain: Block[],
  usageCount: Float64Array,
  novelty: number,
): number {
  let bestIdx = 0;
  let bestScore = Infinity;
  for (let i = 0; i < brain.length; i++) {
    const dist = euclidean(target, brain[i]!.descriptor);
    const score = dist + novelty * usageCount[i]!;
    if (score < bestScore) {
      bestScore = score;
      bestIdx = i;
    }
  }
  return bestIdx;
}

function knnTopN(
  target: Float64Array,
  brain: Block[],
  usageCount: Float64Array,
  novelty: number,
  n: number,
): { idx: number; weight: number }[] {
  const scored: { idx: number; score: number }[] = [];
  for (let i = 0; i < brain.length; i++) {
    const dist = euclidean(target, brain[i]!.descriptor);
    scored.push({ idx: i, score: dist + novelty * usageCount[i]! });
  }
  scored.sort((a, b) => a.score - b.score);
  const top = scored.slice(0, Math.min(n, scored.length));
  // Weight by inverse score so the closest match is loudest
  const invSum = top.reduce((s, x) => s + 1 / (x.score + 1e-9), 0);
  return top.map(x => ({ idx: x.idx, weight: (1 / (x.score + 1e-9)) / invSum }));
}
