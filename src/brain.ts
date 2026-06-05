import {
  applyWindow,
  computeDescriptor,
  euclidean,
  type DescriptorType,
  type WindowType,
} from "./dsp.js";

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
  novelty: number;    // 0.0–1.0 — bias against already-used blocks
  boredom: number;    // 0.0–1.0 — decay rate of usage counts
  stickiness: number; // 0.0–1.0 — tendency to follow adjacent blocks
  voices: number;     // 1–4 — simultaneous brain blocks per target grain
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

// Hann window for OLA synthesis
function hannWindow(size: number): Float64Array {
  const w = new Float64Array(size);
  for (let i = 0; i < size; i++) w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (size - 1)));
  return w;
}

export function synthesize(
  targetBlocks: Block[],
  brainBlocks: Block[],
  hopSize: number,
  matchParams: MatchParams,
): Float64Array {
  if (brainBlocks.length === 0 || targetBlocks.length === 0) return new Float64Array(0);

  const { novelty, boredom, stickiness, voices } = matchParams;
  const numVoices = Math.max(1, Math.min(4, Math.round(voices)));
  const blockSize = brainBlocks[0]!.pcm.length;
  const outputLen = hopSize * targetBlocks.length + blockSize;
  const output = new Float64Array(outputLen);
  const normSum = new Float64Array(outputLen);

  const win = hannWindow(blockSize);
  const usageCount = new Float64Array(brainBlocks.length);
  let lastIdx = -1;
  let stickyRemaining = 0;

  const stickinessThreshold = 0.5 * stickiness;

  for (let t = 0; t < targetBlocks.length; t++) {
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

    if (numVoices === 1) {
      usageCount[primaryIdx]! += 1;
      const brainPcm = brainBlocks[primaryIdx]!.pcm;
      const outPos = t * hopSize;
      for (let i = 0; i < blockSize && outPos + i < outputLen; i++) {
        output[outPos + i]! += brainPcm[i]! * win[i]!;
        normSum[outPos + i]! += win[i]!;
      }
    } else {
      const topN = knnTopN(targetDesc, brainBlocks, usageCount, novelty, numVoices);
      const outPos = t * hopSize;
      for (const { idx, weight } of topN) {
        usageCount[idx]! += 1;
        const brainPcm = brainBlocks[idx]!.pcm;
        for (let i = 0; i < blockSize && outPos + i < outputLen; i++) {
          output[outPos + i]! += brainPcm[i]! * win[i]! * weight;
          normSum[outPos + i]! += win[i]! * weight;
        }
      }
    }
  }

  for (let i = 0; i < outputLen; i++) {
    if (normSum[i]! > 1e-9) output[i]! /= normSum[i]!;
  }

  return output;
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
