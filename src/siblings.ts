// Sample Design — faithful port of "Audio Siblings" v2 (Y R A K I / Mariano Sibilia).
// Batch random pitch-shifting that recontextualises material. Pitch is done by VARISPEED
// (like the original's AudioBufferSource.playbackRate): the whole buffer is resampled, so
// pitch up = shorter, pitch down = longer, and there are NO grain-join clicks.
// Two modes: "pitch" (one transposed sibling per source) and "layered" (stack N randomly
// transposed copies of randomly chosen sources). Pure TS, 0 deps.

import { resampleLinear } from "./wav.js";
import { fft, applyWindow, nextPow2 } from "./dsp.js";

export type Strategy = "chromatic" | "major" | "minor" | "pentatonic";
export type SiblingMode = "pitch" | "layered" | "slice" | "automate";

export interface SiblingParams {
  minSt: number;            // semitone range min (e.g. -12)
  maxSt: number;            // semitone range max (e.g. +12)
  strategy: Strategy;
  mode: SiblingMode;
  layers: number;           // 2–12 (layered mode)
  autoAttenuation: boolean; // per-layer gain 1/N to avoid clipping (layered)
  generations: number;      // number of outputs (layered mode)
}

export interface Sibling {
  pcm: Float64Array;
  sampleRate: number;
  label: string;      // filename suffix, e.g. "+3st" or "layered 2"
  sourceName: string; // base name of the originating source ("layered" for stacks)
}

const SCALE_DEGREES: Record<Exclude<Strategy, "chromatic">, number[]> = {
  major: [0, 2, 4, 5, 7, 9, 11],
  minor: [0, 2, 3, 5, 7, 8, 10],
  pentatonic: [0, 2, 4, 7, 9],
};

// Faithful port of Audio Siblings' pickSemitone: chromatic = uniform integer in range;
// scale strategies = a random allowed semitone whose pitch-class is a scale degree (any octave).
export function pickSemitone(minSt: number, maxSt: number, strategy: Strategy): number {
  const lo = Math.min(minSt, maxSt);
  const hi = Math.max(minSt, maxSt);
  if (strategy === "chromatic") return Math.floor(Math.random() * (hi - lo + 1)) + lo;
  const degs = SCALE_DEGREES[strategy] ?? [];
  const allowed: number[] = [];
  for (let k = Math.floor((lo - 11) / 12) - 1; k <= Math.ceil((hi + 11) / 12) + 1; k++) {
    for (const d of degs) {
      const v = 12 * k + d;
      if (v >= lo && v <= hi) allowed.push(v);
    }
  }
  return allowed.length ? allowed[Math.floor(Math.random() * allowed.length)]! : Math.floor(Math.random() * (hi - lo + 1)) + lo;
}

// Varispeed pitch shift: factor = 2^(st/12); output length = round(len / factor).
// Linear interpolation over the source — same result as AudioBufferSource.playbackRate.
function varispeed(pcm: Float64Array, semitones: number): Float64Array {
  if (pcm.length === 0) return new Float64Array(0);
  const factor = Math.pow(2, semitones / 12);
  if (Math.abs(factor - 1) < 1e-6) return pcm.slice();
  const outLen = Math.max(1, Math.round(pcm.length / factor));
  const out = new Float64Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos = i * factor;
    const idx = Math.floor(pos);
    const frac = pos - idx;
    const a = idx < pcm.length ? pcm[idx]! : 0;
    const b = idx + 1 < pcm.length ? pcm[idx + 1]! : a;
    out[i] = a + frac * (b - a);
  }
  return out;
}

const labelFor = (st: number): string => `${st >= 0 ? "+" : ""}${st}st`;

// ── Pitch automation (like a clip Transposition envelope, baked into the audio) ──
// A random clean envelope shape (ramp up/down, exponential ease, peak/return) drives a
// continuous varispeed read, so the pitch glides audibly across the sample. The SDK can't
// write Live clip envelopes, so we bake it. The sweep is even across the OUTPUT timeline
// (what you hear), so the glide is felt from start to end rather than collapsing into the
// first instant when pitching up.

const AUTO_E = 256;

// Random transposition envelope: a glide between TWO endpoints (semitones over normalised time
// 0..1), not a wobble around the original pitch. Anchoring every glide at 0st (as before) made
// them all sound alike and timid; picking two real endpoints with a guaranteed interval makes the
// sweep clearly heard and musically varied (e.g. −5→+5, or +7 sliding back down to 0).
function randomEnvelope(minSt: number, maxSt: number, strategy: Strategy): { env: Float64Array; from: number; to: number } {
  const span = maxSt - minSt;
  let a = pickSemitone(minSt, maxSt, strategy);
  let b = pickSemitone(minSt, maxSt, strategy);
  // Force an audible interval: anything under ~3st barely reads as a glide. Push b away from a,
  // keeping it inside the range and preserving a random direction where possible.
  const minInterval = Math.min(3, span);
  if (span > 0 && Math.abs(b - a) < minInterval) {
    const dir = b >= a ? 1 : -1;
    b = a + dir * minInterval;
    if (b > maxSt || b < minSt) b = a - dir * minInterval; // flip if we ran off the edge
    b = Math.max(minSt, Math.min(maxSt, b));
  }
  const shape = Math.floor(Math.random() * 4);
  const env = new Float64Array(AUTO_E);
  for (let i = 0; i < AUTO_E; i++) {
    const t = i / (AUTO_E - 1);
    let c: number;                                       // 0..1 progress along the glide
    if (shape === 0) c = t;                              // linear
    else if (shape === 1) c = 1 - (1 - t) * (1 - t);     // ease-out (fast then settle)
    else if (shape === 2) c = t * t;                     // ease-in (slow then accelerate)
    else c = (1 - Math.cos(t * Math.PI)) / 2;            // smoothstep S-curve
    env[i] = a + (b - a) * c;
  }
  return { env, from: a, to: b };
}

// Varispeed driven by the envelope, paced by OUTPUT time. The output length is derived from
// the envelope's average playback rate (outLen ≈ N / avgRate), so reading the source at the
// per-output-sample rate consumes exactly the whole source while the pitch envelope evolves
// EVENLY across what you hear. Indexing by the read position instead would race through the
// envelope whenever the pitch rises, collapsing the glide into the first instant.
function pitchAutomate(pcm: Float64Array, env: Float64Array): Float64Array {
  const N = pcm.length;
  if (N < 2) return pcm.slice();
  // Average rate over the sweep → output length that makes the read land on the source end.
  let avgRate = 0;
  for (let i = 0; i < AUTO_E; i++) avgRate += Math.pow(2, env[i]! / 12);
  avgRate /= AUTO_E;
  const outLen = Math.max(2, Math.min(N * 6, Math.round(N / Math.max(avgRate, 1e-6))));
  const out = new Float64Array(outLen);
  let pos = 0;
  for (let n = 0; n < outLen; n++) {
    const idx = Math.min(Math.floor(pos), N - 1), frac = pos - idx;
    const a = pcm[idx]!, b = idx + 1 < N ? pcm[idx + 1]! : a;
    out[n] = a + frac * (b - a);
    const st = env[Math.min(AUTO_E - 1, Math.floor((n / outLen) * AUTO_E))]!;
    pos += Math.pow(2, st / 12);
    if (pos > N - 1) pos = N - 1; // ride the tail instead of cutting if rounding overshoots
  }
  return out;
}

// ── Onset detection (spectral flux) + slicing into one-shots ────────────────
// Cuts a loop into its individual hits (hats, claps, kick…) so Sample Design can
// output separate one-shots instead of one whole pitched loop.
function detectOnsets(pcm: Float64Array, sampleRate: number): number[] {
  const fftSize = 1024;
  const hop = 512;
  const N = pcm.length;
  if (N < fftSize * 2) return [0];
  const half = fftSize / 2 + 1;

  const flux: number[] = [];
  const re = new Float64Array(fftSize);
  const im = new Float64Array(fftSize);
  let prev = new Float64Array(half);
  for (let pos = 0; pos + fftSize <= N; pos += hop) {
    re.fill(0); im.fill(0);
    for (let i = 0; i < fftSize; i++) re[i] = pcm[pos + i]!;
    applyWindow(re, "hann");
    fft(re, im);
    const cur = new Float64Array(half);
    let f = 0;
    for (let k = 0; k < half; k++) {
      const m = Math.sqrt(re[k]! * re[k]! + im[k]! * im[k]!);
      cur[k] = m;
      const d = m - prev[k]!;
      if (d > 0) f += d;
    }
    flux.push(f);
    prev = cur;
  }

  // Peak-pick the flux with an adaptive (moving-mean) threshold + min spacing
  const W = 8;
  const minGap = Math.max(1, Math.round((0.05 * sampleRate) / hop)); // 50 ms
  const onsets: number[] = [];
  let last = -minGap;
  for (let i = 1; i < flux.length - 1; i++) {
    let sum = 0, cnt = 0;
    for (let j = i - W; j <= i + W; j++) if (j >= 0 && j < flux.length) { sum += flux[j]!; cnt++; }
    const thr = (sum / cnt) * 1.5 + 1e-6;
    if (flux[i]! > thr && flux[i]! >= flux[i - 1]! && flux[i]! > flux[i + 1]! && i - last >= minGap) {
      onsets.push(i * hop);
      last = i;
    }
  }
  if (onsets.length === 0 || onsets[0]! > hop) onsets.unshift(0);
  return onsets;
}

// ── Drum-hit classification by spectral analysis ────────────────────────────
// Heuristic (no ML): averaged magnitude spectrum → centroid / flatness / band energy
// → broad category. Reliable for kick/hat/tom; snare vs clap is fuzzy (both broadband
// noise) so the mid-noisy class is labelled "snare". Thresholds are tunable.
interface SpecFeat { E: number; cen: number; lowR: number; hiR: number; bodyR: number; flat: number; }

function spectrumFeatures(pcm: Float64Array, sampleRate: number): SpecFeat | null {
  const fftSize = 2048;
  const half = fftSize / 2 + 1;
  if (pcm.length < 256) return null;
  const re = new Float64Array(fftSize);
  const im = new Float64Array(fftSize);
  const avg = new Float64Array(half);
  const frames = Math.max(1, Math.min(12, Math.floor((pcm.length - fftSize) / 1024) + 1));
  let nf = 0;
  for (let fr = 0; fr < frames; fr++) {
    const pos = fr * 1024;
    re.fill(0); im.fill(0);
    for (let i = 0; i < fftSize && pos + i < pcm.length; i++) re[i] = pcm[pos + i]!;
    applyWindow(re, "hann");
    fft(re, im);
    for (let k = 0; k < half; k++) avg[k]! += Math.sqrt(re[k]! * re[k]! + im[k]! * im[k]!);
    nf++;
  }
  for (let k = 0; k < half; k++) avg[k]! /= nf;

  const binHz = sampleRate / fftSize;
  let E = 0, cenNum = 0;
  for (let k = 0; k < half; k++) { E += avg[k]!; cenNum += k * binHz * avg[k]!; }
  if (E <= 1e-9) return null;
  const cen = cenNum / E;
  const band = (lo: number, hi: number) => {
    let s = 0;
    for (let k = 0; k < half; k++) { const f = k * binHz; if (f >= lo && f < hi) s += avg[k]!; }
    return s;
  };
  const lowR = (band(0, 120) + band(120, 300)) / E;
  const hiR = (band(2500, 7000) + band(7000, sampleRate / 2 + 1)) / E;
  const bodyR = band(200, 2000) / E; // low-mid "body" — present in snares/toms, scarce in hats
  let logsum = 0, arith = 0, cnt = 0;
  for (let k = 1; k < half; k++) { const m = avg[k]! + 1e-12; logsum += Math.log(m); arith += m; cnt++; }
  const flat = cnt > 0 ? Math.exp(logsum / cnt) / (arith / cnt) : 0;
  return { E, cen, lowR, hiR, bodyR, flat };
}

function classifySlice(pcm: Float64Array, sampleRate: number): string {
  const f = spectrumFeatures(pcm, sampleRate);
  if (!f) return "perc";
  if (f.lowR > 0.5 && f.cen < 300) return "kick";
  // hat: very bright + noisy + little low-mid body
  if (f.cen > 6000 && f.flat > 0.12 && f.bodyR < 0.18) return "hat";
  // snare/clap: noisy with real low-mid body
  if (f.flat > 0.1 && f.cen > 1000 && f.bodyR > 0.2) return "snare";
  if (f.cen < 800 && f.flat < 0.1) return "tom";
  return "perc";
}

// Onset positions as fractions (0–1) of the buffer — for drawing clickable hit markers.
export function detectOnsetFractions(pcm: Float64Array, sampleRate: number): number[] {
  if (pcm.length === 0) return [];
  return detectOnsets(pcm, sampleRate).map((s) => s / pcm.length);
}

// Slice a source at its onsets into one-shots, with a small pre-roll and de-click fades.
function sliceSource(pcm: Float64Array, sampleRate: number): Float64Array[] {
  const onsets = detectOnsets(pcm, sampleRate);
  const N = pcm.length;
  const preroll = Math.round(sampleRate * 0.003);
  const minLen = Math.round(sampleRate * 0.02); // drop slivers < 20 ms
  const fadeIn = Math.round(sampleRate * 0.001);
  const fadeOut = Math.round(sampleRate * 0.008);
  const slices: Float64Array[] = [];

  for (let k = 0; k < onsets.length; k++) {
    const start = Math.max(0, onsets[k]! - preroll);
    const end = k + 1 < onsets.length ? onsets[k + 1]! : N;
    const len = end - start;
    if (len < minLen) continue;
    const slice = new Float64Array(len);
    let peak = 0;
    for (let i = 0; i < len; i++) { const v = pcm[start + i]!; slice[i] = v; if (Math.abs(v) > peak) peak = Math.abs(v); }
    if (peak < 1e-4) continue; // skip silence
    for (let i = 0; i < fadeIn && i < len; i++) slice[i]! *= i / fadeIn;
    for (let i = 0; i < fadeOut && i < len; i++) slice[len - 1 - i]! *= i / fadeOut;
    slices.push(slice);
  }
  return slices;
}

export function generateSiblings(
  sources: { pcm: Float64Array; sampleRate: number; name: string }[],
  params: SiblingParams,
): Sibling[] {
  const usable = sources.filter((s) => s.pcm.length > 0);
  if (usable.length === 0) return [];
  const { minSt, maxSt, strategy } = params;
  const out: Sibling[] = [];

  if (params.mode === "slice") {
    // Cut each source into one-shots and classify each by spectral analysis
    for (const src of usable) {
      const slices = sliceSource(src.pcm, src.sampleRate);
      const counts: Record<string, number> = {};
      for (const pcm of slices) {
        const type = classifySlice(pcm, src.sampleRate);
        counts[type] = (counts[type] ?? 0) + 1;
        out.push({ pcm, sampleRate: src.sampleRate, label: `${type} ${counts[type]}`, sourceName: src.name });
      }
    }
    return out;
  }

  if (params.mode === "pitch") {
    // N transposed variations per source (N = generations)
    const gens = Math.max(1, Math.round(params.generations));
    for (const src of usable) {
      for (let g = 0; g < gens; g++) {
        const st = pickSemitone(minSt, maxSt, strategy);
        out.push({ pcm: varispeed(src.pcm, st), sampleRate: src.sampleRate, label: labelFor(st), sourceName: src.name });
      }
    }
    return out;
  }

  if (params.mode === "automate") {
    // N variations per source, each with a unique random pitch-transposition envelope (baked,
    // even sweep across the output) — like dropping a random clip Transposition automation
    const gens = Math.max(1, Math.round(params.generations));
    for (const src of usable) {
      for (let g = 0; g < gens; g++) {
        const { env, from, to } = randomEnvelope(minSt, maxSt, strategy);
        out.push({ pcm: pitchAutomate(src.pcm, env), sampleRate: src.sampleRate, label: `${labelFor(from)}→${labelFor(to)}`, sourceName: src.name });
      }
    }
    return out;
  }

  // Layered: `generations` outputs, each a stack of `layers` randomly chosen, transposed copies
  const numLayers = Math.max(2, Math.min(12, Math.round(params.layers)));
  const gens = Math.max(1, Math.round(params.generations));
  const gain = params.autoAttenuation ? 1 / numLayers : 1;

  for (let g = 0; g < gens; g++) {
    // Pick numLayers sources without repeats when possible (reuse if numLayers > sources)
    const pool: number[] = [];
    const chosen: number[] = [];
    for (let i = 0; i < numLayers; i++) {
      if (pool.length === 0) for (let j = 0; j < usable.length; j++) pool.push(j);
      chosen.push(pool.splice(Math.floor(Math.random() * pool.length), 1)[0]!);
    }
    const targetSr = usable[chosen[0]!]!.sampleRate;
    const layerPcms = chosen.map((idx) => {
      const s = usable[idx]!;
      const base = s.sampleRate === targetSr ? s.pcm : resampleLinear(s.pcm, s.sampleRate, targetSr);
      return varispeed(base, pickSemitone(minSt, maxSt, strategy));
    });
    const len = layerPcms.reduce((m, p) => Math.max(m, p.length), 0);
    const mix = new Float64Array(len);
    for (const p of layerPcms) for (let i = 0; i < p.length; i++) mix[i]! += p[i]! * gain;
    out.push({ pcm: mix, sampleRate: targetSr, label: `layered ${g + 1}`, sourceName: "layered" });
  }
  return out;
}
