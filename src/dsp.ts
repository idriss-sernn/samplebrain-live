export type WindowType = "hann" | "hamming" | "triangle" | "rectangle";

export function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

export function applyWindow(block: Float64Array, type: WindowType): void {
  const N = block.length;
  for (let i = 0; i < N; i++) {
    let w = 1;
    if (type === "hann") w = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (N - 1)));
    else if (type === "hamming") w = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (N - 1));
    else if (type === "triangle") w = 1 - Math.abs((2 * i) / (N - 1) - 1);
    block[i]! *= w;
  }
}

// In-place Cooley-Tukey radix-2 FFT — length must be power of 2
export function fft(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]!; re[i] = re[j]!; re[j] = t;
      t = im[i]!; im[i] = im[j]!; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1, curIm = 0;
      for (let j = 0; j < len >> 1; j++) {
        const half = len >> 1;
        const uRe = re[i + j]!;
        const uIm = im[i + j]!;
        const vRe = re[i + j + half]! * curRe - im[i + j + half]! * curIm;
        const vIm = re[i + j + half]! * curIm + im[i + j + half]! * curRe;
        re[i + j] = uRe + vRe;
        im[i + j] = uIm + vIm;
        re[i + j + half] = uRe - vRe;
        im[i + j + half] = uIm - vIm;
        const nr = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nr;
      }
    }
  }
}

// Mel ↔ Hz conversions
function hzToMel(hz: number): number {
  return 2595 * Math.log10(1 + hz / 700);
}
function melToHz(mel: number): number {
  return 700 * (Math.pow(10, mel / 2595) - 1);
}

// Build triangular Mel filterbank — result is [numFilters][fftBins] (fftBins = fftSize/2+1)
function buildMelFilterbank(numFilters: number, fftSize: number, sampleRate: number): Float64Array[] {
  const nyquist = sampleRate / 2;
  const bins = fftSize / 2 + 1;
  const melMin = hzToMel(0);
  const melMax = hzToMel(nyquist);

  const melPoints = Array.from({ length: numFilters + 2 }, (_, i) =>
    melMin + (i / (numFilters + 1)) * (melMax - melMin)
  );
  const hzPoints = melPoints.map(melToHz);
  const binIdx = hzPoints.map((h) => Math.floor((bins - 1) * h / nyquist));

  return Array.from({ length: numFilters }, (_, m) => {
    const filter = new Float64Array(bins);
    const lo = binIdx[m]!;
    const center = binIdx[m + 1]!;
    const hi = binIdx[m + 2]!;
    for (let k = lo; k < center; k++) filter[k] = (k - lo) / Math.max(center - lo, 1);
    for (let k = center; k <= hi; k++) filter[k] = (hi - k) / Math.max(hi - center, 1);
    return filter;
  });
}

// DCT-II
function dct2(x: Float64Array, numCoeffs: number): Float64Array {
  const N = x.length;
  const out = new Float64Array(numCoeffs);
  for (let k = 0; k < numCoeffs; k++) {
    let sum = 0;
    for (let n = 0; n < N; n++) {
      sum += x[n]! * Math.cos((Math.PI * k * (2 * n + 1)) / (2 * N));
    }
    out[k] = sum;
  }
  return out;
}

// Normalise vector to unit L2 norm in-place
export function normalizeL2(v: Float64Array): void {
  let sq = 0;
  for (let i = 0; i < v.length; i++) sq += v[i]! * v[i]!;
  const norm = Math.sqrt(sq);
  if (norm < 1e-12) return;
  for (let i = 0; i < v.length; i++) v[i]! /= norm;
}

export type DescriptorType = "fft" | "mfcc" | "mix";

const MEL_FILTERS = 26;
const MFCC_COEFFS = 13;
const FFT_DESCRIPTOR_BINS = 64;
const filterCache = new Map<string, Float64Array[]>();

function getFilterbank(fftSize: number, sampleRate: number): Float64Array[] {
  const key = `${fftSize}:${sampleRate}`;
  if (!filterCache.has(key)) filterCache.set(key, buildMelFilterbank(MEL_FILTERS, fftSize, sampleRate));
  return filterCache.get(key)!;
}

export function computeDescriptor(
  block: Float64Array,
  sampleRate: number,
  type: DescriptorType,
  mfccRatio: number, // 0.0 = pure FFT, 1.0 = pure MFCC
): Float64Array {
  const fftSize = nextPow2(block.length);
  const re = new Float64Array(fftSize);
  const im = new Float64Array(fftSize);
  re.set(block);
  fft(re, im);

  const bins = fftSize / 2 + 1;
  const mag = new Float64Array(bins);
  for (let k = 0; k < bins; k++) mag[k] = Math.sqrt(re[k]! * re[k]! + im[k]! * im[k]!);

  if (type === "fft") {
    const d = new Float64Array(FFT_DESCRIPTOR_BINS);
    const step = Math.max(1, Math.floor(bins / FFT_DESCRIPTOR_BINS));
    for (let i = 0; i < FFT_DESCRIPTOR_BINS; i++) d[i] = mag[i * step] ?? 0;
    normalizeL2(d);
    return d;
  }

  if (type === "mfcc") {
    const fb = getFilterbank(fftSize, sampleRate);
    const energies = new Float64Array(MEL_FILTERS);
    for (let m = 0; m < MEL_FILTERS; m++) {
      let e = 0;
      for (let k = 0; k < bins; k++) e += mag[k]! * fb[m]![k]!;
      energies[m] = Math.log(Math.max(e, 1e-10));
    }
    const mfccs = dct2(energies, MFCC_COEFFS);
    normalizeL2(mfccs);
    return mfccs;
  }

  // mix: concatenate normalised FFT + normalised MFCC, weighted by ratio
  const fftDesc = new Float64Array(FFT_DESCRIPTOR_BINS);
  const step = Math.max(1, Math.floor(bins / FFT_DESCRIPTOR_BINS));
  for (let i = 0; i < FFT_DESCRIPTOR_BINS; i++) fftDesc[i] = mag[i * step] ?? 0;
  normalizeL2(fftDesc);

  const fb = getFilterbank(fftSize, sampleRate);
  const energies = new Float64Array(MEL_FILTERS);
  for (let m = 0; m < MEL_FILTERS; m++) {
    let e = 0;
    for (let k = 0; k < bins; k++) e += mag[k]! * fb[m]![k]!;
    energies[m] = Math.log(Math.max(e, 1e-10));
  }
  const mfccs = dct2(energies, MFCC_COEFFS);
  normalizeL2(mfccs);

  const fftWeight = 1 - mfccRatio;
  const mix = new Float64Array(FFT_DESCRIPTOR_BINS + MFCC_COEFFS);
  for (let i = 0; i < FFT_DESCRIPTOR_BINS; i++) mix[i] = fftDesc[i]! * fftWeight;
  for (let i = 0; i < MFCC_COEFFS; i++) mix[FFT_DESCRIPTOR_BINS + i] = mfccs[i]! * mfccRatio;
  normalizeL2(mix);
  return mix;
}

export function euclidean(a: Float64Array, b: Float64Array): number {
  let d = 0;
  for (let i = 0; i < a.length; i++) { const diff = a[i]! - b[i]!; d += diff * diff; }
  return d; // squared — no sqrt needed for comparison
}
