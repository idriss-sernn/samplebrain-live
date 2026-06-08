import * as fs from "node:fs/promises";

export interface WavData {
  samples: Float64Array; // mono, normalised [-1, 1]
  sampleRate: number;
  channels: number;
  numSamples: number;
}

export async function decodeWav(filePath: string): Promise<WavData> {
  const buf = await fs.readFile(filePath);
  const magic = buf.toString("ascii", 0, 4);

  if (magic === "FORM") return decodeAiff(buf);
  if (magic === "RIFF") return decodeRiff(buf);

  // Log actual bytes to help diagnose unknown formats
  const hexDump = buf.subarray(0, 16).toString("hex");
  throw new Error(`Unknown audio format. Magic bytes: "${magic}" (${hexDump})`);
}

// ─── RIFF/WAV ───────────────────────────────────────────────────────────────

function decodeRiff(buf: Buffer): WavData {
  if (buf.toString("ascii", 8, 12) !== "WAVE") {
    throw new Error("RIFF file is not WAVE");
  }

  let offset = 12;
  let fmtChunk: Buffer | null = null;
  let dataChunk: Buffer | null = null;

  while (offset < buf.length - 8) {
    const chunkId = buf.toString("ascii", offset, offset + 4);
    const chunkSize = buf.readUInt32LE(offset + 4);
    const chunkData = buf.subarray(offset + 8, offset + 8 + chunkSize);
    if (chunkId === "fmt ") fmtChunk = chunkData;
    else if (chunkId === "data") dataChunk = chunkData;
    offset += 8 + chunkSize + (chunkSize % 2);
    if (fmtChunk && dataChunk) break;
  }

  if (!fmtChunk || !dataChunk) throw new Error("Malformed WAV: missing fmt or data chunk");

  const audioFormat = fmtChunk.readUInt16LE(0); // 1=PCM, 3=IEEE float
  const channels = fmtChunk.readUInt16LE(2);
  const sampleRate = fmtChunk.readUInt32LE(4);
  const bitsPerSample = fmtChunk.readUInt16LE(14);
  const bytesPerSample = bitsPerSample / 8;
  if (channels === 0 || bytesPerSample === 0) throw new Error(`Malformed WAV: invalid format (channels=${channels}, bits=${bitsPerSample})`);
  const numFrames = Math.floor(dataChunk.length / (channels * bytesPerSample));

  const mono = new Float64Array(numFrames);
  for (let i = 0; i < numFrames; i++) {
    let sum = 0;
    for (let c = 0; c < channels; c++) {
      const pos = (i * channels + c) * bytesPerSample;
      let s = 0;
      if (audioFormat === 3) {
        if (bitsPerSample === 32) s = dataChunk.readFloatLE(pos);
        else if (bitsPerSample === 64) s = dataChunk.readDoubleLE(pos);
      } else {
        if (bitsPerSample === 8) s = (dataChunk.readUInt8(pos) - 128) / 128;
        else if (bitsPerSample === 16) s = dataChunk.readInt16LE(pos) / 32768;
        else if (bitsPerSample === 24) {
          const raw = dataChunk.readUIntLE(pos, 3);
          s = (raw >= 0x800000 ? raw - 0x1000000 : raw) / 8388608;
        } else if (bitsPerSample === 32) s = dataChunk.readInt32LE(pos) / 2147483648;
      }
      sum += s;
    }
    mono[i] = sum / channels;
  }

  return { samples: mono, sampleRate, channels, numSamples: numFrames };
}

// ─── AIFF / AIFF-C ──────────────────────────────────────────────────────────

// 80-bit IEEE 754 extended float (big-endian) — used for AIFF sample rate
function readExtended80(buf: Buffer, offset: number): number {
  const exp = ((buf[offset]! & 0x7F) << 8) | buf[offset + 1]!;
  const mantHi = buf.readUInt32BE(offset + 2);
  const mantLo = buf.readUInt32BE(offset + 6);
  if (exp === 0 && mantHi === 0 && mantLo === 0) return 0;
  const mant = mantHi / 0x80000000 + mantLo / 0x8000000000000000;
  return Math.pow(2, exp - 16383) * mant;
}

function decodeAiff(buf: Buffer): WavData {
  const formType = buf.toString("ascii", 8, 12);
  const isAifc = formType === "AIFC";

  let offset = 12;
  let channels = 0;
  let sampleRate = 0;
  let bitsPerSample = 0;
  let numSampleFrames = 0;
  let compressionType = "NONE";
  let soundData: Buffer | null = null;

  while (offset < buf.length - 8) {
    const chunkId = buf.toString("ascii", offset, offset + 4);
    const chunkSize = buf.readUInt32BE(offset + 4);
    const chunkData = buf.subarray(offset + 8, offset + 8 + chunkSize);

    if (chunkId === "COMM") {
      channels = chunkData.readUInt16BE(0);
      numSampleFrames = chunkData.readUInt32BE(2);
      bitsPerSample = chunkData.readUInt16BE(6);
      sampleRate = Math.round(readExtended80(chunkData, 8));
      if (isAifc && chunkData.length > 18) {
        compressionType = chunkData.toString("ascii", 18, 22).trim().toUpperCase();
      }
    } else if (chunkId === "SSND") {
      const ssndOffset = chunkData.readUInt32BE(0);
      soundData = chunkData.subarray(8 + ssndOffset);
    }

    offset += 8 + chunkSize + (chunkSize % 2);
    if (channels && soundData) break;
  }

  if (!channels || !soundData || !sampleRate) {
    throw new Error("Malformed AIFF: missing COMM or SSND chunk");
  }

  // fl32 / fl64 = IEEE float (AIFF-C)
  const isFloat = compressionType === "FL32" || compressionType === "FL64";
  const bytesPerSample = bitsPerSample / 8;
  if (bytesPerSample === 0) throw new Error(`Malformed AIFF: invalid bit depth (bits=${bitsPerSample})`);
  const numFrames = numSampleFrames || Math.floor(soundData.length / (channels * bytesPerSample));
  const mono = new Float64Array(numFrames);

  for (let i = 0; i < numFrames; i++) {
    let sum = 0;
    for (let c = 0; c < channels; c++) {
      const pos = (i * channels + c) * bytesPerSample;
      let s = 0;
      if (isFloat) {
        if (bitsPerSample === 32) s = soundData.readFloatBE(pos);
        else if (bitsPerSample === 64) s = soundData.readDoubleBE(pos);
      } else {
        // PCM big-endian
        if (bitsPerSample === 8) s = (soundData.readInt8(pos)) / 128;
        else if (bitsPerSample === 16) s = soundData.readInt16BE(pos) / 32768;
        else if (bitsPerSample === 24) {
          const raw = (soundData[pos]! << 16) | (soundData[pos + 1]! << 8) | soundData[pos + 2]!;
          s = (raw >= 0x800000 ? raw - 0x1000000 : raw) / 8388608;
        } else if (bitsPerSample === 32) s = soundData.readInt32BE(pos) / 2147483648;
      }
      sum += s;
    }
    mono[i] = sum / channels;
  }

  return { samples: mono, sampleRate, channels, numSamples: numFrames };
}

// ─── Encode WAV (always RIFF float32, little-endian) ────────────────────────

export function encodeWav(samples: Float64Array, sampleRate: number, ampWeight = 1.0): Buffer {
  const bitsPerSample = 32;
  const numChannels = 1;
  const byteRate = (sampleRate * numChannels * bitsPerSample) / 8;
  const blockAlign = (numChannels * bitsPerSample) / 8;
  const dataSize = samples.length * blockAlign;
  const buf = Buffer.allocUnsafe(44 + dataSize);

  buf.write("RIFF", 0, "ascii");
  buf.writeUInt32LE(36 + dataSize, 4);
  buf.write("WAVE", 8, "ascii");
  buf.write("fmt ", 12, "ascii");
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(3, 20); // IEEE float
  buf.writeUInt16LE(numChannels, 22);
  buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(byteRate, 28);
  buf.writeUInt16LE(blockAlign, 32);
  buf.writeUInt16LE(bitsPerSample, 34);
  buf.write("data", 36, "ascii");
  buf.writeUInt32LE(dataSize, 40);

  let peak = 0;
  for (let i = 0; i < samples.length; i++) {
    const v = Math.abs(samples[i]!);
    if (v > peak) peak = v;
  }
  const gain = (peak > 1e-9 ? 0.99 / peak : 1) * Math.max(0.01, Math.min(1, ampWeight));

  for (let i = 0; i < samples.length; i++) {
    buf.writeFloatLE(samples[i]! * gain, 44 + i * 4);
  }

  return buf;
}
