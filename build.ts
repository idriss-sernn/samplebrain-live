import * as esbuild from "esbuild";
import * as fs from "node:fs";

const manifest = JSON.parse(fs.readFileSync("manifest.json", "utf8"));
const production = process.argv.includes("--production");

const commonOptions = {
  bundle: true,
  format: "cjs" as const,
  platform: "node" as const,
  sourcesContent: false,
  logLevel: "info" as const,
  minify: production,
  sourcemap: !production,
  loader: { ".html": "text" as const },
};

await esbuild.build({
  entryPoints: ["src/extension.ts"],
  outfile: manifest.entry,
  ...commonOptions,
});
