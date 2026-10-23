import { cp, mkdir, rm, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const siteRoot = join(repositoryRoot, ".pages-dist");
const files = [
  "sure-tagger-main/demo/index.html",
  "sure-tagger-main/demo/styles.css",
  "sure-tagger-main/demo/app.js",
  "sure-tagger-main/demo/fixture-data.js",
  "sure-tagger-main/demo/README.md",
  "assets/readme/three-layer-audio-metadata-tag.png",
  "assets/readme/tagbench/README.md",
  "assets/readme/tagbench/ATTRIBUTION.md",
];
const samples = [
  "ES2012d_utterance_00018",
  "EN2001e_utterance_00071",
  "fleurs_t2_0347",
  "t3_desed_0112",
  "t3_fsd50k_0045",
];

for (const sample of samples) {
  for (const file of ["audio.wav", "full_predictions.json", "reference.json", "demo_questions.json"]) {
    files.push(`assets/readme/tagbench/samples/${sample}/${file}`);
  }
}

await rm(siteRoot, { recursive: true, force: true });
for (const file of files) {
  const target = join(siteRoot, file);
  await mkdir(dirname(target), { recursive: true });
  await cp(join(repositoryRoot, file), target);
}

await writeFile(join(siteRoot, ".nojekyll"), "");
await writeFile(
  join(siteRoot, "index.html"),
  `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta http-equiv="refresh" content="0; url=./sure-tagger-main/demo/" />
    <title>SURE-Tagger · Interactive demo</title>
  </head>
  <body>
    <p><a href="./sure-tagger-main/demo/">Open the SURE-Tagger interactive demo</a></p>
  </body>
</html>
`,
);

console.log(`GitHub Pages files assembled in ${siteRoot}`);
