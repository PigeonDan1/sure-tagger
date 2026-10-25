# SURE-Tagger interactive demo

The page ships with five recorded Tag-Bench clips and five controlled [LayeredSpeech-500 scenes](../../assets/readme/synthetic-samples/README.md). It plays the audio, shows SURE-Tagger fields and descriptions, compares three model outputs, and filters clips by combined tag conditions. Synthetic clips also link to their renderer ground truth and field-level review. These examples are not an ASR performance report.

## Run locally

From the repository root:

```bash
python3 sure-tagger-main/demo/serve.py
```

Open <http://127.0.0.1:8765/sure-tagger-main/demo/>. The page defaults to English; use the `中文` switch for Chinese. GitHub's file viewer shows HTML source, so use the local server or a static host to run the page.

## Use your own tags

The existing [`build_demo_data.py`](../scripts/build_demo_data.py) can make `demo/data.js` from your manifest, tags, and copied audio. Open the page with `?data=local` to load that file instead of the bundled examples. The generated file is ignored by Git.

To refresh the bundled examples after changing the committed recorded or synthetic sample JSON, run `node sure-tagger-main/demo/build-fixture-data.mjs` from the repository root and commit the resulting `fixture-data.js`.

## GitHub Pages preparation

`node sure-tagger-main/demo/build-pages.mjs` creates an isolated `.pages-dist/` preview with the demo, its icon, ten bundled samples, and their source/attribution notes. The workflow in `.github/workflows/deploy-demo-pages.yml` runs only when manually dispatched after a repository administrator enables **Settings → Pages → GitHub Actions**. The resulting project site would be `https://pigeondan1.github.io/sure-tagger/`.

The demo is currently mirrored at <https://me-chuan.github.io/sure-tagger/>. The bundled audio clips, tags, and synthetic review records are public there. This repository's own Pages workflow remains manual.
