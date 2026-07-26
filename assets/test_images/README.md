# Built-in image codec test asset

`eia_resolution_chart_256.png` is a 256×256 rasterized copy of the
[EIA Resolution Chart 1956](https://commons.wikimedia.org/wiki/File:EIA_Resolution_Chart_1956.svg).

- Author/source: self-painted Wikimedia Commons upload by BPK
- License: public domain, worldwide dedication
- Retrieved: 2026-07-26
- Original: 990×765 SVG
- Local transformation: Wikimedia 330 px PNG preview resized to 256×256 RGBA
- SHA-256: `85f9d416b9ada6f65cf8fe0c11e234cd03b7782e734319a10b860b5099ea81e0`

This image is held out from the procedural training set and is used only for
post-training reconstruction evaluation.

## Current test card

`kakeya_codec_card_v2_source.png` was generated for this project with OpenAI's
built-in image-generation tool on 2026-07-26. It deliberately combines a
photographic street scene with exact Chinese/English text, digits, color
patches, grayscale ramps, grids, fine lines, circles, and checkerboards.

`kakeya_codec_card_v2_256.png` is the 256×256 RGB calibration asset used by
the API. It deliberately occupies one quarter of `ProceduralDocumentDataset`
so the experiment measures codec capacity and fidelity. It must not be
reported as a held-out generalization result.

- Generation prompt: scientific imaging calibration card; exact text
  `KAKEYA 256`, `图文压缩测试`, `ABC abc 0123456789`, `细节 DETAIL`,
  `R G B`, and `1px 2px 4px`
- Local transformation: generated 1024×1024 PNG resized to 256×256 RGB
- SHA-256: `5a47d4985a7aa308c3a7b93c8aa69f81633ba01f237ab6fdc8f8630c00fb1860`
