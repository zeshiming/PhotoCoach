# Evaluation image sources

The benchmark images are downloaded copies of Wikimedia Commons media that are marked CC0 / public domain on their source pages.

- `outdoors-man-portrait.jpg` — [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Outdoors-man-portrait.jpg), author Abhi Puthenpurackal, CC0.
- `wild-forest.jpg` — [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Landscape_of_a_wild_forest.jpg), CC0.
- `lush-landscape.jpg` — [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:A_lush_verdant_landscape.jpg), CC0.

The benchmark uses these images as evaluation inputs. The image files are not user data and are kept under `eval/assets/`.

The benchmark references the normalized copies under `eval/assets/normalized/` (RGB JPEG, max dimension 1600) so the same image encoding works across vision providers. The normalized files are derived from the CC0 originals above without adding new content.
