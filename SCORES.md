# Scores

<!-- prettier-ignore-start -->

| Paper | Accuracy (Flickr)\* | Accuracy (Referit)\* | Note | 
| ----- | ------ | ------- | ---- |
| [Align2Ground: Weakly Supervised Phrase Grounding Guided by Image-Caption Alignment](https://arxiv.org/pdf/1903.11649.pdf) | 11.5 | - | 14.7 on COCO. Please note that the paper reports also 71.0% and 38.7% accuracy respectively on Flickr30k and VisualGenome, but in this case accuracy is computed with "the pointing game" metric. |
| [Phrase Localization Without Paired Training Examples](https://arxiv.org/pdf/1908.07553.pdf) | 50.49 | 26.48 | Unsupervised |
| [Weakly-supervised Visual Grounding of Phrases with Linguistic Features](https://arxiv.org/pdf/1705.01371.pdf) | - | - | 24.4 on VisualGenome, 15.9 on MS COCO |
| [Knowledge Aided Consistency for Weakly Supervised Phrase Grounding](https://arxiv.org/pdf/1803.03879.pdf) | 38.71 | 15.83 | |
| [Adaptive Reconstruction Network for Weakly Supervised Referring Expression Grounding](https://arxiv.org/pdf/1908.10568.pdf) | - | - | Results only on RefCOCO, RefCOCO+, RefCOCOg, RefCLEF |
| [Multi-level Multimodal Common Semantic Space for Image-Phrase Grounding](https://arxiv.org/pdf/1811.11683.pdf) | 69.2\*\* | 61.9\*\* | Referred as previous SOTA by Align2Ground |

<!-- prettier-ignore-end -->

\* Accuracy is reported in percentage and calculated with `IoU >= 0.5` when not
specified

\*\* Accuracy computed with pointing game metric
