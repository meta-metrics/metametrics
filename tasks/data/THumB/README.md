# <img src="https://github.com/jungokasai/THumB/blob/master/figs/thumb.png" height="25" alt="thumb-up"> THumB for MSCOCO Image Captioning
`mscoco_THumB-1.0.jsonl` provides THumB (**T**ransparent **Hum**an **B**enchmark) scores for image captions on the MSCOCO dataset. The evaluation protocol and rules of THumB are described in [Kasai et al., 2021](https://arxiv.org/abs/2111.08940).
| Category |  |
| ------ | ------ |
| P | Precision (1-5) |
| R | Recall (1-5) |
| Flu | Fluency (negative) |
| Con | Conciseness (negative) |
| Inc | Inclusive Language (negative) |
| human_score | (P+R)/2 + Flu + Con + Inc | 
| SYS | system (captioning model) | 
| seg_id | segmentation (example) ID | 
| set_id | integer ID in the whole test split (0-4999) | 
| hyp | hypothesis (predicted) caption from the model | 
| image | image jpg file name in MSCOCO | 


## Evaluated Models and Call for Better Model Development
We follow the spirit of [SacreBLEU](https://arxiv.org/abs/1804.08771) and applied minimal postprocessing to model outputs to get **clean, truecased, untokenized** text. See [Kasai et al., 2021](https://arxiv.org/abs/2111.08940) for more detail.

Future captioning model developers, **please please do provide clean, truecased, untokenized captions**, independent of your preprocessing for reproducible (automatic) evaluations.

500 captions are evaluated for every model.
| Model | Reference |
| ------ | ------ |
| Up-Down | [Anderson et al., 2018](https://arxiv.org/abs/1707.07998) |
| Unified-VLP | [Zhou et al., 2020](https://arxiv.org/abs/1909.11059) |
| VinVL-base | [Zhang et al., 2021](https://arxiv.org/abs/2101.00529) |
| VinVL-base | [Zhang et al., 2021](https://arxiv.org/abs/2101.00529) |
| Human | [Chen et al., 2015](https://arxiv.org/abs/1504.00325) |

## MSCOCO test image data
You can download all 5K test images [here](https://arkdata.cs.washington.edu/mscoco/karpathy-test-split-images.tar.gz). We follow the standard "Karpathy" split. `cnndm_references.jsonl` is human-generated reference captions (excluding Human) from [Chen et al., 2015](https://arxiv.org/abs/1504.00325).

## Citation
```
@inproceedings{kasai2022thumb,
    title   = {Transparent Human Evaluation for Image Captioning},
    author  = {Jungo Kasai and Keisuke Sakaguchi and Lavinia Dunagan and Jacob Morrison and Ronan Le Bras and Yejin Choi and Noah A. Smith},
    year    = {2022},
    booktitle = {Proc.\ of NAACL},
    url     = {https://arxiv.org/abs/2111.08940}, 
}
```
