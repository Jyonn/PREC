# PREC: Boosting Deep CTR Prediction with a Plug-and-Play Pre-trainer for News Recommendation

COLING 2022 Oral Paper

```
@inproceedings{liu2022boosting,
  title={Boosting Deep CTR Prediction with a Plug-and-Play Pre-trainer for News Recommendation},
  author={Liu, Qijiong and Zhu, Jieming and Dai, Quanyu and Wu, Xiaoming},
  booktitle = {Proceedings of the 28th International Conference on Computational Linguistics},
  year={2022}
}
```

## Tokenization

We use open-source tools [UniTok](https://github.com/Jyonn/UnifiedTokenizer) for tokenization.

```bash
python build_dataset.py
```

## Pre-train news pre-trainer

```bash
python worker.py --config config/MINDsmall-3L12H768D.yaml --exp exp/mind-mlm-news.yaml
```

## Export news representation

```bash
python worker.py --config config/MINDsmall-3L12H768D.yaml --exp exp/export-news.yaml
```

## Pre-train user pre-trainer

```bash
python worker.py --config config/MINDsmall-user-6L12H768D.yaml --exp exp/mind-mlm-user.yaml
```

## Export user representation

```bash
python worker.py --config config/MINDsmall-user-6L12H768D.yaml --exp exp/export-user.yaml
```

## Downstream models

We apply open-source benchmark tool [FuxiCTR](https://github.com/xue-pai/FuxiCTR) for CTR prediction
