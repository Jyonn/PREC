# PREC: Boosting Deep CTR Prediction with a Plug-and-Play Pre-trainer for News Recommendation

COLING 2022 Oral Paper [aclanthology](https://aclanthology.org/2022.coling-1.249/)

```
@inproceedings{liu2022prec,
    title = "Boosting Deep {CTR} Prediction with a Plug-and-Play Pre-trainer for News Recommendation",
    author = "Liu, Qijiong  and
      Zhu, Jieming  and
      Dai, Quanyu  and
      Wu, Xiaoming",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.249",
    pages = "2823--2833"
}
```

## Tokenization

We use open-source tools [UniTok](https://github.com/Jyonn/UnifiedTokenizer) for tokenization.

```bash
python build_dataset.py
```

## Pre-train news pre-trainer

```bash
python worker.py --config config/MINDsmall-3L12H768D.yaml --exp exp/mind-news.yaml
```

## Export news representation

```bash
python worker.py --config config/MINDsmall-3L12H768D.yaml --exp exp/export-news.yaml
```

## Pre-train user pre-trainer

```bash
python worker.py --config config/MINDsmall-user-6L12H768D.yaml --exp exp/mind-user.yaml
```

## Export user representation

```bash
python worker.py --config config/MINDsmall-user-6L12H768D.yaml --exp exp/export-user.yaml
```

## Downstream models

We apply open-source benchmark tool [FuxiCTR](https://github.com/xue-pai/FuxiCTR) for CTR prediction
