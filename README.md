# Generative Recommender Systems

Implementation of [Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065) (NeurIPS 2023). The method is referred to as **TIGER** (Transformer Index for GEnerative Recommenders).

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Download the dataset

This project uses the [Amazon Product Reviews 2014](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) dataset (same as the TIGER paper). Download the **5-core** review data and metadata for your category of choice.

For example, for Toys and Games:

1. Download `reviews_Toys_and_Games_5.json.gz` (5-core reviews) and `meta_Toys_and_Games.json.gz` (metadata)
2. Place them in `data/2014/` and decompress:

```bash
cd data/2014
gunzip reviews_Toys_and_Games_5.json.gz
gunzip meta_Toys_and_Games.json.gz
```

Your `data/2014/` directory should look like:

```
data/2014/
├── reviews_Toys_and_Games_5.json
└── meta_Toys_and_Games.json
```

For other categories (Beauty, Sports and Outdoors), find the corresponding files on the [dataset page](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) and follow the same pattern.


## Citations

```bibtex
@article{rajput2023recommender,
  title={Recommender systems with generative retrieval},
  author={Rajput, Shashank and Mehta, Nikhil and Singh, Anima and Hulikal Keshavan, Raghunandan and Vu, Trung and Heldt, Lukasz and Hong, Lichan and Tay, Yi and Tran, Vinh and Samost, Jonah and others},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={10299--10315},
  year={2023}
}
```

```bibtex
@inproceedings{he2016ups,
  title={Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering},
  author={He, Ruining and McAuley, Julian},
  booktitle={proceedings of the 25th international conference on world wide web},
  pages={507--517},
  year={2016}
}
```

```bibtex
@inproceedings{mcauley2015image,
  title={Image-based recommendations on styles and substitutes},
  author={McAuley, Julian and Targett, Christopher and Shi, Qinfeng and Van Den Hengel, Anton},
  booktitle={Proceedings of the 38th international ACM SIGIR conference on research and development in information retrieval},
  pages={43--52},
  year={2015}
}
```