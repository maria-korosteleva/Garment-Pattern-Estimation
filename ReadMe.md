[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/neuraltailor-reconstructing-sewing-pattern/on-dataset-of-3d-garments-with-sewing)](https://paperswithcode.com/sota/on-dataset-of-3d-garments-with-sewing?p=neuraltailor-reconstructing-sewing-pattern)

# NeuralTailor: Reconstructing Sewing Pattern Structures from 3D Point Clouds of Garments

![Overview of the Neural Tailor Pipeline](img/header.png)

Official implementation of [NeuralTailor: Reconstructing Sewing Pattern Structures from 3D Point Clouds of Garments](https://arxiv.org/abs/2201.13063). Provides our pre-trained models, scripts to evalute them, and tools to train framework components from scratch.

| :zap:        Our NeuralTailor paper was accepted to SIGGRAPH 2022!   |
|----------------------------------------------------------------------|

## Dataset

For training and evaluation, NeuralTailor uses dataset created with [Garment-Pattern-Generator](https://github.com/maria-korosteleva/Garment-Pattern-Generator).
* Dataset is available from Zenodo: [Dataset of 3D Garments with Sewing Patterns](https://doi.org/10.5281/zenodo.5267549)

## Docs
Provided in `./docs` folder

1. Installtion instructions: [Installation](docs/Installation.md)
2. How to run training and evaluation: [Running](docs/Running.md)

## Citation

If you are using our system in your research, consider citing our paper.

```
@article{NeuralTailor2022,
  author = {Korosteleva, Maria and Lee, Sung-Hee},
  title = {NeuralTailor: Reconstructing Sewing Pattern Structures from 3D Point Clouds of Garments},
  year = {2022},
  issue_date = {July 2022},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  volume = {41},
  number = {4},
  doi = {10.1145/3528223.3530179},
  journal = {ACM Trans. Graph.},
  numpages = {16},
  keywords = {structured deep learning, sewing patterns, garment reconstruction}
}
```


## Contact
For bug reports, feature suggestion, and code-related questions, please [open an issue](https://github.com/maria-korosteleva/Garment-Pattern-Estimation/issues). 

For other inquires, contact the authors: 

* Maria Korosteleva ([mariako@kaist.ac.kr](mailto:mariako@kaist.ac.kr)) (Main point of contact). 

* Sung-Hee Lee ([sunghee.lee@kaist.ac.kr](mailto:sunghee.lee@kaist.ac.kr)).
