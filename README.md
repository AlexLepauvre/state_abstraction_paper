#  Task-structured preferences guide forward planning in sequential decision-making paper repository

[![DOI](https://img.shields.io/badge/DOI-10.xxxx%2Fxxxxx-blue)](https://doi.org/10.xxxx/xxxxx)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Quarto](https://img.shields.io/badge/Built%20with-Quarto-4D4D4D?logo=quarto)](https://quarto.org)


**Authors:** Alex Lepauvre¹, Florian Ott, Stefan Kiebel¹,²

¹ Department of Psychology, Technische Universität Dresden, Dresden, Germany
² Centre for Tactile Internet with Human-in-the-Loop, TU Dresden, Germany

---

## About

This repository contains the complete source code, computational notebooks, and data
pipeline used to generate the results and figures for the paper:

> **Task-structured preferences guide forward planning in sequential decision-making paper repository**
> Alex Lepauvre, Florian Ott, Stefan Kiebel
> BioRxiv, 2026. DOI: [10.xxxx/xxxxx](https://doi.org/10.xxxx/xxxxx)

It is based on [Quarto Manuscripts: Jupyter Lab](https://quarto.org/docs/manuscripts/authoring/jupyterlab.html)

## Links

- **Paper (HTML):** https://alexlepauvre.github.io/state_abstraction_paper/
- **Preprint:** [arXiv / bioRxiv / OSF link](https://...)

## Repository Structure

```
.
├── index.qmd/                                    # Main manuscript file
├── S1-supplementary_figures_tables.ipynb/        # Supplementary material 1
├── S2-choice_model_comparison.ipynb/             # Supplementary material 2
├── S3-parameters_recovery.ipynb/                 # Supplementary material 3
└── requirements.txt                              # Environment definition
```

## Reviewing manuscript and code
If you are intersted in viewing the paper and the underlying code, head to https://alexlepauvre.github.io/state_abstraction_paper/. To see the code underlying all figures and results presented in the paper, click on the Article notebook tab on the left hand menu. 

### Replicating results

If you want to run the code to replicate our findings, you can install and run the code as follows:

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
# Set up the environment, e.g.:
pip install virtualenv
virtualenv task_structure_repo
source task_structure_repo/bin/activate
pip install -r requirements.txt
```

You can then run the notebook `index.ipynb`, which contains all the main results and figures. You can also execute all the supplementary notebooks. 


This is a template repo for generating a manuscript from Quarto that accompanies the tutorial at: 

