# Holistic Uncertainty Estimation for Open-Set Text Classification

This repository contains the implementation of the methods described in the paper **"Uncertainty Estimation for the Open-Set Text Classification systems"**.

## Repository Setup Steps

### 1. Downloading Datasets and Model Weights

#### Raw Datasets
- **PAN Authorship Attribution**: [Download](https://drive.google.com/file/d/1F_NKLjHSpiPEviUC8B2h8zJkB9_Ftp8l/view?usp=sharing)
- **CLINC150 Intent Classification**: [GitHub Repository](https://github.com/clinc/oos-eval)
- **Topic Classification Datasets** (Yahoo Answers, AGNews, DBPedia): [Download](https://disk.360.yandex.ru/d/udOdPqMirS6Neg)

#### Processed Datasets with Protocols and Precomputed Embeddings
All five datasets with OSR protocols and precomputed embeddings for reproducibility:  
[Download](https://disk.360.yandex.ru/d/79zzA7pUn1lJYg)

- Extract the `datasets` folder and place it in the root directory of the project. The structure should look like this:
```
project_dir
│   README.md
└───datasets
│   ├───pan
│   ├───clinc150
│   ├───yahoo_answers
│   ├───agnews
│   └───dbpedia
└───model_weights  # [PLACEHOLDER] Pretrained SCF weights will be added here
│   ...
```

> **[PLACEHOLDER FOR PRETRAINED SCF WEIGHTS LINK]**  
> *Pretrained SCF model checkpoints will be available soon.*

### 2. Building the Docker Image (Optional)
- Navigate to the `docker_scripts` directory and run the build script:
```bash
cd docker_scripts
bash build.sh
```
- The image building process takes approximately 10 minutes and installs all necessary dependencies.

### 3. (Optional) Creating a Development Container
```bash
cd docker_scripts
bash launch_container.sh
```
- After running the above commands, you can connect to the container using Visual Studio Code (VSCode).

---

## Method Evaluation

To reproduce tables and figures from the paper, follow these steps:

### 1. Running Uncertainty Estimation Benchmark
Navigate to the `scripts` directory and run the evaluation script:
```bash
cd scripts
bash evaluate_text_osr.sh
```

### 2. Storing Output
- All plots and tables will be saved in the `outputs/experiments/text_osr` directory.
- For example, metrics for the PAN dataset at FPIR=0.1 will be saved in:  
  `outputs/experiments/text_osr/pan/filter_plots/0.1/`

### 3. Configuration Files
- **Training configs** (SCF and ArcFace backbones): `configs/uncertainty_models/`
  ```
  configs/uncertainty_models/
  ├─── old_cfg/
  ├─── text_model_agnews.yaml          # ArcFace baseline
  ├─── text_model_agnews_scf.yaml      # SCF probabilistic head
  ├─── text_model_clinc150.yaml
  ├─── text_model_clinc150_scf.yaml
  ├─── text_model_dbpedia.yaml
  ├─── text_model_dbpedia_scf.yaml
  ├─── text_model_pan.yaml
  ├─── text_model_pan_scf.yaml
  ├─── text_model_yahoo.yaml
  └─── text_model_yahoo_scf.yaml
  ```

- **Evaluation configs** (uncertainty benchmark): `/app/configs/uncertainty_benchmark/`
  ```
  /app/configs/uncertainty_benchmark/
  ├─── legacy/
  ├─── text_osr_agnews.yaml
  ├─── text_osr_clinc150.yaml
  ├─── text_osr_dbpedia.yaml
  ├─── text_osr_pan.yaml
  └─── text_osr_yahoo.yaml
  ```

### 4. (Optional) LaTeX Tables
Generate LaTeX-formatted tables from raw `.csv` outputs:
```bash
bash create_latex_table.sh
```

---

## Training and Inference from scratch


### 1. Training Models
Train SCF sample quality models using Hydra configs:
```bash
cd scripts
bash train_text_scf.sh -cn=text_model_pan_scf  # Example for PAN dataset
```
Available configs: `text_model_{dataset}_scf.yaml` for each of the 5 datasets.

### 2. Protocol Construction and Embedding Computation
Scripts for constructing OSR protocols and computing embeddings from trained SCF models:
```
osr_scripts/
├── construct_protocol_topic_data.py      # For Yahoo/AGNews/DBPedia
└── protocol_construction_pan_clinc150.py # For PAN and CLINC150
```


---

<!-- ## Citation
If you use this code or the HolUE method for text OSR in your research, please cite:
```bibtex
@article{erlygin2024uncertainty,
  title={Uncertainty Estimation for the Open-Set Text Classification systems},
  author={Erlygin, Leonid A. and Zaytsev, Alexey A.},
  journal={Information Processes},
  volume={24},
  number={1},
  pages={1--16},
  year={2024}
}
``` -->

---

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
The research was supported by the Russian Science Foundation grant No. 25-11-00355.