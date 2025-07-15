TMEL‑Mamba

Theory‑Guided Linear‑Time State‑Space Model for Cross‑Cultural Student‑Performance Prediction

  

TL;DR — TMEL‑Mamba integrates Multiple‑Intelligence, Social‑Cognitive and Learning‑Analytics theories into a linear‑time Mamba state‑space network, achieving state‑of‑the‑art accuracy (RMSE 1.86, R² 0.748) on secondary‑school data while retaining interpretability and 38 % lower GPU memory than Transformers.

✨ Key Features

Module

Purpose

Theory Anchors

Dual‑Channel Encoder

Separates cognitive vs. behavioural streams

Multiple‑Intelligence

Triadic Interaction Selector

Models learner ↔ peer ↔ environment dynamics

Social‑Cognitive

Multi‑Scale SSM Blocks

Capture short‑/long‑term temporal patterns

Learning‑Analytics

COET Regularisers

Enforce theory‑consistent constraints

All three

Linear complexity (O(L)) inference, plug‑and‑play with any tabular/sequence edu data.

🔬 Theoretical Background

TMEL‑Mamba instantiates the COET framework, mapping constructs from:

Multiple‑Intelligence Theory → cognitive channel features

Social‑Cognitive Theory → interaction selector with self‑/social‑efficacy scores

Learning‑Analytics Principles → temporal granularity & early‑warning labels

See the paper for formal definitions and ablation results.

<img width="415" height="277" alt="image" src="https://github.com/user-attachments/assets/5bcc49ec-5b24-4e07-92af-27ace70d8169" />


📦 Installation

# 1. Clone repo
$ git clone https://github.com/your‑username/tmel‑mamba.git
$ cd tmel‑mamba

# 2. Create environment (conda or venv)
$ conda create -n mamba‑edu python=3.9 -y
$ conda activate mamba‑edu

# 3. Install dependencies
$ pip install -r requirements.txt

pip install tmel‑mamba  # coming soon to PyPI

🏁 Quick Start

# Train on built‑in secondary‑school dataset
python run.py \
  --config configs/ss_highschool.yaml \
  --gpus 1 \
  --seed 42

# Evaluate a trained checkpoint
python eval.py \
  --ckpt outputs/ss_highschool/best.ckpt \
  --split test

# Predict a single student sequence
python predict.py --input_path data/demo/student123.csv

Pre‑trained checkpoints and logs are available under Releases.

📊 Datasets

Dataset

Domain

#Students

Horizon

License

SS‑HighSchool

Secondary (East Asia)

6 608

6 semesters

CC‑BY‑4.0

Uni‑MiddleEast

University (GCC)

2 480

4 years

CC‑BY‑NC‑SA 4.0

Detailed preprocessing scripts are in data_prep/ with step‑by‑step notebooks.

📈 Reproducing Experiments

bash scripts/run_all.sh   # trains 8 baselines + TMEL‑Mamba, logs to wandb

Model

RMSE ↓

R² ↑

Mem (GB) ↓

LightGBM

2.14

0.621

0.9

Transformer

1.92

0.705

5.4

TMEL‑Mamba

1.86

0.748

3.3

All metrics match Table 3 in the paper; seed = 42 for parity.

🔍 Interpretability

Run the built‑in Streamlit dashboard to explore attention heatmaps vs. pedagogical constructs:

streamlit run apps/explainability.py --server.port 8501

📝 Citation

If you find this repo useful, please cite our work:

@article{zhou2025tmelmamba,
  title={TMEL‑Mamba: A Theory‑Guided Linear‑Time State‑Space Model for Cross‑Cultural Student Performance Prediction},
  author={Zhou, Mingyu and Lee, Hye‑jin and Kumar, Rajeev},
  journal={Computers & Education: Artificial Intelligence},
  year={2025},
  note={Under Review},
  eprint={2405.12345},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

🤝 Contributing

Fork this repo & create your feature branch (git checkout -b feat/my‑feature)

Commit changes with conventional commits (feat: …, fix: …)

Push to your branch and open a Pull Request.

All contributions—new datasets, training tricks, docs—are welcome!

📄 License

This project is licensed under the MIT License – see the LICENSE file for details.

🙏 Acknowledgements

This project builds upon the Mamba state‑space library by Dao et al. (2024) and draws theoretical insight from educational psychology research by Gardner, Bandura and Siemens.

