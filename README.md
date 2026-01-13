higgs
=========

This project is a modern reimplementation of the research paper "Searching for Exotic Particles in High-Energy Physics with Deep Learning" by Baldi, Sadowski, and Whiteson. Instead of using the original Pylearn2/Theano stack, this project using PyTorch.

## Reference

* Baldi, P., Sadowski, P. & Whiteson, D. Searching for exotic particles in high-energy physics with deep learning. Nat Commun 5, 4308 (2014).
* **Datasets:** [UCI Machine Learning Repository: HIGGS Data Set](https://archive.ics.uci.edu/dataset/280/higgs)
* **Original Repository:** [uci-igb/higgs-susy](https://github.com/uci-igb/higgs-susy)

## Project Structure

```
.
├── configs/              # Contains .yaml configuration files for each experiment
├── data/
│   └── raw/              # Directory for the HIGGS.csv file
├── models/               # Trained models (.pt) are automatically saved here
├── results/
│   └── metrics/          # Evaluation results (.json) are automatically saved here
├── src/                  # Contains all Python source code
│   ├── data_loader.py    # Script to load and preprocess the data
│   └── model_builder.py  # Neural Network architecture definition
├── .gitignore            # Ignores unnecessary files from commits
├── evaluate.py           # Script to evaluate a trained model on the test set
├── requirements.txt      # List of required Python libraries
└── train.py              # Main script to train a model
```

## Usage Guide

### 1\. Environment Setup

1. **Clone the Repository** 

```bash
git clone https://github.com/velionn/higgs.git
cd higgs
```

2. **Download the Dataset**

* Download the `HIGGS.csv` dataset from [UCI Machine Learning Repository: HIGGS Data Set](https://archive.ics.uci.edu/dataset/280/higgs).
* Extract and place the `HIGGS.csv` file (approx. 8GB) into the `data/raw/`.

3.  **Create a Virtual Environment**

```bash
python3 -m venv venv
source venv/bin/activate
```

4.  **Install Libraries**

```bash
pip3 install torch torchvision
```

*Install other libraries:*

```bash
pip install -r requirements.txt
```

### 2\. Running Training

Training is executed using the `train.py` script, which requires a configuration file as an argument. All experiment configurations are defined in the `configs/` folder.

```bash
python train.py --config [PATH_TO_CONFIG_FILE]
```

The best performing model will be automatically saved in the `models/` folder.

### 3\. Running Evaluation

After a model has been trained, you can evaluate its final performance on the test set using `evaluate.py`. This script uses the same configuration file that was uset for training

```bash
python evaluate.py --config [PATH_TO_CONFIG_FILE]
```

The evaluation result (AUC, accuracy, loss) will be printed to the terminal and saved to a `.json` file in the `result/metrisc/` folder

<!-- end list -->

