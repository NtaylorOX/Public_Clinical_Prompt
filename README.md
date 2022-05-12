![chronosig](/chronosig-logo-no-text-transparent.png)

# Repo to contain code relating to prompt based models for Mimic-III/Biomedical tasks

This repo contains code for the following work titled: "ClinicalPrompt - Application of prompt learning to clinical decision support", paper: https://arxiv.org/abs/2205.05535

We do our best to provide clear instructions for recreating the experiments here, however the code was an envolving beast and certain scripts are quite involved and you will likely want to understand which environment specific arguments are provided. Our local setup was primarily single gpu with a specific number index. This may not suit your own setup, but generally you can always use gpu number of 0 to have pytorch/cuda select the default/first available gpu in most cases

We owe a huge thanks to the open source prompting framework, OpenPrompt:https://github.com/thunlp/OpenPrompt as a starting point for anyone conducting prompt learning research. We have stored the current version of the OpenPrompt code that we used in the OpenPrompt folder, however this is an evolving codebase and thus future versions may break this repo, so be warned.


## Mimic-III ICD9 diagnosis code classification 

This is a multi-class classification problem where discharge summaries from ICU are used to classify the primary diagnosis code. Similar to the task here: https://github.com/simonlevine/clinical-longformer.

We are going to go with this task of classifying the top 50 diagnoses are a start, but will also develop a novel "triage" oriented task with the same data by grouping the ICD9 codes into clinically similar disease groups i.e. treatment pathways. 

### Data directory setup

Data cannot be stored here, so access to the raw mimic-iii data will be required.

![image](https://user-images.githubusercontent.com/49034323/151138574-05e97f18-b1c1-4a8f-808b-b8ebd0265148.png)


The raw data is contained in the following: "./data/physionet.org/files/mimiciii/1.4/zipped_data/"


### Formatting for icd9 top 50 classification

To create the training/valid/test splits for the top N icd9 diagnosis code and triage classification tasks first run the following scripts in order on the raw notes data. Perform following commands from the base dir of the repo.

#### 1.)

```
python mimic-all-tasks/preprocessing_scripts/format_notes.py
```
This will do some initial basic cleaning of the raw notes into appropriate dataframes for the different icd9 based classificaiton tasks. Compressed data will be saved alongside the original raw data as "NOTEEVENTS.FILTERED.csv.gz" by default

#### 2.)

```
python mimic-all-tasks/preprocessing_scripts/format_data_for_training.py
```
This will organise data into appropriate dataframes for the different icd9 based classificaiton tasks - either the topNicd9 classification, or triage tasks. Train/validate/test sets will be created containing all icd9_codes/data. Data will be saved at "./mimic-all-tasks/mimic3-icd9-data/intermediary_data/note2diagnosis-icd-{train/validate/test}.csv" 

#### 3a - TopNicd9 classification 

```
python mimic-all-tasks/preprocessing_scripts/format_mimic_topN_icd9.py
```
By default this will take the top 50 most frequent icd9 diagnosis codes as remove all other data (still contains the vast majority of the data) and place new train/validate/test splits inside the folder "/mimic-icd9-classification/data/intermediary_data/top_50_icd9/{train/validate/test}.csv"

#### 3b - Triage icd9 classification

```
python mimic-all-tasks/preprocessing_scripts/format_mimic_icd9_triage.py
```
This is a more experimental task where we have further split icd9 diagnosis codes into groupings that reflect their disease ontology and likely department/treatment pathways.

By default this will take the top 20 most frequent icd9 diagnosis codes and group into new triage categories, as remove all other data (still contains the vast majority of the data) and place new train/validate/test splits inside the folder "/mimic-icd9-classification/data/intermediary_data/triage/{train/validate/test}.csv"

## Mimic-III clinical outcomes tasks - Mortality prediction and Length of Stay

Follow instructions provided by: https://github.com/bvanaken/clinical-outcome-prediction 

## Continual Training of PubMedBERT on Mimic-III

To further boost the performance of ClinicalPrompt, the choice of PLM can be various. We here offer a continual training script using MIMIC-III data for domain adaption in ./mimic-pubmed-bert folder. For anyone wants to pretrain the [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract) using the MIMIC data to shift the domain towards medical texts, the script and jupyter notebooks are both offered where you can run it with minor changes to specify the data and model directory. 

We also offer **trained models** with the same strategy to save your time on huggingface hub: [512-length-version](https://huggingface.co/Tsubasaz/clinical-pubmed-bert-base-512), [128-length-version](https://huggingface.co/Tsubasaz/clinical-pubmed-bert-base-128)



# Experiments


As prompt learning allows a rather large combination of possible templates and verbalizers, we opted to select a few key combinations and run training/evaluation for one of our tasks before selecting the "optimal" combination for other experiments. This was for efficiency reasons more than anything. We chose our newly introduced ICD9 triage task for this purpose. 

We relied on bash scripts for running our experiments, and these can all be run from the ./mimic-all-tasks directory.

## Prompt learning
### Full training of different prompt combinations with frozen and finetuned PLM

#### Fine-tune PLM 
```
bash prompt-based-models/run_onetask_finetune_prompt_comparisons.sh
```
#### Frozen PLM
```
bash prompt-based-models/run_onetask_frozen_prompt_comparison.sh 
```
### Full training on optimal template_verbalizer combination for all tasks
```
bash prompt-based-models/run_prompt_all_full_experiments.sh 
```

### Fewshot training for both frozen and finetuned PLM
Next we can run the fewshot experiments on all tasks
```
bash prompt-based-models/run_all_fewshot_prompt_experiments.sh
```

### hyperparameter search
```
bash prompt-based-models/run_prompt_hp_search.sh 
```

### optimized run
```
bash prompt-based-models/run_prompt_optimized_task.sh 
```

### sensitivity analysis - number of trainable parameters vs performance
```
bash prompt-based-models/run_prompt_sensitivity_analysis.sh 
```
### ablation study
```
bash prompt-based-models/run_random_prompt_ablation.sh 
```

## Traditional fine-tuning with pytorch-lightning

### Full training 
```
bash pytorch-lightning-models/run_all_task_experiments.sh 
```

### Fewshot training for both frozen and finetuned PLM
Next we can run the fewshot experiments on all tasks
```
bash pytorch-lightning-models/run_all_fewshot_experiments.sh 
```

### hyperparameter search
```
bash pytorch-lightning-models/run_pl_hp_search.sh  
```

### optimized run
```
bash pytorch-lightning-models/run_task_optimized.sh  
```

### sensitivity analysis - number of trainable parameters vs performance
```
bash pytorch-lightning-models/run_sensitivity_analysis.sh 
```

# Tensorboard logs

## Logfile naming logic
We use a relatively crude and potentially flawed logic for naming the log files to indicate the experimental settings, although all of the parameters and hyperparamters are saved alongside the tensorboard events files anyway. We use these filenames to be able to easily split the experiments based on the pretrained language model(PLM) used, the task, whether the PLM was frozen and what the training sample size was. When parsing these log files to analyse results etc can then use regex to easily organise and compare.

**Prompt learning log example**
.\prompt-based-models\logs\icd9_triage\frozen_plm\emilyalsentzer\Bio_ClinicalBERT_tempmanual2_verbsoft0_full_100\version_22-03-2022--14-15\{tensorboard-events-file}

**Pytorch lightning log example**

.\pytorch-lightning-models\logs\icd9_triage\full_100\frozen_plm\emilyalsentzer\Bio_ClinicalBERT\version_20-03-2022--00-23-43

## Launching tensorboard to view training in real time
We utilise the tensorboard events files created during training and evaluation to conduct our results analysis. For both the prompt learning and pytorch lightning pipelines logs and model checkpoints will automatically be stored in respective ./logs folders. Presuming you have tensorboard installed, run the following from command line with the "logdir" set to the directory containing the logs you want to see.

```
tensorboard --logdir "LOG_DIR"
```

# Analysing results from tensorboard events file

This was perhaps a odd way to do it - but we can derive all results directly from the tensorboard log files and massage them anyway we please. It is very involved and an example of how we produced plots etc can be found in a notebook here: [link](./mimic-all-tasks/plot_scripts/create_plots_from_tb.ipynb)

Or you can just do whatever you want with the trained models as you normally would for inference etc.

# Setup of repo on local machine

## create virtual python environment 
This will depend on OS and how python is installed. On linux can use either conda or venv. 

## with venv

```
# You only need to run this command once per-VM
sudo apt-get install python3-venv -y

# The rest of theses steps should be run every time you create
#  a new notebook (and want a virutal environment for it)

cd the/directory/your/notebook/will/be/in

# Create the virtual environment
# The '--system-site-packages' flag allows the python packages 
#  we installed by default to remain accessible in the virtual 
#  environment.  It's best to use this flag if you're doing this
#  on AI Platform Notebooks so that you keep all the pre-baked 
#  goodness
python3 -m venv myenv --system-site-packages
source myenv/bin/activate #activate the virtual env

# Register this env with jupyter lab. It’ll now show up in the
#  launcher & kernels list once you refresh the page
python -m ipykernel install --user --name=myenv

# Any python packages you pip install now will persist only in
#  this environment_
deactivate # exit the virtual env


```

## with conda

```

conda update conda

conda create -n yourenvname python=3.9 anaconda

source activate yourenvname

```

## git clone this repo

```
git clone https://github.com/NtaylorOX/Public_Prompt_Mimic_III.git 
```

## Use pip package manager
```
pip install -r requirements.txt
```



