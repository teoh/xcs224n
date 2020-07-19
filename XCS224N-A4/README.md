# Assignment 4 (NMT)
Note: Heavily inspired by the https://github.com/pcyin/pytorch_nmt repository

Create virtual environment
```bash
conda env create --file local_env.yml
```

Activate and deactivate environment
```bash
conda activate local_nmt
conda deactivate
```

Install necessary packages (On your VM)
```bash
pip install -r gpu_requirements.txt
```

Generate the required vocab files
```bash
sh run.sh vocab
```

Train model on your local machine
```bash
sh run.sh train_local
```

When submitting, you can run
```bash
./collect_submission.sh
```
and submit the `assignment4.zip` to gradescope or upload all of your `.py` files, the contents of your `en_es_data` folder, the contents of your `sanity_check_en_es_data` folder, and the contents of your `outputs` folder.
