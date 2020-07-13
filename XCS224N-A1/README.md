- Install virtualenv
```
sudo pip install virtualenv 
```

- Create and activate virtual environment (using virtualenv)
```
virtualenv venv -p python3.5
source venv/bin/activate
```

- Install all the requirements (only if you're using virtualenv)
```
pip install -r requirements.txt
```

- Create and activate virtual environment (using conda)
```
conda env create --file local_env.yml
conda activate A1
conda deactivate
```

- Perform sanity checks on `computeCoOccurrenceMatrix`, `distinctWords` and `reduceToKDim`
```
python co_occurence.py
```

- Compute `co-occurrence`, run `SVD` and create `co_occurence_embeddings.png`
```
python run.py
```

- Assignment submission
```
# zip the assignment submission folder
sh collect_submission.sh
```
**In case you have trouble with the provided `collect_submission.sh` script, submit the following files:**
```
co_occurrence_embeddings.png
co_occurrence.py
```

- Miscellaneous

If this error occurs:
```
ImportError: No named '_tkinter', please install the python3-tk package
```
Install `tkinter`
```
sudo apt-get install python3-tk
```
