- Install virtualenv
```
sudo pip install virtualenv 
```

- Create and activate virtual environment (using virtualenv)
```
virtualenv venv -p python3.5
source venv/bin/activate
```

- Install all requirements (only if you're using virtualenv)
```
pip install -r requirements.txt
```

- Create and activate virtual environment (using conda)
```
conda env create --file local_env.yml
conda activate A2
conda deactivate
```


- Train word2vec and generate files *sampleVectors.json* and *word_vectors.png*

**Note: Do not change the hyperparameter values in run.py script**  
```
python run.py
```

- Sanity check on sampleVectors.json
```
python test_sample_vectors.py
```

- Assignment submission(for students)
```
# zip the assignment submission folder
sh collect_submission.sh
```
**In case you have trouble with the provided `collect_submission.sh` script, submit the following files:**
```
run.py
sample_vectors.json
sgd.py
test_sample_vectors.py
word2vec.py
word_vectors.png
```
