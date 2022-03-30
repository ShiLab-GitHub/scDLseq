# Requirements
* numpy                     1.19.2
* pandas                    1.3.4
* pytorch                   1.10.0
* scanpy                    1.8.2
* torchtext                 0.11.0
* tqdm                      4.62.3

# Set up
After unzipping, you will get the following four python script files: pre-procesing.py, model.py, trainer,py and annotation.py

# preprocessing
## pre-procese all datasets involved to .txt file that can be processed by torchtext


```python
python pre-processing.py --name data_name --annotation annotation_filename --indir raw_data_dir --outdir output_dir
```

# training


```python
python trainer.py --traindir training_set_dir --devdir validation_set_dir --modeldir model_dir --name data_name
```

# annotatinon


```python
python annotation.py --refdir ref_set_dir --testdir test_set_dir --outdir out_put_dir --modeldir model_dir --name data_name
```


```python

```
