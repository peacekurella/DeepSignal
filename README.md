# haggling
Repositiory for social signal prediction

Run the bash script in dataUtils to generate train, test data sets.
Enter Y on the prompt to enable data normalization, N to skip it. Place
.pkl files in input directory at the root of the repo.
```buildoutcfg
> bash dataUtils/genDataset.sh
> Enable Normalization (Y/N) : 
```

After execution of bash script run body2body/train.py to start training the model. 
The following command to access all command line arguments
```buildoutcfg
> python seq2seq/train.py --help
```

To view the tensorboard 
```buildoutcfg
> tensorboard --logdir <path to logs>
``` 

To run the inference. The following command to access all command line arguments
```buildoutcfg
> python seq2seq/test.py --help
```