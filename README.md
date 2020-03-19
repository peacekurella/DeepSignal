# haggling
Repositiory for social signal prediction

Run the bash script in dataUtils to generate train, test data sets.
Enter Y on the prompt to enable data normalization, N to skip it
```buildoutcfg
> bash dataUtils/genDataset.sh
> Enable Normalization (Y/N) : 
```

After execution of bash script run body2body/train.py to start training the model. 
The following command to access all commandline arguments
```buildoutcfg
> python body2body/train.py --help
```