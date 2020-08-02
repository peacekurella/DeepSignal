# DeepSignal
*A data driven approach to generate non verbal social signals.*

## Basic repository usage

Run preprocess.py to preprocess the .pkl files and write them into TFRecords
```buildoutcfg
> python preprocess.py
```

Training requires the data in TFRecord format. All experiment related arguments are given as FLAGS inside the
train file. 
```buildoutcfg
> python train.py
```

To view the tensorboard 
```buildoutcfg
> tensorboard --logdir <path to logs>
``` 

The following command runs the test script
```buildoutcfg
> python test.py --help
```

## Future work 
  - LSTMs instead of GRUs
  - Update data representation
  - Use non deterministic ( VAE ) models for signal generation

Detailed information about the project can be found in report.pdf
