# AdamW_benchmark
Benchmark for optimizers with decoupled weight decay.

The resnet model is copied over from the
[official tensorflow models](https://github.com/tensorfloAw/models/tree/master/official/resnet).

### Setup

First download and extract the CIFAR-10 data from Alex's website, specifying the location with the `--data_dir` flag. Run the following:

```
python cifar10_download_and_extract.py
```

Then to train the model, run the following:
```
python cifar10_main.py
```
Use `--data_dir` to specify the location of the CIFAR-10 data used in the previous step. There are more flag options as described in `cifar10_main.py`.
