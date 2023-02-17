# NN-QSR

## Requirements
A suitable [conda](https://conda.io/) environment named `qsr` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate qsr
```

TODO: Make an environment.yaml file


## Model builder

### TRAINING

This script is used to train new models from scratch. This is an example of a command
to train an 8x8 Rydberg lattice with a patched transformer:

```
python ModelBuilder.py L=64 NLOOPS=16 sgrad=False steps=4000 --ptf _2D=True patch=2
```

Training parameters are shown when running:

```
python ModelBuilder.py --help --training
```

```
Training Arguments:

    L          (int)     -- Total lattice size (8x8 would be L=64)

    Q          (int)     -- Number of minibatches per batch

    K          (int)     -- size of each minibatch

    B          (int)     -- Total batch size (should be Q*K)

    NLOOPS     (int)     -- Number of loops within the off_diag_labels function. Higher values save ram and
                            generally makes the code run faster (up to 2x). Note, you can only set this
                            as high as your effective sequence length. (Take L and divide by your patch size).

    steps      (int)     -- Number of training steps

    dir        (str)     -- Output directory, set to <NONE> for no output

    lr         (float)   -- Learning rate

    kl         (float)   -- loss term for kl divergence

    sgrad      (bool)    -- whether or not to sample with gradients.
                            (Uses less ram when but slightly slower)

    hamiltonian (str)    -- which hamiltonian to use
```

### RNN

All optional rnn parameters can be viewed by running 

```
python ModelBuilder.py --help --rnn
```

These are the RNN parameters:

```
RNN Optional arguments:

    L          (int)     -- The total number of atoms in your lattice

    Nh         (int)     -- RNN hidden size

    patch      (int)     -- Number of atoms input/predicted at once (patch size).
                            The Input sequence will have an effective length of L/patch

    _2D        (bool)    -- Whether or not to make patches 2D (Ex patch=2 and _2D=True give shape 2x2 patches)

    rnntype    (string)  -- Which type of RNN cell to use. Only ELMAN and GRU are valid options at the moment

```

### Patched Transformer (PTF)



All optional ptf parameters can be viewed by running 

```
python ModelBuilder.py --help --ptf
```

These are your PTF parameters:

```
PTF Optional arguments:

    L          (int)     -- The total number of atoms in your lattice

    Nh         (int)     -- Transformer token size. Input patches are projected to match the token size.

    patch      (int)     -- Number of atoms input/predicted at once (patch size).
                            The Input sequence will have an effective length of L/patch

    _2D        (bool)    -- Whether or not to make patches 2D (Ex patch=2 and _2D=True give shape 2x2 patches)

    dropout    (float)   -- The amount of dropout to use in the transformer layers

    num_layers (int)     -- The number of transformer layers to use

    nheads     (int)     -- The number of heads to use in Multi-headed Self-Attention. This should divide Nh

```


### Large-Patched Transformer (LPTF)


All optional LPTF parameters can be viewed by running 

```
python ModelBuilder.py --help --lptf
```

These are your LPTF parameters:

```
LPTF Optional arguments:

    L          (int)     -- The total number of atoms in your lattice.

    Nh         (int)     -- Transformer token size. Input patches are projected to match the token size.
                            Note: When using an RNN subsampler this Nh MUST match the rnn's Nh

    patch      (int)     -- Number of atoms input/predicted at once (patch size).
                            The Input sequence will have an effective length of L/patch

    _2D        (bool)    -- Whether or not to make patches 2D (Ex patch=2 and _2D=True give shape 2x2 patches)

    dropout    (float)   -- The amount of dropout to use in the transformer layers

    num_layers (int)     -- The number of transformer layers to use

    nheads     (int)     -- The number of heads to use in Multi-headed Self-Attention. This should divide Nh

    subsampler (Sampler) -- The inner model to use for probability factorization. This is set implicitly
                            by including --rnn or --ptf arguments.

```
