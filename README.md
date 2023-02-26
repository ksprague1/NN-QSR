# NN-QSR

## Requirements
A suitable [conda](https://conda.io/) environment named `qsr` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate qsr
```

## Model builder

### TRAINING

This script is used to train new models from scratch. This is an example of a command
to train an 8x8 Rydberg lattice with a patched transformer:
```
python ModelBuilder.py --train L=64 NLOOPS=16 K=1024 sub_directory=2x2 --ptf _2D=True patch=2
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
                
        sgrad      (bool)    -- whether or not to sample with gradients. 
                                (Uses less ram when but slightly slower)
                                
        true_grad  (bool)    -- Set to false to approximate the gradients
                                
        sub_directory (str)  -- String to add to the end of the output directory (inside a subfolder)
        
    
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
        
        nhead     (int)      -- The number of heads to use in Multi-headed Self-Attention. This should divide Nh
    
        repeat_pre (bool)    -- repeat the precondition instead of projecting it out
    

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
        
        nhead     (int)     -- The number of heads to use in Multi-headed Self-Attention. This should divide Nh
        
        subsampler (Sampler) -- The inner model to use for probability factorization. This is set implicitly
                                by including --rnn or --ptf arguments.
    
    

```

## Model Loader

This script primarily contains the function load_model(filename) which can load in any trained model using the settings.json file

```
python ModelLoader.py --help
```


```
Runs Inference on a trained network and outputs the result in a text file. It does this by generating multiple batches of samples
    with energy labels then averaging across all energies
    
    A cmd call should look like this
    
    >>> python Transfer.py <Model Directory> <Evaluation Samples> <Batch Size>
    
    Ex: Running inference on an RNN:
    
    >>> python ModelLoader.py DEMO\RNN 4096 256
    

```

## Fine Tuning

Say you want to train a model on an 16x16 system then fine-tune it on a 32x32 system. The script FineTune.py does this by loading the original trained model and letting you change the hamiltonian and training settings for further training.


```
python FineTune.py --help
```


```
Loads a model and resumes training with new training options (i.e a larger system size or a slight alteration to the hamiltonian)
    
    A cmd call should look like this
    
    >>> python FineTune.py <Model Directory> --train <name>=<value> --<hamiltonian name> <name>=<value>
    
    Ex: Running inference on an RNN:
    
    >>> python FineTune.py DEMO\RNN --train L=256 NLOOPS=64 K=512 steps=4000 --rydberg Lx=16 Ly=16
    

```

## Knowledge Transfer

Say you already have a decent model trained on a large lattice size. You can train a different model with a new architecture to give a similar probability distribution by minimizing KL divergence between your (frozen) trained model and new model. (Maximize the log-likelyhood of each drawn sample). This is done in Transfer.py


```
python Transfer.py --help
```


```
Trains a new network by running inference on a trained network and minimizing KL divergence
    
    A cmd call should look like this
    
    >>> python Transfer.py <Model Directory> --<param1> <name11>=<value11> <name12>=<value12> --<param2> <name21>=<value21> . . .
    
    Ex: A Patched Transformer with 2x2 patches, system total size of 8x8 learning off of a trained RNN:
    
    >>> python Transfer.py DEMO\RNN --train steps=1000 L=64 K=1024 --ptf _2D=True patch=2
    

```
