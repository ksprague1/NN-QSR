from LPTF import *

def build_model(args):
    """
    Builds a Sampler network using command line arguments
    
    A cmd call should look like this
    
    >>> python ModelBuilder.py <name>=<value> --<network> <name>=<value>
    
    Ex: A Patched Transformer with 2x2 patches, system total size of 8x8, a batch size of K*Q=1024 and 16 loops when calculating
        the off diagonal probabilities to save on memory:
    
    >>> python ModelBuilder.py L=64 NLOOPS=16 K=1024 patch=2x2 --ptf _2D=True patch=2
    
    Ex2: A Large Patched Transformer using an RNN subsampler with 3x3 patches on the LPTF and 1D patches of size 3 on the RNN
    
    >>> python ModelBuilder.py L=576 NLOOPS=64 patch=3x3B --lptf _2D=True patch=3 --rnn L=9 _2D=False patch=3 Nh=128
    
    """
    is_lptf=False
    split0=split1=len(args)
    if "--lptf" in args:
        split0=args.index("--lptf")
        is_lptf=True
        train_opt=TrainOpt(K=256,Q=1,dir="LPTF")
    if "--rnn" in args:
        split1=args.index("--rnn")
        
        SMODEL=PRNN
        train_opt=TrainOpt(K=512,Q=1,dir="RNN")
    if "--ptf" in args:
        split1=args.index("--ptf")
        SMODEL=PTF
        train_opt=TrainOpt(K=256,Q=1,dir="PTF")

        
    #Initialize default options
    sub_opt=SMODEL.DEFAULTS.copy()
    
    
    # Update options with command line arguments
    split0=min(split0,split1)
    train_opt.apply(args[:split0])
    sub_opt.apply(args[split1+1:])
    
    #Add in extra info to options
    sub_opt.model_name=SMODEL.__name__
    train_opt.B=train_opt.K*train_opt.Q
    #extra condition on the PTF to make the conditioned sampling work
    if SMODEL==PTF and is_lptf:sub_opt.Nh=[sub_opt.Nh,train_opt.Nh]
    
    # Build models
    #for the lptf we need to have a model and submodel
    if is_lptf:
        subsampler = SMODEL(**sub_opt.__dict__)
        #set lptf options
        lptf_opt=LPTF.DEFAULTS.copy()
        lptf_opt.model_name=LPTF.__name__
        lptf_opt.apply(args[split0+1:split1])
        print(args[split0+1:split1])
        #make lptf model and global settings
        model = torch.jit.script(LPTF(subsampler,train_opt.L,**lptf_opt.__dict__))
        full_opt = Options(train=train_opt.__dict__,model=lptf_opt.__dict__,submodel=sub_opt.__dict__)
    else:
        #set model to submodel and create global settings
        full_opt = Options(train=train_opt.__dict__,model=sub_opt.__dict__)
        model = torch.jit.script(SMODEL(train_opt.L,**sub_opt.__dict__))
        
    return model,full_opt,train_opt

def helper(args):
    
    while True:
        if "--lptf" in args:
            print(LPTF.INFO)
            break
        if "--rnn" in args:
            print(PRNN.INFO)
            break
        if "--ptf" in args:
            print(PTF.INFO)
            break
        args+=[input("What Model do you need help with?\nOptions are --rnn, --lptf, and --ptf:\n")]
        


if __name__=="__main__":        
    import sys
    
    
    if "--help" in sys.argv:
        print()
        helper(sys.argv)
    else:
        print(sys.argv[1:])
        
        model,full_opt,train_opt = build_model(sys.argv[1:])

        #Initialize optimizer
        beta1=0.9;beta2=0.999
        optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=train_opt.lr, 
        betas=(beta1,beta2)
        )

        print(train_opt)
        mydir=setup_dir(train_opt)
        orig_stdout = sys.stdout

        full_opt.save(mydir+"\\settings.json")

        f = open(mydir+'\\output.txt', 'w')
        sys.stdout = f
        try:
            reg_train(train_opt,(model,optimizer),printf=True,mydir=mydir)
        except Exception as e:
            print(e)
            sys.stdout = orig_stdout
            f.close()
            1/0
        sys.stdout = orig_stdout
        f.close()