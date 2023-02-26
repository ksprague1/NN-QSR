from ModelLoader import *

INFO="""Loads a model and resumes training with new training options (i.e a larger system size or a slight alteration to the hamiltonian)
    
    A cmd call should look like this
    
    >>> python FineTune.py <Model Directory> --train <name>=<value> --<hamiltonian name> <name>=<value>
    
    Ex: Running inference on an RNN:
    
    >>> python FineTune.py DEMO\\RNN --train L=256 NLOOPS=64 K=512 steps=4000 --rydberg Lx=16 Ly=16
    
"""

if __name__=="__main__":        
  import sys

  if "--help" in sys.argv:
    print(INFO)
  else:   
    
    print(sys.argv[1:])
    
    options_dict = OptionManager.parse_cmd(sys.argv[2:])

    if options_dict["TRAIN"].dir =="out":
        options_dict["TRAIN"].dir="FINE-TUNE"
    
    filename = sys.argv[1]
    model,full_opt = load_model(filename,options_dict["TRAIN"])
    
    
    model = torch.jit.script(model)
    #Initialize optimizer
    beta1=0.9;beta2=0.999
    optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=options_dict["TRAIN"].lr, 
    betas=(beta1,beta2)
    )
    
    name = "TFIM" if "TFIM" in options_dict else "RYDBERG"
    options_dict["HAMILTONIAN"] = options_dict[name]
    options_dict["HAMILTONIAN"].name=name
    #print(options_dict["HAMILTONIAN"])
    full_opt.hamiltonian = options_dict["HAMILTONIAN"].__dict__
    print(full_opt)
    mydir=setup_dir(options_dict)
    orig_stdout = sys.stdout
    
    
    
    full_opt.save(mydir+"\\settings.json")
    
    f = open(mydir+'\\output.txt', 'w')
    sys.stdout = f
    try:
        reg_train(options_dict,(model,optimizer),printf=True,mydir=mydir)
    except Exception as e:
        print(e)
        sys.stdout = orig_stdout
        f.close()
        1/0
    sys.stdout = orig_stdout
    f.close()