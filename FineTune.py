from ModelLoader import *


if __name__=="__main__":        
    import sys
    print(sys.argv[1:])
    
    options_dict = OptionManager.parse_cmd(sys.argv[2:])

    
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