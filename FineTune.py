from ModelLoader import *


if __name__=="__main__":        
    import sys
    print(sys.argv[1:])
    
    train_opt=TrainOpt(K=256,Q=1,dir="FINE-TUNE")
    train_opt.apply(sys.argv[2:])
    
    filename = sys.argv[1]
    model,full_opt = load_model(filename,train_opt)
    
    
    model = torch.jit.script(model)
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