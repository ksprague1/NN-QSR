from ModelBuilder import *
from ModelLoader import *


INFO="""Trains a new network by running inference on a trained network and minimizing KL divergence
    
    A cmd call should look like this
    
    >>> python Transfer.py <Model Directory> --<param1> <name11>=<value11> <name12>=<value12> --<param2> <name21>=<value21> . . .
    
    Ex: A Patched Transformer with 2x2 patches, system total size of 8x8 learning off of a trained RNN:
    
    >>> python Transfer.py DEMO\\RNN --train steps=1000 L=64 K=1024 --ptf _2D=True patch=2
    
"""


def transfer(teacher,student,optim,op,printf=False,mydir=None):
    try:
        debug=[]
        #Set up save location
        if mydir==None:
            mydir = setup_dir(op)
            
        op=op["TRAIN"]
        t=time.time()
        for x in range(op.steps):

            
            student.zero_grad()
            loss=0
            mlogp=0
            #allow for a larger batch size by adding gradients. . .
            for k in range(op.Q):
                with torch.no_grad():
                    sample,logp = teacher.sample(op.K,op.L)

                logq=student.logprobability(sample)
                #minimize KL divergence
                # D_KL(P||Q) = E_p[log(P)-log(Q)]  (E_p[log(P)] should be constant so we can remove that term)

                loss_i = -logq.mean()/op.Q

                #update weights of student
                loss_i.backward()
                
                loss+=loss_i
                mlogp+=logp.mean()/op.Q
                
            optimizer.step()

            with torch.no_grad():
                KL = mlogp+loss
            
            #repeat values for legacy compatability
            debug+=[[0,torch.log(KL).item(),0,KL.item(),loss.item(),KL.item(),0,time.time()-t]]
            #print(torch.log(KL).item(),KL)
            if x%500==0:
                print(int(time.time()-t),end=",%.2f|"%(torch.log(KL).item()))
                if x%4000==0:print()
                if printf:sys.stdout.flush()
    except KeyboardInterrupt:pass
    
        
    import os
    DEBUG = np.array(debug)
    
    if op.dir!="<NONE>":
        np.save(mydir+"/DEBUG",DEBUG)
        student.save(mydir+"/T")

    return DEBUG
    
    

if __name__=="__main__":        
  import sys

  if "--help" in sys.argv:
    print(INFO)
  else:    
    print(sys.argv[1:])
    #load teacher model
    filename = sys.argv[1]
    teacher,t_opt = load_model(filename)
    
    #load student model and optimizer
    student,s_opt,train_opt = build_model(sys.argv[2:])
    
    train_opt["HAMILTONIAN"] = Options(**t_opt.hamiltonian)
    
    s_opt.hamiltonian = t_opt.hamiltonian
    
    train_opt["TRAIN"].dir="TRANSFER"
    
    beta1=0.9;beta2=0.999
    optimizer = torch.optim.Adam(
    student.parameters(), 
    lr=train_opt["TRAIN"].lr, 
    betas=(beta1,beta2)
    )
    
    
    
    print(train_opt)
    #setup output directory
    mydir=setup_dir(train_opt)
    
    #add extra info to settings
    train_opt["TRAIN"].source_model=sys.argv[1]
    s_opt.train=train_opt["TRAIN"].__dict__
    
    print(s_opt)
    print(t_opt)
    
    orig_stdout = sys.stdout
    
    s_opt.save(mydir+"\\settings.json")
    
    f = open(mydir+'\\output.txt', 'w')
    sys.stdout = f
    try:
        transfer(teacher,student,optimizer,train_opt,printf=True,mydir=mydir)
    except Exception as e:
        print(e)
        sys.stdout = orig_stdout
        f.close()
        1/0
    sys.stdout = orig_stdout
    f.close()