from RNN_QSR import *
from LPTF import *



INFO="""Runs Inference on a trained network and outputs the result in a text file. It does this by generating multiple batches of samples with energy labels then averaging across all energies
    
    A cmd call should look like this
    
    >>> python Transfer.py <Model Directory> <Evaluation Samples> <Batch Size>
    
    Ex: Running inference on an RNN:
    
    >>> python ModelLoader.py DEMO\\RNN 4096 256
    
"""



def errformat(m,s):
    exp = -int(np.floor(np.log(s)/np.log(10)))
    print( str(round(m,exp))+" +/- "+str(round(s,exp)))

    
def fill_queue(net,h,E_queue,op):
    for i in range(op.Q):
        print("|",end="")
        with torch.no_grad():
            sample,logp = net.sample(op.K,op.L)
            sump,sqrtp = net.off_diag_labels_summed(sample,nloops=op.NLOOPS)
            E_i=h.localenergyALT(sample,logp,sump,sqrtp)
            E_queue[i*op.K:(i+1)*op.K]=E_i

def make_h(op):
    # Hamiltonian parameters
    
    if op.hamiltonian["name"] == "RYDBERG":
        h = Rydberg(**op.hamiltonian)
    elif op.hamiltonian["name"] =="TFIM":
        #hope for the best here since there aren't defaults
        h = TFIM(**op.hamiltonian)
    else:        
        h_opt=Rydberg.DEFAULTS.copy()
        h_opt.Lx=h_opt.Ly=int(op.train.L**0.5)
        h = Rydberg(**h_opt.__dict__)
    return h

MODELS = {"LPTF":LPTF,"PTF":PTF,"PRNN":PRNN}

def load_model(filename,train_opt=None):
    
    
    op=Options()
    
    op.load(filename+"/settings.json")
    
    if train_opt is None:
        train_opt = Options(**op.train)
    op.train = train_opt.__dict__
    
    op.model["L"]=op.train["L"]
    
    if op.model["model_name"]=="LPTF":
        subsampler = MODELS[op.submodel["model_name"]](**op.submodel)
        net = LPTF(subsampler,**op.model)
    else:
        net = MODELS[op.model["model_name"]](**op.model)
    
    tmp = torch.load(filename+"/T")
    momentum_update(0,net,tmp)
    return net,op
    
    
    
if __name__=="__main__":        
  import sys

  if "--help" in sys.argv:
    print(INFO)
  else:   
    print(sys.argv[1:])
    op=Options()
    filename = sys.argv[1]
    
    net,op = load_model(filename)
    print(op)

    train_opt = Options(**op.train)
    
    train_opt.B = int(sys.argv[2])
    train_opt.K = int(sys.argv[3]) if len(sys.argv)>3 else train_opt.K
    train_opt.Q = train_opt.B//train_opt.K
    train_opt.B=train_opt.K*train_opt.Q
    
    print(train_opt.K,train_opt.Q,train_opt.B)
    
    
    h=make_h(op)
    E_queue = torch.zeros([train_opt.B],device=device)
    
    fill_queue(net,h,E_queue,train_opt)
    
    var,mean = torch.var_mean(E_queue/train_opt.L)
    stdv=((var/train_opt.B)**0.5).item()
    errformat(mean.item(),stdv)
    with open(filename+"/RESULT-%d.txt"%train_opt.B,"w") as f:
        f.write("%f +/- %f\n"%(mean.item(),stdv))
    