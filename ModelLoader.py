from RNN_QSR import *
from LPTF import *


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
    N = op.L   # Total number of atoms
    V = 7.0     # Strength of Van der Waals interaction
    Omega = 1.0 # Rabi frequency
    delta = 1.0 # Detuning 

    if op.hamiltonian=="Rydberg":
        Lx=Ly=int(op.L**0.5)
        op.L=Lx*Ly
        h = Rydberg(Lx,Ly,V,Omega,delta)
    else:
        #hope for the best here since there aren't defaults
        h = TFIM(op.L,op.h,op.J)
    return h

MODELS = {"LPTF":LPTF,"PTF":PTF,"PRNN":PRNN}

def load_model(filename,train_opt=None):
    
    
    op=Options()
    
    op.load(filename+"/settings.json")
    
    if train_opt is None:
        train_opt = Options(**op.train)
    op.train = train_opt.__dict__
    
    if op.model["model_name"]=="LPTF":
        subsampler = MODELS[op.submodel["model_name"]](**op.submodel)
        net = LPTF(subsampler,train_opt.L,**op.model)
    else:
        net = MODELS[op.model["model_name"]](train_opt.L,**op.model)
    
    tmp = torch.load(filename+"/T")
    momentum_update(0,net,tmp)
    return net,op
    
    
    
if __name__=="__main__":        
    import sys
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
    
    
    h=make_h(train_opt)
    E_queue = torch.zeros([train_opt.B],device=device)
    
    fill_queue(net,h,E_queue,train_opt)
    
    var,mean = torch.var_mean(E_queue/train_opt.L)
    stdv=((var/train_opt.B)**0.5).item()
    errformat(mean.item(),stdv)
    with open(filename+"/RESULT-%d.txt"%train_opt.B,"w") as f:
        f.write("%f +/- %f\n"%(mean.item(),stdv))
    