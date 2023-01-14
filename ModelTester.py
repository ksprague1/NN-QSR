from RNN_QSR import *
from PTFRNN import PTFRNN
from Patched_TF import PatchTransformerB as PTF
from Patched_TF import PatchedRNN as PRNN
from TF import FastTransformer as TF
from TF import PE2D

PatchTransformerB=PTF
FastTransformer=TF
PatchedRNN=PRNN

def errformat(m,s):
    exp = -int(np.floor(np.log(s)/np.log(10)))
    print( str(round(m,exp))+" +/- "+str(round(s,exp)))

def fill_queue(net,h,E_queue,op):
    for i in range(op.Q):
        print("|",end="")
        sample,sump,sqrtp = net.sample_with_labelsALT(op.K,op.L,grad=False,nloops=op.NLOOPS)
        with torch.no_grad():
            lp=net.logprobability(sample)
                
        with torch.no_grad():
            E_i=h.localenergyALT(sample,lp,sump,sqrtp)
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

if __name__=="__main__":        
    import sys
    print(sys.argv[1:])
    op=Opt()
    filename = sys.argv[1]
    op.from_file(filename+"/settings.txt")
    op.B=op.K*op.Q
    print(op)
    
    models = {"PTFRNN":PTFRNN,"PTF":PTF,"TF":TF,"PRNN":PRNN,"RNN":FASTGRU}
    Lx=int(op.L**0.5)
    params = {"PTFRNN":{"Lx":Lx,"Nh":op.Nh},
              "PTF":{"Lx":Lx,"Nh":op.Nh,"num_layers":2},
              "TF":{"Lx":Lx,"Ly":Lx,"Nh":op.Nh,"num_layers":2},
              "PRNN":{"Lx":Lx,"Nh":op.Nh},
              "RNN":{"rnntype":"GRU","Nh":op.Nh}}
    
    op.B = int(sys.argv[2])
    op.K = int(sys.argv[3]) if len(sys.argv)>3 else op.K
    op.Q = op.B//op.K
    op.B=op.K*op.Q
    
    print(op.K,op.Q,op.B)
    
    net = models[op.dir](**params[op.dir])
    tmp = torch.load(filename+"/T")

    momentum_update(0,net,tmp)
    
    h=make_h(op)
    E_queue = torch.zeros([op.B],device=device)
    
    fill_queue(net,h,E_queue,op)
    
    var,mean = torch.var_mean(E_queue/op.L)
    stdv=((var/op.B)**0.5).item()
    errformat(mean.item(),stdv)
    with open(filename+"/RESULT-%d.txt"%op.B,"w") as f:
        f.write("%f +/- %f\n"%(mean.item(),stdv))
    