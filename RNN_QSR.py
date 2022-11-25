
# # Quantum state to represent
# 
# - Start with Rydberg system
# 
# Transverse and longitudinal view of ising model
# 
# Excited state encourages nearby (within radius $R_b$)states to tend towards ground states

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
import math,time
import torch
from torch import nn
ngpu=1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)
from numba import cuda


# # Estimating the Rydberg Hamiltonian:


# In[2]:


class Hamiltonian():
    def __init__(self,L,offDiag,device=device):
        self.offDiag  = offDiag           # Off-diagonal interaction
        self.L        = L               # Number of spins
        self.device   = device
        self.Vij      = self.Vij=nn.Linear(self.L,self.L).to(device)
        self.buildlattice()
    def buildlattice():
        """Creates the matrix representation of the on-diagonal part of the hamiltonian
            - This should fill Vij with values"""
        raise NotImplementedError
    def localenergy(self,samples,logp,logppj):
        """
        Takes in s, ln[p(s)] and ln[p(s')] (for all s'), then computes Hloc(s) for N samples s.
        
        Inputs:
            samples - [B,L,1] matrix of zeros and ones for ground/excited states
            logp - size B vector of logscale probabilities ln[p(s)]
            logppj - [B,L] matrix of logscale probabilities ln[p(s')] where s'[i][j] had one state flipped at position j
                    relative to s[i]
        Returns:
            size B vector of energies Hloc(s)
        
        """
        # Going to calculate Eloc for each sample in a separate spot
        # so eloc will have shape [B]
        # recall samples has shape [B,L,1]
        B=samples.shape[0]
        eloc = torch.zeros(B,device=self.device)
        # Chemical potential
        with torch.no_grad():
            tmp=self.Vij(samples.squeeze(2))
            eloc += torch.sum(tmp*samples.squeeze(2),axis=1)
        # Off-diagonal part
        #logppj is shape [B,L]
        #logppj[:,j] has one state flipped at position j
        for j in range(self.L):
            #make sure torch.exp is a thing
            eloc += self.offDiag * torch.exp((logppj[:,j]-logp)/2)

        return eloc
    def localenergyALT(self,samples,logp,sumsqrtp,logsqrtp):
        """
        Takes in s, ln[p(s)] and exp(-logsqrtp)*sum(sqrt[p(s')]), then computes Hloc(s) for N samples s.
        
        Inputs:
            samples  - [B,L,1] matrix of zeros and ones for ground/excited states
            logp     - size B vector of logscale probabilities ln[p(s)]
            logsqrtp - size B vector of average (log p)/2 values used for numerical stability 
                       when calculating sum_s'(sqrt[p(s')/p(s)]) 
            sumsqrtp - size B vector of exp(-logsqrtp)*sum(sqrt[p(s')]).
        Returns:
            size B vector of energies Hloc(s)
        
        """
        # Going to calculate Eloc for each sample in a separate spot
        # so eloc will have shape [B]
        # recall samples has shape [B,L,1]
        B=samples.shape[0]
        eloc = torch.zeros(B,device=self.device)
        # Chemical potential
        with torch.no_grad():
            tmp=self.Vij(samples.squeeze(2))
            eloc += torch.sum(tmp*samples.squeeze(2),axis=1)
        # Off-diagonal part
        
        #in this function the entire sum is precomputed and it was premultiplied by exp(-logsqrtp) for stability
        eloc += self.offDiag *sumsqrtp* torch.exp(logsqrtp-logp/2)

        return eloc
    def ground(self):
        """Returns the ground state energy E/L"""
        raise NotImplementedError


# In[3]:

if __name__=="__main__":
    help(Hamiltonian)


# In[4]:


@cuda.jit
def Vij(Ly,Lx,Rcutoff,V,matrix):
    #matrix will be size [Lx*Ly,Lx*Ly]
    
    i,j=cuda.grid(2)
    if i>Ly or j>Lx:
        return
    R=Rcutoff**6
    #flatten two indices into one
    idx = Ly*j+i
    # only fill in the upper diagonal
    for k in range(idx+1,Lx*Ly):
        #expand one index into two
        i2 = k%Ly
        j2=k//Ly
        div = ((i2-i)**2+(j2-j)**2)**3
        if div<=R:
            matrix[idx][k]=V/div

class Rydberg(Hamiltonian):
    E={16:-0.45776822,36:-0.4221,64:-0.40522,144:-0.38852,256:-0.38052,576:-0.3724,1024:-0.3687}
    def __init__(self,Lx,Ly,V,Omega,delta,R=2.01,device=device):
        self.Lx       = Lx              # Size along x
        self.Ly       = Ly              # Size along y
        self.V        = V               # Van der Waals potential
        self.delta    = delta           # Detuning
        self.R=R
        # off diagonal part is -0.5*Omega
        super(Rydberg,self).__init__(Lx*Ly,-0.5*Omega,device)
        
    def buildlattice(self):
        Lx,Ly=self.Lx,self.Ly
        
        #diagonal hamiltonian portion can be written as a matrix multiplication then a dot product        
        mat=np.zeros([self.L,self.L])
        
        Vij[(1,1),(Lx,Ly)](Lx,Ly,self.R,self.V,mat)
        with torch.no_grad():
            self.Vij.weight[:,:]=torch.Tensor(mat)
            self.Vij.bias.fill_(-self.delta)
    def ground(self):
        return Rydberg.E[self.Lx*self.Ly]


# In[5]:


class TFIM(Hamiltonian):
    En1={10:-1.2381,40:-1.2642,500:-1.2725,100:-1.2696 ,1000:-1.2729}
    def __init__(self,L,h_x,J=1.0,device=device):
        self.J = J
        super(TFIM,self).__init__(L,h_x,device)
        
    def buildlattice(self):
        #building hamiltonian matrix for diagonal part
        mat=np.zeros([self.L,self.L])
        for i in range(self.L - 1):
            mat[i, i+1] = -self.J

        with torch.no_grad():
            self.Vij.weight[:,:]=torch.Tensor(mat)
            self.Vij.bias.fill_(0.0) # no longitudinal field

    def localenergy(self,samples,logp,logppj):
        return super(TFIM,self).localenergyALT(2*samples-1,logp,logppj)

    def localenergyALT(self,samples,logp,sumsqrtp,logsqrtp):
        return super(TFIM,self).localenergyALT(2*samples-1,logp,sumsqrtp,logsqrtp)
    def ground(self):
        if self.J==1 and self.offDiag==-1:
            if self.L in TFIM.En1:
                return TFIM.En1[self.L]
        return -10




# In[6]:


class Sampler(nn.Module):
    def __init__(self,device=device):
        self.device=device
        super(Sampler, self).__init__()
    def logprobability(self,input):
        """Compute the logscale probability of a given state
            Inputs:
                input - [B,L,1] matrix of zeros and ones for ground/excited states
            Returns:
                logp - [B] size vector of logscale probability labels
        """
        raise NotImplementedError
    def sample(self,B,L):
        """ Generates a set states
        Inputs:
            B (int)            - The number of states to generate in parallel
            L (int)            - The length of generated vectors
        Returns:
            samples - [B,L,1] matrix of zeros and ones for ground/excited states
        """
        raise NotImplementedError
    
    @torch.jit.ignore
    def sample_with_labels(self,B,L,grad=False,nloops=1):
        """Inputs:
            B (int) - The number of states to generate in parallel
            L (int) - The length of generated vectors
            grad (boolean) - Whether or not to use gradients
        Returns:
            samples - [B,L,1] matrix of zeros and ones for ground/excited states
            logppl - [B,L] matrix of logscale probabilities ln[p(s')] where s'[i+B*j] had one spin flipped at position j
                    relative to s[i]
        """
        sample=self.sample(B,L)
        return self._off_diag_labels(sample,B,L,grad,nloops)
    
    
    @torch.jit.ignore
    def _off_diag_labels(self,sample,B,L,grad,D=1):
        """label all of the flipped states faster (no loop in rnn) but using more ram"""
        sflip = torch.zeros([B,L,L,1],device=self.device)
        #collect all of the flipped states into one array
        for j in range(L):
            #get all of the states with one spin flipped
            sflip[:,j] = sample*1.0
            sflip[:,j,j] = 1-sflip[:,j,j]
        #compute all of their logscale probabilities
        if not grad:
            with torch.no_grad():
                probs=torch.zeros([B*L],device=self.device)
                tmp=sflip.view([B*L,L,1])
                for k in range(D):
                    probs[k*B*L//D:(k+1)*B*L//D] = self.logprobability(tmp[k*B*L//D:(k+1)*B*L//D])
        else:
            probs = self.logprobability(sflip.view([B*L,L,1]))

        return sample,probs.reshape([B,L])
    
    @torch.jit.ignore
    def sample_with_labelsALT(self,B,L,grad=False,nloops=1):
        """Returns:
            samples  - [B,L,1] matrix of zeros and ones for ground/excited states
            logsqrtp - size B vector of average (log p)/2 values used for numerical stability 
                       when calculating sum_s'(sqrt[p(s')/p(s)]) 
            sumsqrtp - size B vector of exp(-logsqrtp)*sum(sqrt[p(s')]).
        """
        sample,probs = self.sample_with_labels(B,L,grad,nloops)
        #get the average of our logprobabilities and divide by 2
        logsqrtp=probs.mean(dim=1)/2
        #compute the sum with a constant multiplied to keep the sum closeish to 1
        sumsqrtp = torch.exp(probs/2-logsqrtp.unsqueeze(1)).sum(dim=1)
        return sample,sumsqrtp,logsqrtp
    
    


# In[7]:

if __name__=="__main__":
    import pydoc

    #help function prints too much stuff so I'm getting the info and removing the unimportant bits
    documentation = pydoc.text.document(Sampler, "").split("--------")[0]
    out= documentation.split("Base class for all neural network modules")[0]+documentation.split("builtins.object")[1]
    toprint,i=out[0],1
    while i<len(out):toprint+= out[i] if (out[i]!= "\x08" and out[i-1] != "\x08") else "";i+=1
    print(toprint)


# # Simple RNN to start
# 

# In[8]:


class RNN(Sampler):
    TYPES={"GRU":nn.GRU,"ELMAN":nn.RNN,"LSTM":nn.LSTM}
    def __init__(self,rnntype="GRU",Nh=128,device=device, **kwargs):
        super(RNN, self).__init__(device=device)
        #rnn takes input shape [B,L,1]
        self.rnn = RNN.TYPES[rnntype](input_size=1,hidden_size=Nh,batch_first=True)
        
        
        self.lin = nn.Sequential(
                nn.Linear(128,128),
                nn.ReLU(),
                nn.Linear(128,1),
                nn.Sigmoid()
            )
        
        self.rnntype=rnntype
        self.to(device)
    def forward(self, input):
        # h0 is shape [d*numlayers,B,H] but D=numlayers=1 so
        # h0 has shape [1,B,H]
        
        if self.rnntype=="LSTM":
            h0=[torch.zeros([1,input.shape[0],128],device=self.device),
               torch.zeros([1,input.shape[0],128],device=self.device)]
            #h0 and c0
        else:
            h0=torch.zeros([1,input.shape[0],128],device=self.device)
        out,h=self.rnn(input,h0)
        return self.lin(out)
    
    def logprobability(self,input):
        """Compute the logscale probability of a given state
            Inputs:
                input - [B,L,1] matrix of zeros and ones for ground/excited states
            Returns:
                logp - [B] size vector of logscale probability labels
        """
        
        #Input should have shape [B,L,1]
        B,L,one=input.shape
        
        #first prediction is with the zero input vector
        data=torch.zeros([B,L,one],device=self.device)
        #data is the input vector shifted one to the right, with the very first entry set to zero instead of using pbc
        data[:,1:,:]=input[:,:-1,:]
        
        #real is going to be a set of actual values
        real=input
        #and pred is going to be a set of probabilities
        #if real[i]=1 than you multiply your conditional probability by pred[i]
        #if real[i]=0 than you multiply by 1-pred[i]
        
        #probability predictions may be done WITH gradients
        #with torch.no_grad():
        
        pred = self.forward(data)
        ones = real*pred
        zeros=(1-real)*(1-pred)
        total = ones+zeros
        #this is the sum you see in the cell above
        #add 1e-10 to the prediction to avoid nans when total=0
        logp=torch.sum(torch.log(total+1e-10),dim=1).squeeze(1)
        return logp
    def sample(self,B,L):
        """ Generates a set states
        Inputs:
            B (int)            - The number of states to generate in parallel
            L (int)            - The length of generated vectors
        Returns:
            samples - [B,L,1] matrix of zeros and ones for ground/excited states
        """
        if self.rnntype=="LSTM":
            h=[torch.zeros([1,B,128],device=self.device),
               torch.zeros([1,B,128],device=self.device)]
            #h is h0 and c0
        else:
            h=torch.zeros([1,B,128],device=self.device)
        #Sample set will have shape [N,L,1]
        #need one extra zero batch at the start for first pred hence input is [N,L+1,1] 
        input = torch.zeros([B,L+1,1],device=self.device)
        #sampling can be done without gradients
        with torch.no_grad():
          for idx in range(1,L+1):
            #run the rnn on shape [B,1,1]
            
            out,h=self.rnn(input[:,idx-1:idx,:],h)
            out=out[:,0,:]
            #if probs[i]=1 then there should be a 100% chance that sample[i]=1
            #if probs[i]=0 then there should be a 0% chance that sample[i]=1
            #stands that we generate a random uniform u and take int(u<probs) as our sample
            probs=self.lin(out)
            sample = (torch.rand([B,1],device=device)<probs).to(torch.float32)
            input[:,idx,:]=sample
        #input's first entry is zero to get a predction for the first atom
        return input[:,1:,:]


# In[9]:



# In[10]:


def new_rnn_with_optim(rnntype,Nh,lr=1e-3,beta1=0.9,beta2=0.999):
    rnn = RNN(rnntype=rnntype,Nh=Nh)
    optimizer = torch.optim.Adam(
    rnn.parameters(), 
    lr=lr, 
    betas=(beta1,beta2)
    )
    return rnn,optimizer





# In[11]:


def momentum_update(m, target_network, network):
    for target_param, param in zip(target_network.parameters(), network.parameters()):
        target_param.data.copy_(target_param.data*m + param.data*(1.0-m))


# In[12]:




# # Setting Constants

# In[13]:


class Opt:
    """
    Description of these options:
    
    L (int) -- Total lattice size (np.prod(lattice.shape))
    Q  (int) -- Number of minibatches in the queue
    K  (int) -- size of each minibatch
    B  (int) -- Total batch size (should be Q*K)
    TOL (float) -- Tolerance to decide if train energies are too good to be true
    M (float)   -- Momentum used for sample RNN update
    USEQUEUE (bool)-- If False, the entire queue is updated with samples from the sample rnn at each step
    NLOOPS (int)   -- This saves ram at the cost of more runtime.
    hamiltonian (string) -- Which hamiltonian to train on
    steps (int) -- Number of training steps
    dir (str) -- Output directory, set to <NONE> for no output
    """
    DEFAULTS={'L':16,'Q':32,'K':16,'B':32*16,'TOL':0.15,'M':31/32,'USEQUEUE':True,'NLOOPS':1,
              "hamiltonian":"Rydberg","steps": 12000,"dir":"out"}
    def __init__(self,**kwargs):
        self.__dict__.update(Opt.DEFAULTS)
        self.__dict__.update(kwargs)

    def __str__(self):
        out=""
        for key in self.__dict__:
            line=key+" "*(30-len(key))+ "\t"*3+str(self.__dict__[key])
            out+=line+"\n"
        return out
    
    def apply(self,args):
        kwargs = dict()
        for arg in args:
            try:
                key,val=arg.split("=")
                kwargs[key]=self.sus_cast(val)
            except:pass
        self.__dict__.update(kwargs)
    def sus_cast(self,x0):
        try:
            x=x0
            x=float(x0)
            x=int(x0)
        except:return x
        return x


# In[14]:
import os
def mkdir(dir_):
    try:
        os.mkdir(dir_)
    except:return -1
    return 0

def setup_dir(op):
    """Makes directory for output and saves the run settings there
    Inputs: 
        op (Opt) - Options object
    Outputs:
        Output directory mydir
    
    """
    if op.dir!="<NONE>":
        if op.USEQUEUE:
            mydir= op.dir+"/%s/%d-M=%.3f-B=%d-K=%d"%(op.hamiltonian,op.L,op.M,op.B,op.K)
        else:
            mydir= op.dir+"/%s/%d-NoQ-B=%d-K=%d"%(op.hamiltonian,op.L,op.B,op.K)
    if op.hamiltonian == "TFIM":
        mydir+=("-h=%.1f"%op.h)
    mkdir(op.dir)
    mkdir(op.dir+"/%s"%op.hamiltonian)
    mkdir(mydir)

    with open(mydir+"/settings.txt","w") as f:
        f.write(str(op)+"\n")
    print("Output folder path established")
    return mydir



def queue_train(op):

    
    mydir = setup_dir(op)
    
    trainrnn,optimizer=new_rnn_with_optim("GRU",128,lr=1e-3)
    samplernn = RNN(rnntype="GRU",Nh=128)
    for target_param in samplernn.parameters():
        target_param.data.copy_(target_param.data*5)
    momentum_update(0.0,samplernn,trainrnn)


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


    exact_energy = h.ground()
    print(exact_energy,op.L)


    debug=[]
    losses=[]
    true_energies=[]

    # In[17]:

    samplequeue = torch.zeros([op.B,op.L,1],device=device)
    sump_queue=torch.zeros([op.B],device=device)
    sqrtp_queue=torch.zeros([op.B],device=device)
    Eo_queue = torch.zeros([op.Q],device=device)
    lp_queue = torch.zeros([op.B],device=device)


    def fill_queue():
        global sample,simp,sqrtp,E_i,lp
        for i in range(op.Q):
            sample,sump,sqrtp = samplernn.sample_with_labelsALT(op.K,op.L,grad=False,nloops=op.NLOOPS)
            with torch.no_grad():
                #create and store true values for energy
                lp=samplernn.logprobability(sample)
                E_i=h.localenergyALT(sample,lp,sump,sqrtp)
                Eo_queue[i]=E_i.mean()/(op.L)
                samplequeue[i*op.K:(i+1)*op.K]=sample
                sump_queue[i*op.K:(i+1)*op.K]=sump
                sqrtp_queue[i*op.K:(i+1)*op.K]=sqrtp
                lp_queue[i*op.K:(i+1)*op.K] = lp

    fill_queue()
    print(Eo_queue.mean().item())

    if not op.USEQUEUE:
        nqueue_updates = op.Q
    else:
        nqueue_updates=1


    i=0
    t=time.time()
    for x in range(op.steps):

        for k in range(nqueue_updates):
            sample,sump,sqrtp = samplernn.sample_with_labelsALT(op.K,op.L,grad=False,nloops=op.NLOOPS)

            with torch.no_grad():
                #create and store true values for energy
                lp=samplernn.logprobability(sample)
                E_i=h.localenergyALT(sample,lp,sump,sqrtp)
                Eo_queue[i]=E_i.mean()/(op.L)
                samplequeue[i*op.K:(i+1)*op.K]=sample
                sump_queue[i*op.K:(i+1)*op.K]=sump
                sqrtp_queue[i*op.K:(i+1)*op.K]=sqrtp
                lp_queue[i*op.K:(i+1)*op.K] = lp
            i=(i+1)%op.Q

        logp=trainrnn.logprobability(samplequeue)

        with torch.no_grad():
            E=h.localenergyALT(samplequeue,logp,sump_queue,sqrtp_queue)

            #energy mean and variance
            Ev,Eo=torch.var_mean(E)


        #energy correction mean and variance
        CEv,CEo = torch.var_mean(E*torch.exp(logp-lp_queue))

        ERR  = Eo/(op.L)

        if op.B==1:
            loss = (E*logp).mean()
        else:
            loss = (E*logp - Eo*logp).mean()

        #Main loss curve to follow
        losses.append(ERR.cpu().item())


        #if Energy sees an unrealistic improvement
        #ignore this at early stages
        f=0
        if x>2000 and Eo_queue.min()-losses[-1]>op.TOL:
            #backprop the loss because it may clear temp arrays
            loss.backward()
            #zero my gradients
            trainrnn.zero_grad()
            #update momentum and comtinue
            with torch.no_grad():
                #set both rnns to the same thing then fill the queue
                #momentum_update(0.0,samplernn,trainrnn)

                #set it back to the slow moving sample rnn??
                momentum_update(0.0,trainrnn,samplernn)
                #refill the queue to get accurate probabilities
                #fill_queue()
                f=1
                #print("",end="<%.3f|%d>"%(Eo_queue.mean().item(),x))
        else:
            trainrnn.zero_grad()
            loss.backward()
            optimizer.step()
            #do your regular momentum update
            momentum_update(op.M,samplernn,trainrnn)

        debug+=[[Ev.item()**0.5,Eo.item(),CEv.item()**0.5,CEo.item(),loss.item(),E_i.mean().item(),f,time.time()-t]]

        if x%500==0:
            print(int(time.time()-t),end=",%.2f|"%(losses[-1]))
    print(time.time()-t,x+1)


    # In[18]:


    DEBUG = np.array(debug)
    
    
    if op.dir!="<NONE>":
        #print(DEBUG[-1][3]/Lx/Ly-exact_energy,DEBUG[-1][3]/Lx/Ly,DEBUG[-1][1]/Lx/Ly,exact_energy)
        torch.save(trainrnn,mydir+"/T")
        torch.save(samplernn,mydir+"/S")
        np.save(mydir+"/DEBUG",DEBUG)

def reg_train(op):


    mydir = setup_dir(op)
    
    trainrnn,optimizer=new_rnn_with_optim("GRU",128,lr=1e-3)

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
    exact_energy = h.ground()
    print(exact_energy,op.L)

    debug=[]
    losses=[]
    true_energies=[]

    # In[17]:

    samplequeue = torch.zeros([op.B,op.L,1],device=device)
    sump_queue=torch.zeros([op.B],device=device)
    sqrtp_queue=torch.zeros([op.B],device=device)

    def fill_queue():
        for i in range(op.Q):
            sample,sump,sqrtp = trainrnn.sample_with_labelsALT(op.K,op.L,grad=False,nloops=op.NLOOPS)
            samplequeue[i*op.K:(i+1)*op.K]=sample
            sump_queue[i*op.K:(i+1)*op.K]=sump
            sqrtp_queue[i*op.K:(i+1)*op.K]=sqrtp
    i=0
    t=time.time()
    for x in range(op.steps):
        
        #gather samples
        fill_queue()
        # get probability labels
        logp=trainrnn.logprobability(samplequeue)

        #obtain energy
        with torch.no_grad():
            E=h.localenergyALT(samplequeue,logp,sump_queue,sqrtp_queue)
            #energy mean and variance
            Ev,Eo=torch.var_mean(E)

        ERR  = Eo/(op.L)
        
        
        if op.B==1:
            loss = (E*logp).mean()
        else:
            loss = (E*logp - Eo*logp).mean()

        #Main loss curve to follow
        losses.append(ERR.cpu().item())

        #update weights
        trainrnn.zero_grad()
        loss.backward()
        optimizer.step()

        # many repeat values but it keeps the same format as no queue
        debug+=[[Ev.item()**0.5,Eo.item(),Ev.item()**0.5,Eo.item(),loss.item(),Eo.item(),0,time.time()-t]]

        if x%500==0:
            print(int(time.time()-t),end=",%.2f|"%(losses[-1]))
    print(time.time()-t,x+1)

    # In[18]:


    DEBUG = np.array(debug)
    import os

    if op.dir!="<NONE>":
        #print(DEBUG[-1][3]/Lx/Ly-exact_energy,DEBUG[-1][3]/Lx/Ly,DEBUG[-1][1]/Lx/Ly,exact_energy)
        torch.save(trainrnn,mydir+"/T")
        #torch.save(samplernn,mydir+"/S")
        np.save(mydir+"/DEBUG",DEBUG)
        
        
        
if __name__=="__main__":        
    import sys
    print(sys.argv[1:])
    op=Opt()
    op.apply(sys.argv[1:])
    op.B=op.K*op.Q
    print(op)
    if op.USEQUEUE:
        queue_train(op)
    else:
        reg_train(op)