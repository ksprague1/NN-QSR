
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
print(device,torch.cuda.get_device_properties(device))
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



class Rydberg(Hamiltonian):
    E={16:-0.4534,36:-0.4221,64:-0.40522,144:-0.38852,256:-0.38052,576:-0.3724,1024:-0.3687,2304:-0.3645}
    Err = {16: 0.0001,36: 0.0005,64: 0.0002, 144: 0.0002, 256: 0.0002, 576: 0.0006,1024: 0.0007,2304: 0.0007}
    def __init__(self,Lx,Ly,V=7.0,Omega=1.0,delta=1.0,device=device):
        self.Lx       = Lx              # Size along x
        self.Ly       = Ly              # Size along y
        self.V        = V               # Van der Waals potential
        self.delta    = delta           # Detuning
        # off diagonal part is -0.5*Omega
        super(Rydberg,self).__init__(Lx*Ly,-0.5*Omega,device)
    @staticmethod
    def Vij(Ly,Lx,V,matrix):
    #matrix will be size [Lx*Ly,Lx*Ly]
      for i in range(Ly):
        for j in range(Lx):
            #R=Rcutoff**6
            #flatten two indices into one
            idx = Ly*j+i
            # only fill in the upper diagonal
            for k in range(idx+1,Lx*Ly):
                #expand one index into two
                i2 = k%Ly
                j2=k//Ly
                div = ((i2-i)**2+(j2-j)**2)**3
                #if div<=R:
                matrix[idx][k]=V/div
    def buildlattice(self):
        Lx,Ly=self.Lx,self.Ly
        
        #diagonal hamiltonian portion can be written as a matrix multiplication then a dot product        
        mat=np.zeros([self.L,self.L])
        Rydberg.Vij(Lx,Ly,self.V,mat)
        #self.Vij.double()
        with torch.no_grad():
            self.Vij.weight[:,:]=torch.Tensor(mat)
            self.Vij.bias.fill_(-self.delta)
    def ground(self):
        return Rydberg.E[self.Lx*self.Ly]


# In[5]:


class TFIM(Hamiltonian):
    """Implementation of the Transverse field Ising model with Periodic Boundary Conditions"""
    def __init__(self,L,h_x,J=1.0,device=device):
        self.J = J
        self.h=h_x
        super(TFIM,self).__init__(L,h_x,device)
        
    def buildlattice(self):
        #building hamiltonian matrix for diagonal part
        mat=np.zeros([self.L,self.L],dtype=np.float64)
        for i in range(self.L):
            mat[i, (i+1)%self.L] = -self.J
        
        with torch.no_grad():
            self.Vij.weight[:,:]=torch.Tensor(mat)
            self.Vij.bias.fill_(0.0) # no longitudinal field

    def localenergy(self,samples,logp,logppj):
        return super(TFIM,self).localenergyALT(2*samples-1,logp,logppj)

    def localenergyALT(self,samples,logp,sumsqrtp,logsqrtp):
        return super(TFIM,self).localenergyALT(2*samples-1,logp,sumsqrtp,logsqrtp)
    def ground(self):
        """Exact solution for the ground state the 1D TFIM model with PBC"""
        N,h=self.L,self.h/self.J
        Pn = np.pi/N*(np.arange(-N+1,N,2))
        E0 = -1/N*np.sum(np.sqrt(1+h**2-2*h*np.cos(Pn)))
        return E0*self.J


# In[6]:


class Sampler(nn.Module):
    def __init__(self,device=device):
        self.device=device
        super(Sampler, self).__init__()
    def save(self,fn):
        torch.save(self,fn)
    def logprobability(self,input):
        # type: (Tensor) -> Tensor
        """Compute the logscale probability of a given state
            Inputs:
                input - [B,L,1] matrix of zeros and ones for ground/excited states
            Returns:
                logp - [B] size vector of logscale probability labels
        """
        raise NotImplementedError
    @torch.jit.export
    def sample(self,B,L):
        # type: (int,int) -> Tensor
        """ Generates a set states
        Inputs:
            B (int)            - The number of states to generate in parallel
            L (int)            - The length of generated vectors
        Returns:
            samples - [B,L,1] matrix of zeros and ones for ground/excited states
            logprobs - [B] matrix of logscale probabilities (float Tensor)
        """
        raise NotImplementedError
    @torch.jit.export
    def off_diag_labels(self,sample,nloops=1):
        # type: (Tensor,int) -> Tuple[Tensor,Tensor]
        """
        Inputs:
            samples  - [B,L,1] matrix of zeros and ones for ground/excited states
        
        Returns:
            probs - size [B,L] tensor of probabilities of the excitation-flipped states
        """
        D=nloops
        B,L,_=sample.shape
        sflip = torch.zeros([B,L,L,1],device=self.device)
        #collect all of the flipped states into one array
        for j in range(L):
            #get all of the states with one spin flipped
            sflip[:,j] = sample*1.0
            sflip[:,j,j] = 1-sflip[:,j,j]
        #compute all of their logscale probabilities
        with torch.no_grad():
            probs=torch.zeros([B*L],device=self.device)
            tmp=sflip.view([B*L,L,1])
            for k in range(D):
                probs[k*B*L//D:(k+1)*B*L//D] = self.logprobability(tmp[k*B*L//D:(k+1)*B*L//D])

        return probs.reshape([B,L])
    @torch.jit.export
    def off_diag_labels_summed(self,sample,nloops=1):
        # type: (Tensor,int) -> Tuple[Tensor,Tensor]
        """
        Inputs:
            samples  - [B,L,1] matrix of zeros and ones for ground/excited states
        
        Returns:
            logsqrtp - size B vector of average (log p)/2 values used for numerical stability 
                       when calculating sum_s'(sqrt[p(s')/p(s)]) 
            sumsqrtp - size B vector of exp(-logsqrtp)*sum(sqrt[p(s')]).
        """
        probs = self.off_diag_labels(sample,nloops)
        #get the average of our logprobabilities and divide by 2
        logsqrtp=probs.mean(dim=1)/2
        #compute the sum with a constant multiplied to keep the sum closeish to 1
        sumsqrtp = torch.exp(probs/2-logsqrtp.unsqueeze(1)).sum(dim=1)
        return sumsqrtp,logsqrtp
    
    



# In[8]: Functions for making Patches & doing probability traces
class Patch2D(nn.Module):
    def __init__(self,n,Lx):
        super().__init__()
        self.n=n
        self.Lx=Lx
    def forward(self,x):
        # type: (Tensor) -> Tensor
        n,Lx=self.n,self.Lx
        """Unflatten a tensor back to 2D, break it into nxn chunks, then flatten the sequence and the chunks
            Input:
                Tensor of shape [B,L]
            Output:
                Tensor of shape [B,L//n^2,n^2]
        """
        #make the input 2D then break it into 2x2 chunks 
        #afterwards reshape the 2x2 chunks to vectors of size 4 and flatten the 2d bit
        return x.view([x.shape[0],Lx,Lx]).unfold(-2,n,n).unfold(-2,n,n).reshape([x.shape[0],int(Lx*Lx//n**2),int(n**2)])

    def reverse(self,x):
        # type: (Tensor) -> Tensor
        """Inverse function of forward
            Input:
                Tensor of shape [B,L//n^2,n^2]
            Output:
                Tensor of shape [B,L]
        """
        n,Lx=self.n,self.Lx
        # original sequence order can be retrieved by chunking twice more
        #in the x-direction you should have chunks of size 2, but in y it should
        #be chunks of size Ly//2
        return x.unfold(-2,Lx//n,Lx//n).unfold(-2,n,n).reshape([x.shape[0],Lx*Lx])
    
class Patch1D(nn.Module):
    def __init__(self,n,L):
        super().__init__()
        self.n=n
        self.L = L
    def forward(self,x):
        # type: (Tensor) -> Tensor
        """Break a tensor into chunks, essentially a wrapper of reshape
            Input:
                Tensor of shape [B,L]
            Output:
                Tensor of shape [B,L/n,n]
        """
        #make the input 2D then break it into 2x2 chunks 
        #afterwards reshape the 2x2 chunks to vectors of size 4 and flatten the 2d bit
        return x.reshape([x.shape[0],self.L//self.n,self.n])

    def reverse(self,x):
        # type: (Tensor) -> Tensor
        """Inverse function of forward
            Input:
                Tensor of shape [B,L/n,n]
            Output:
                Tensor of shape [B,L]
        """
        # original sequence order can be retrieved by chunking twice more
        #in the x-direction you should have chunks of size 2, but in y it should
        #be chunks of size Ly//2
        return x.reshape([x.shape[0],self.L])


    
@torch.jit.script
def genpatch2onehot(patch,p):
    # type: (Tensor,int) -> Tensor
    """ Turn a sequence of size p patches into a onehot vector
    Inputs:
        patch - Tensor of shape [?,p]
        p (int) - the patch size
    
    """
    #moving the last dimension to the front
    patch=patch.unsqueeze(0).transpose(-1,0).squeeze(-1)
    out=torch.zeros(patch.shape[1:],device=patch.device)
    for i in range(p):
        out+=patch[i]<<i
    return nn.functional.one_hot(out.to(torch.int64), num_classes=1<<p)
    
# In[10]:

class PRNN(Sampler):
    """
    Patched Recurrent Neural Network Implementation.
    
    The network is patched as the sequence is broken into patches of size p, then entire patches are sampled at once.
    This means the sequence length is reduced from L to L/p but the output layer must now use a softmax over 2**p possible
    patches. Setting p above 5 is not recommended.
    
    Note for _2D = True, p actually becomes a pxp patch so the sequence is reduced to L/p^2 and it's a softmax over
    2^(p^2) patches so p=2 is about the only patch size which makes sense
    
    """
    TYPES={"GRU":nn.GRU,"ELMAN":nn.RNN,"LSTM":nn.LSTM}
    def __init__(self,L,p,_2D=False,rnntype="GRU",Nh=128,device=device, **kwargs):
        super(PRNN, self).__init__(device=device)
        
        if _2D:
            self.patch=Patch2D(p,L)
            self.L = int(L**2//p**2)
            self.p=int(p**2)
        else:
            self.patch=Patch1D(p,L)
            self.L = int(L//p)
            self.p = int(p)
        
        assert rnntype!="LSTM"
        #rnn takes input shape [B,L,1]
        self.rnn = PRNN.TYPES[rnntype](input_size=self.p,hidden_size=Nh,batch_first=True)
        
        
        self.lin = nn.Sequential(
                nn.Linear(Nh,Nh),
                nn.ReLU(),
                nn.Linear(Nh,1<<self.p),
                nn.Softmax(dim=-1)
            )
        self.Nh=Nh
        self.rnntype=rnntype
        
        #create a tensor of all possible patches
        self.options=torch.zeros([1<<self.p,self.p],device=self.device)
        tmp=torch.arange(1<<self.p,device=self.device)
        for i in range(self.p):
            self.options[:,i]=(tmp>>i)%2
            
        
        self.to(device)
    def forward(self, input):
        # h0 has shape [1,B,H]
        h0=torch.zeros([1,input.shape[0],self.Nh],device=self.device)
        out,h=self.rnn(input,h0)
        return self.lin(out)
    
    @torch.jit.export
    def logprobability(self,input):
        # type: (Tensor) -> Tensor
        """Compute the logscale probability of a given state
            Inputs:
                input - [B,L,1] matrix of zeros and ones for ground/excited states
            Returns:
                logp - [B] size vector of logscale probability labels
        """
                
        #shape is modified to [B,L//4,4]
        input = self.patch(input.squeeze(-1))
        data=torch.zeros(input.shape,device=self.device)
        #batch first
        data[:,1:]=input[:,:-1]
        # [B,L//4,Nh] -> [B,L//4,16]
        output = self.forward(data)
        
        #real is going to be a onehot with the index of the appropriate patch set to 1
        #shape will be [B,L//4,16]
        real=genpatch2onehot(input,self.p)
        
        #[B,L//4,16] -> [B,L//4]
        total = torch.sum(real*output,dim=-1)
        #[B,L//4] -> [B]
        logp=torch.sum(torch.log(total),dim=1)
        return logp
    @torch.jit.export
    def sample(self,B,L):
        # type: (int,int) -> Tuple[Tensor,Tensor]
        """ Generates a set states
        Inputs:
            B (int)            - The number of states to generate in parallel
            L (int)            - The length of generated vectors
        Returns:
            samples - [B,L,1] matrix of zeros and ones for ground/excited states
        """
        #length is divided by four due to patching
        L=L//self.p
        #if self.rnntype=="LSTM":
        #    h=[torch.zeros([1,B,self.Nh],device=self.device),
        #       torch.zeros([1,B,self.Nh],device=self.device)]
            #h is h0 and c0
        #else:
        h=torch.zeros([1,B,self.Nh],device=self.device)
        #Sample set will have shape [B,L,p]
        #need one extra zero batch at the start for first pred hence input is [L+1,B,1] 
        input = torch.zeros([B,L+1,self.p],device=self.device)
        sample = torch.zeros([B,self.p],device=self.device)
        logp = torch.zeros([B],device=self.device)
        #with torch.no_grad():
        for idx in range(1,L+1):
            #out should be batch first [B,L,Nh]
            out,h=self.rnn(sample.unsqueeze(1),h)
            #check out the probability of all 1<<p vectors
            probs=self.lin(out[:,0,:]).view([B,1<<self.p])
            #sample from the probability distribution
            indices = torch.multinomial(probs,1,False).squeeze(1)
            #extract samples
            sample = self.options[indices]
            
            onehot = nn.functional.one_hot(indices, num_classes=1<<self.p)
            
            logp+= torch.log(torch.sum(onehot*probs,dim=-1))
            
            #set input to the sample that was actually chosen
            input[:,idx] = sample
        #remove the leading zero in the input    
        #sample is repeated 16 times at 3rd index so we just take the first one
        return self.patch.reverse(input[:,1:]).unsqueeze(-1),logp
        
    @torch.jit.export
    def off_diag_labels(self,sample,nloops=1):
        # type: (Tensor,int) -> Tensor
        """label all of the flipped states  - set D as high as possible without it slowing down runtime
        Parameters:
            sample - [B,L,1] matrix of zeros and ones for ground/excited states
            B,L (int) - batch size and sequence length
            D (int) - Number of partitions sequence-wise. We must have L%D==0 (D divides L)
            
        Outputs:
            
            sample - same as input
            probs - [B,L] matrix of probabilities of states with the jth excitation flipped
        """
        D=nloops
        B,L,_=sample.shape
        sample0=sample
        #sample is batch first at the moment
        sample = self.patch(sample.squeeze(-1))
        
        sflip = torch.zeros([B,L,L//self.p,self.p],device=self.device)
        #collect all of the flipped states into one array
        for j in range(L//self.p):
            #have to change the order of in which states are flipped for the cache to be useful
            for j2 in range(self.p):
                sflip[:,j*self.p+j2] = sample*1.0
                sflip[:,j*self.p+j2,j,j2] = 1-sflip[:,j*self.p+j2,j,j2]
            
        #compute all of their logscale probabilities
        with torch.no_grad():
            
            data=torch.zeros(sample.shape,device=self.device)
            
            data[:,1:]=sample[:,:-1]
            
            #add positional encoding and make the cache
            
            h=torch.zeros([1,B,self.Nh],device=self.device)
            
            out,_=self.rnn(data,h)
            
            #cache for the rnn is the output in this sense
            #shape [B,L//4,Nh]
            cache=out
            probs=torch.zeros([B,L],device=self.device)
            #expand cache to group L//D flipped states
            cache=cache.unsqueeze(1)

            #this line took like 1 hour to write I'm so sad
            #the cache has to be shaped such that the batch parts line up
                        
            cache=cache.repeat(1,L//D,1,1).reshape(B*L//D,L//self.p,cache.shape[-1])
                        
            pred0 = self.lin(out)
            #shape will be [B,L//4,16]
            real=genpatch2onehot(sample,self.p)
            #[B,L//4,16] -> [B,L//4]
            total0 = torch.sum(real*pred0,dim=-1)

            for k in range(D):

                N = k*L//D
                #next couple of steps are crucial          
                #get the samples from N to N+L//D
                #Note: samples are the same as the original up to the Nth spin
                real = sflip[:,N:(k+1)*L//D]
                #flatten it out and set to sequence first
                tmp = real.reshape([B*L//D,L//self.p,self.p])
                #set up next state predction
                fsample=torch.zeros(tmp.shape,device=self.device)
                fsample[:,1:]=tmp[:,:-1]
                #grab your rnn output
                if k==0:
                    out,_=self.rnn(fsample,cache[:,0].unsqueeze(0)*0.0)
                else:
                    out,_=self.rnn(fsample[:,N//self.p:],cache[:,N//self.p-1].unsqueeze(0)*1.0)
                # grab output for the new part
                output = self.lin(out)
                # reshape output separating batch from spin flip grouping
                pred = output.view([B,L//D,(L-N)//self.p,1<<self.p])
                real = genpatch2onehot(real[:,:,N//self.p:],self.p)
                total = torch.sum(real*pred,dim=-1)
                #sum across the sequence for probabilities
                #print(total.shape,total0.shape)
                logp=torch.sum(torch.log(total),dim=-1)
                logp+=torch.sum(torch.log(total0[:,:N//self.p]),dim=-1).unsqueeze(-1)
                probs[:,N:(k+1)*L//D]=logp
                
        return probs




def new_rnn_with_optim(rnntype,op,beta1=0.9,beta2=0.999):
    rnn = torch.jit.script(PRNN(op.L,op.patch,rnntype=rnntype,Nh=op.Nh))
    optimizer = torch.optim.Adam(
    rnn.parameters(), 
    lr=op.lr, 
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
    Nh (int) -- RNN hidden size / Transformer token size
    lr (float) -- Learning rate
    patch (int) -- Number of atoms to consider at once
    _2D (bool) -- If you should make 2D patches and pe or not
    kl (float >=0) -- loss term for kl divergence
    ffq (bool) -- whether or not the model has a builtin fill queue function
    """
    DEFAULTS={'L':16,'Q':1,'K':256,'B':256,'TOL':0.15,'M':31/32,'USEQUEUE':False,'NLOOPS':1,
              "hamiltonian":"Rydberg","steps": 12000,"dir":"out","Nh":128,"lr":5e-4,"patch":1,"kl":0.0,"ffq":False,
             "_2D":False}
    def __init__(self,**kwargs):
        self.__dict__.update(Opt.DEFAULTS)
        self.__dict__.update(kwargs)

    def __str__(self):
        out=""
        for key in self.__dict__:
            line=key+" "*(30-len(key))+ "\t"*3+str(self.__dict__[key])
            out+=line+"\n"
        return out
    def cmd(self):
        """Returns a string with command line arguments corresponding to the options"""
        out=""
        for key in self.__dict__:
            line=key+"="+str(self.__dict__[key])
            out+=line+" "
        return out[:-1]
    
    def apply(self,args):
        """Takes in a set of command line arguments and turns them into options"""
        kwargs = dict()
        for arg in args:
            try:
                key,val=arg.split("=")
                kwargs[key]=self.sus_cast(val)
            except:pass
        self.__dict__.update(kwargs)
    def sus_cast(self,x0):
        """Casting from a string to a floating point or integer
        TODO: add support for booleans
        """
        try:
            if x0 =="True":return True
            if x0 =="False":return False
            if x0 =="None":return None
            x=x0
            x=float(x0)
            x=int(x0)
        except:return x
        return x
    def from_file(self,fn):
        """Takes a file and converts it to options"""
        kwargs = dict()
        with open(fn,"r") as f:
          for line in f:
            line=line.strip()
            split = line.split("\t")
            key,val = split[0].strip(),split[-1].strip()
            try:
                kwargs[key]=self.sus_cast(val)
            except:pass
        self.__dict__.update(kwargs)


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
            mydir= op.dir+"/%s/%d-M=%.3f-B=%d-K=%d-Nh=%d-kl=%.2f"%(op.hamiltonian,op.L,op.M,op.B,op.K,op.Nh,op.kl)
        else:
            mydir= op.dir+"/%s/%d-NoQ-B=%d-K=%d-Nh=%d-P=%d"%(op.hamiltonian,op.L,op.B,op.K,op.Nh,op.patch)
    if op.hamiltonian == "TFIM":
        mydir+=("-h=%.1f"%op.h)
    mkdir(op.dir)
    mkdir(op.dir+"/%s"%op.hamiltonian)
    mkdir(mydir)
    biggest=-1
    for paths,folders,files in os.walk(mydir):
        for f in folders:
            try:biggest=max(biggest,int(f))
            except:pass
            
    mydir+="/"+str(biggest+1)
    mkdir(mydir)
        
    with open(mydir+"/settings.txt","w") as f:
        f.write(str(op)+"\n")
    print("Output folder path established")
    return mydir



import sys
def reg_train(op,to=None,printf=False,mydir=None):
  try:
    
    if mydir==None:
        mydir = setup_dir(op)
    
    if type(to)==type(None):
        net,optimizer=new_rnn_with_optim("GRU",op)
    else:
        net,optimizer=to


    if op.hamiltonian=="Rydberg":
        # Hamiltonian parameters
        N = op.L   # Total number of atoms
        V = 7.0     # Strength of Van der Waals interaction
        Omega = 1.0 # Rabi frequency
        delta = 1.0 # Detuning 
        
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
        if op.ffq:
            net.ffq(samplequeue,sump_queue,sqrtp_queue,op.Q,op.K,op.L,op.NLOOPS)
        else:
          for i in range(op.Q):
            sample,logp = net.sample(op.K,op.L)
            sump,sqrtp = net.off_diag_labels_summed(sample,nloops=op.NLOOPS)
            samplequeue[i*op.K:(i+1)*op.K]=sample
            sump_queue[i*op.K:(i+1)*op.K]=sump
            sqrtp_queue[i*op.K:(i+1)*op.K]=sqrtp
        return logp
    i=0
    t=time.time()
    for x in range(op.steps):
        
        #gather samples and probabilities
        logp = fill_queue()
                
        # get probability labels for the Q>1 case
        if op.Q!=1:
            logp=net.logprobability(samplequeue)

        #obtain energy
        with torch.no_grad():
            E=h.localenergyALT(samplequeue,logp,sump_queue,sqrtp_queue)
            #energy mean and variance
            Ev,Eo=torch.var_mean(E)

        ERR  = Eo/(op.L)
        
        if op.B==1:
            loss = ((E-op.kl)*logp).mean()
        else:
            loss = (E*logp - (Eo+op.kl)*logp).mean()

        #Main loss curve to follow
        losses.append(ERR.cpu().item())
        
        #update weights
        net.zero_grad()
        loss.backward()
        optimizer.step()

        # many repeat values but it keeps the same format as no queue
        debug+=[[Ev.item()**0.5,Eo.item(),Ev.item()**0.5,Eo.item(),loss.item(),Eo.item(),0,time.time()-t]]

        if x%500==0:
            print(int(time.time()-t),end=",%.3f|"%(losses[-1]))
            if x%4000==0:print()
            if printf:sys.stdout.flush()
    print(time.time()-t,x+1)

    # In[18]:

    import os
    DEBUG = np.array(debug)
    

    if op.dir!="<NONE>":
        np.save(mydir+"/DEBUG",DEBUG)
        #print(DEBUG[-1][3]/Lx/Ly-exact_energy,DEBUG[-1][3]/Lx/Ly,DEBUG[-1][1]/Lx/Ly,exact_energy)
        net.save(mydir+"/T")
        #torch.save(samplernn,mydir+"/S")
        
  except KeyboardInterrupt:
    if op.dir!="<NONE>":
        import os
        DEBUG = np.array(debug)
        np.save(mydir+"/DEBUG",DEBUG)
        #print(DEBUG[-1][3]/Lx/Ly-exact_energy,DEBUG[-1][3]/Lx/Ly,DEBUG[-1][1]/Lx/Ly,exact_energy)
        net.save(mydir+"/T")
        #torch.save(samplernn,mydir+"/S")
        

        
if __name__=="__main__":        
    import sys
    print(sys.argv[1:])
    op=Opt(K=512,B=1,Nh=256,dir="RNN")
    op.apply(sys.argv[1:])
    op.B=op.K*op.Q
    print(op)
    if not op.USEQUEUE:
        reg_train(op)