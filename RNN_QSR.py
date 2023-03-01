
# # Quantum state to represent
# 
# - Start with Rydberg system
# 
# Transverse and longitudinal view of ising model
# 
# Excited state encourages nearby (within radius $R_b$)states to tend towards ground states

# In[1]:


import numpy as np
import math,time,json
import torch
from torch import nn
ngpu=1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device,torch.cuda.get_device_properties(device))


# # Estimating the Rydberg Hamiltonian:
class Options:
    """Base class for managing options"""
    def __init__(self,**kwargs):
        self.__dict__.update(self.get_defaults())
        self.__dict__.update(kwargs)

    def get_defaults(self):
        """This is where you define your default parameters"""
        return dict()
        
    def __str__(self):
        out=""
        for key in self.__dict__:
            line=key+" "*(30-len(key))+ "\t"*3+str(self.__dict__[key])
            out+=line+"\n"
        return out
    def cmd(self):
        """Returns a string with command line arguments corresponding to the options
            Outputs:
                out (str) - a single string of space-separated command line arguments
        """
        out=""
        for key in self.__dict__:
            line=key+"="+str(self.__dict__[key])
            out+=line+" "
        return out[:-1]
    
    def apply(self,args,warn=True):
        """Takes in a tuple of command line arguments and turns them into options
        Inputs:
            args (tuple<str>) - Your command line arguments
        
        """
        kwargs = dict()
        for arg in args:
            try:
                key,val=arg.split("=")
                kwargs[key]=self.cmd_cast(val)
                if warn and (not key in self.__dict__):
                    print("Unknown Argument: %s"%key)
            except:pass
        self.__dict__.update(kwargs)
    def cmd_cast(self,x0):
        """Casting from a string to other datatypes
            Inputs
                x0 (string) - A string which could represent an int or float or boolean value
            Outputs
                x (?) - The best-fitting cast for x0
        """
        try:
            if x0=="True":return True
            elif x0=="False":return False
            elif x0=="None":return None
            x=x0
            x=float(x0)
            x=int(x0)
        except:return x
        return x
    def from_file(self,fn):
        """Depricated: Takes files formatted in the __str__ format and turns them into a set of options
        Instead of using this, consider using save() and load() functions. 
        """
        kwargs = dict()
        with open(fn,"r") as f:
          for line in f:
            line=line.strip()
            split = line.split("\t")
            key,val = split[0].strip(),split[-1].strip()
            try:
                kwargs[key]=self.cmd_cast(val)
            except:pass
        self.__dict__.update(kwargs)
        
    def save(self,fn):
        """Saves the options in json format
        Inputs:
            fn (str) - The file destination for your output file (.json is not appended automatically)
        Outputs:
            A plain text json file
        """
        with open(fn,"w") as f:
            json.dump(self.__dict__, f, indent = 4)
            
    def load(self,fn):
        """Saves  options stored in json format
        Inputs:
            fn (str) - The file source (.json is not appended automatically)
        """
        with open(fn,"r") as f:
            kwargs = json.load(f)
        self.__dict__.update(kwargs)
    def copy(self):
        return Options(**self.__dict__)


class OptionManager():
    
    registry = dict()
    @staticmethod
    def register(name: str , opt: Options):
        OptionManager.registry[name.upper()] = opt
    
    @staticmethod
    def parse_cmd(args: list) -> dict:
        output=dict()
        sub_args=[]
        for arg in args[::-1]:
            # --name Signifies a new set of options
            if arg[:2] == "--":
                arg=arg.upper()
                #make sure the name is registered
                if not arg[2:] in OptionManager.registry:
                    raise Exception("Argument %s Not Registered"%arg)
                #copy the defaults and apply the new options
                opt = OptionManager.registry[arg[2:]].copy()
                opt.apply(sub_args)
                output[arg[2:]] = opt
                # Reset the collection of arguments
                sub_args=[]
            #otherwise keep adding options
            else:
                sub_args+=[arg]
        return output
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
    
    def magnetizations(self, samples):
        B = samples.shape[0]
        L = samples.shape[1]
        mag = torch.zeros(B, device=self.device)
        abs_mag = torch.zeros(B, device=self.device)
        sq_mag = torch.zeros(B, device=self.device)
        stag_mag = torch.zeros(B, device=self.device)

        with torch.no_grad():
            samples_pm = 2 * samples - 1
            mag += torch.sum(samples_pm.squeeze(2), axis=1)
            abs_mag += torch.abs(torch.sum(samples_pm.squeeze(2), axis=1))
            sq_mag += torch.abs(torch.sum(samples_pm.squeeze(2), axis=1))**2
            
            samples_reshape = torch.reshape(samples.squeeze(2), (B, int(np.sqrt(L)), int(np.sqrt(L))))
            for i in range(int(np.sqrt(L))):
                for j in range(int(np.sqrt(L))):
                    stag_mag += (-1)**(i+j) * (samples_reshape[:,i,j] - 0.5)

        return mag, abs_mag, sq_mag, stag_mag / L

    def ground(self):
        """Returns the ground state energy E/L"""
        raise NotImplementedError



class Rydberg(Hamiltonian):
    E={16:-0.4534,36:-0.4221,64:-0.4058,144:-0.38852,256:-0.38052,576:-0.3724,1024:-0.3687,2304:-0.3645}
    Err = {16: 0.0001,36: 0.0005,64: 0.0005, 144: 0.0002, 256: 0.0002, 576: 0.0006,1024: 0.0007,2304: 0.0007}
    
    DEFAULTS = Options(Lx=4,Ly=4,V=7.0,Omega=1.0,delta=1.0)
    def __init__(self,Lx,Ly,V,Omega,delta,device=device,**kwargs):
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

OptionManager.register("rydberg",Rydberg.DEFAULTS)
# In[5]:


class TFIM(Hamiltonian):
    """Implementation of the Transverse field Ising model with Periodic Boundary Conditions"""
    
    DEFAULTS=Options(L=16,h_x=-1.0,J=1.0)
    def __init__(self,L,h_x,J,device=device,**kwargs):
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

OptionManager.register("tfim",TFIM.DEFAULTS)
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
        # type: (Tensor,int) -> Tensor
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
    def __init__(self,nx,ny,Lx,Ly,device=device):
        super().__init__()
        self.nx=nx
        self.ny=ny
        self.Ly=Ly
        self.Lx=Lx
        
        #construct an index tensor for the reverse operation
        indices = torch.arange(Lx*Ly,device=device).unsqueeze(0)
        self.mixed = self.forward(indices).reshape([Lx*Ly])
        #inverse
        self.mixed=torch.argsort(self.mixed)
        
    def forward(self,x):
        # type: (Tensor) -> Tensor
        nx,ny,Lx,Ly=self.nx,self.ny,self.Lx,self.Ly
        """Unflatten a tensor back to 2D, break it into nxn chunks, then flatten the sequence and the chunks
            Input:
                Tensor of shape [B,L]
            Output:
                Tensor of shape [B,L//n^2,n^2]
        """
        #make the input 2D then break it into 2x2 chunks 
        #afterwards reshape the 2x2 chunks to vectors of size 4 and flatten the 2d bit
        return x.view([x.shape[0],Lx,Ly]).unfold(-2,nx,nx).unfold(-2,ny,ny).reshape([x.shape[0],int(Lx*Ly//(nx*ny)),nx*ny])

    def reverse(self,x):
        # type: (Tensor) -> Tensor
        """Inverse function of forward
            Input:
                Tensor of shape [B,L//n^2,n^2]
            Output:
                Tensor of shape [B,L]
        """
        Ly,Lx=self.Ly,self.Lx
        # Reversing is done with an index tensor because torch doesn't have an inverse method for unfold
        return x.reshape([x.shape[0],Ly*Lx])[:,self.mixed]
    
    
    
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
    patch=patch.unsqueeze(0).transpose(-1,0).squeeze(-1).to(torch.int64)
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
    
    INFO = """RNN based sampler where the input sequence is broken up into 'patches' and the output is a sequence of conditional probabilities of all possible patches at position i given the previous 0 to i-1 patches. Each patch is used to update the RNN hidden state, which (after two Fully Connected layers) is used to get the probability labels.
    
    RNN Optional arguments:
    
        L          (int)     -- The total number of atoms in your lattice
    
        Nh         (int)     -- RNN hidden size
    
        patch      (str)     -- Number of atoms input/predicted at once (patch size).
                                The Input sequence will have an effective length of L/prod(patch)
                                Example values: 2x2, 2x3, 2, 4
        
        rnntype    (string)  -- Which type of RNN cell to use. Only ELMAN and GRU are valid options at the moment
    """
    
    DEFAULTS=Options(L=16,patch=1,rnntype="GRU",Nh=256)
    TYPES={"GRU":nn.GRU,"ELMAN":nn.RNN,"LSTM":nn.LSTM}
    def __init__(self,L,patch,rnntype,Nh,device=device, **kwargs):
        
        super(PRNN, self).__init__(device=device)
        if type(patch)==str and len(patch.split("x"))==2:
            px,py = [int(a) for a in patch.split("x")]
            L=int(L**0.5)
            self.patch=Patch2D(px,py,L,L)
            self.L = int(L**2//(px*py))
            self.p=px*py
        else:
            p=int(patch)
            self.patch=Patch1D(p,L)
            self.L = int(L//p)
            self.p = p
        
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
    
    @torch.jit.export
    def logprobability(self,input,h0=None):
        # type: (Tensor,Optional[Tensor]) -> Tensor
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
        
        if h0 is None:
            h0=torch.zeros([1,input.shape[0],self.Nh],device=self.device)
        out,h=self.rnn(data,h0)
        output = self.lin(out)
        
        #real is going to be a onehot with the index of the appropriate patch set to 1
        #shape will be [B,L//4,16]
        real=genpatch2onehot(input,self.p)
        
        #[B,L//4,16] -> [B,L//4]
        total = torch.sum(real*output,dim=-1)
        #[B,L//4] -> [B]
        logp=torch.sum(torch.log(total),dim=1)
        return logp
    @torch.jit.export
    def sample(self,B,L,h0=None):
        # type: (int,int,Optional[Tensor]) -> Tuple[Tensor,Tensor]
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
        if h0 is None:  
            h=torch.zeros([1,B,self.Nh],device=self.device)
        else:
            h=h0
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

OptionManager.register("rnn",PRNN.DEFAULTS)


def new_rnn_with_optim(rnntype,op,beta1=0.9,beta2=0.999):
    rnn = torch.jit.script(PRNN(op.L,**PRNN.DEFAULTS))
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






    
        
# In[14]:
import os
def mkdir(dir_):
    try:
        os.mkdir(dir_)
    except:return -1
    return 0

def setup_dir(op_dict):
    """Makes directory for output and saves the run settings there
    Inputs: 
        op_dict (dict) - Dictionary of Options objects
    Outputs:
        Output directory mydir
    
    """
    op=op_dict["TRAIN"]
    
    if op.dir=="<NONE>":
        return
    
    hname = op_dict["HAMILTONIAN"].name if "HAMILTONIAN" in op_dict else "NA"
    
    mydir= op.dir+"/%s/%d-B=%d-K=%d%s"%(hname,op.L,op.B,op.K,op.sub_directory)

    os.makedirs(mydir,exist_ok = True)
    biggest=-1
    for paths,folders,files in os.walk(mydir):
        for f in folders:
            try:biggest=max(biggest,int(f))
            except:pass
            
    mydir+="/"+str(biggest+1)
    mkdir(mydir)
    
    print("Output folder path established")
    return mydir

class TrainOpt(Options):
    """
    Training Arguments:
    
        L          (int)     -- Total lattice size (8x8 would be L=64)
        
        Q          (int)     -- Number of minibatches per batch
        
        K          (int)     -- size of each minibatch
        
        B          (int)     -- Total batch size (should be Q*K)
        
        NLOOPS     (int)     -- Number of loops within the off_diag_labels function. Higher values save ram and
                                generally makes the code run faster (up to 2x). Note, you can only set this
                                as high as your effective sequence length. (Take L and divide by your patch size).
        
        steps      (int)     -- Number of training steps
        
        dir        (str)     -- Output directory, set to <NONE> for no output
        
        lr         (float)   -- Learning rate
        
        seed       (int)     -- Random seed for the run
                
        sgrad      (bool)    -- whether or not to sample with gradients. 
                                (Uses less ram when but slightly slower)
                                
        true_grad  (bool)    -- Set to false to approximate the gradients
                                
        sub_directory (str)  -- String to add to the end of the output directory (inside a subfolder)
        
    """
    def get_defaults(self):
        return dict(L=16,Q=1,K=256,B=256,NLOOPS=1,hamiltonian="Rydberg",steps=50000,dir="out",lr=5e-4,kl=0.0,sgrad=False,true_grad=False,sub_directory="")

    
OptionManager.register("train",TrainOpt())
    
import sys
def reg_train(op,net_optim=None,printf=False,mydir=None):
  try:
    
    if "RYDBERG" in op:
        h = Rydberg(**op["RYDBERG"].__dict__)
    elif "TFIM" in op:
        #hope for the best here since there aren't defaults
        h = TFIM(**op["TFIM"].__dict__)
    else:        
        h_opt=Rydberg.DEFAULTS.copy()
        h_opt.Lx=h_opt.Ly=int(op["TRAIN"].L**0.5)
        h = Rydberg(**h_opt.__dict__)
    
    
    
    if mydir==None:
        mydir = setup_dir(op)
    
    
    op=op["TRAIN"]
    
    if op.true_grad:assert op.Q==1
    
    if type(net_optim)==type(None):
        net,optimizer=new_rnn_with_optim("GRU",op)
    else:
        net,optimizer=net_optim



    exact_energy = h.ground()
    print(exact_energy,op.L)

    debug=[]
    losses=[]
    true_energies=[]

    # In[17]:

    #samples
    samplebatch = torch.zeros([op.B,op.L,1],device=device)
    #sum of off diagonal labels for each sample (scaled)
    sump_batch=torch.zeros([op.B],device=device)
    #scaling factors for the off-diagonal sums
    sqrtp_batch=torch.zeros([op.B],device=device)

    def fill_batch():
        with torch.no_grad():
            for i in range(op.Q):
                sample,logp = net.sample(op.K,op.L)
                #get the off diagonal info
                sump,sqrtp = net.off_diag_labels_summed(sample,nloops=op.NLOOPS)
                samplebatch[i*op.K:(i+1)*op.K]=sample
                sump_batch[i*op.K:(i+1)*op.K]=sump
                sqrtp_batch[i*op.K:(i+1)*op.K]=sqrtp
        return logp
    i=0
    t=time.time()
    for x in range(op.steps):
        
        #gather samples and probabilities
        
                
        if op.Q!=1:
            fill_batch()
            logp=net.logprobability(samplebatch)
        else:
            if op.sgrad:
                samplebatch,logp = net.sample(op.B,op.L)
            else:
                with torch.no_grad():samplebatch,_= net.sample(op.B,op.L)
                #if you sample without gradients you have to recompute probabilities with gradients
                logp=net.logprobability(samplebatch)
            
            if op.true_grad:
                sump_batch,sqrtp_batch = net.off_diag_labels_summed(samplebatch,nloops=op.NLOOPS)
            else:
                #don't need gradients on the off diagonal when approximating gradients
                 with torch.no_grad(): sump_batch,sqrtp_batch = net.off_diag_labels_summed(samplebatch,nloops=op.NLOOPS)

        #obtain energy
        with torch.no_grad():
            E=h.localenergyALT(samplebatch,logp,sump_batch,sqrtp_batch)
            #energy mean and variance
            Ev,Eo=torch.var_mean(E)
	    
            MAG, ABS_MAG, SQ_MAG, STAG_MAG  = h.magnetizations(samplequeue)
            mag_v, mag = torch.var_mean(MAG)
            abs_mag_v, abs_mag = torch.var_mean(ABS_MAG)
            sq_mag_v, sq_mag = torch.var_mean(SQ_MAG)
            stag_mag_v, stag_mag = torch.var_mean(STAG_MAG)

        ERR  = Eo/(op.L)
        
        if op.true_grad:
            #get the extra loss term
            h_x= h.offDiag *sump_batch* torch.exp(sqrtp_batch-logp/2)
            loss = (logp*E).mean() + h_x.mean()
            
        else:
            loss =(logp*(E-Eo)).mean()  if op.B>1 else (logp*E).mean()


        #Main loss curve to follow
        losses.append(ERR.cpu().item())
        
        #update weights
        net.zero_grad()
        loss.backward()
        optimizer.step()

        # many repeat values but it keeps the same format as no queue
        #debug+=[[Ev.item()**0.5,Eo.item(),Ev.item()**0.5,Eo.item(),loss.item(),Eo.item(),0,time.time()-t]]
        debug += [[Eo.item(), Ev.item(), mag.item(), mag_v.item(), abs_mag.item(), abs_mag_v.item(), sq_mag.item(), sq_mag_v.item(), stag_mag.item(), stag_mag_v.item(), time.time()-t]]

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
  return DEBUG

        
if __name__=="__main__":        
    import sys
    print(sys.argv[1:])
    train_opt=TrainOpt(K=512,Q=1,dir="RNN")
    
    rnn_opt=PRNN.DEFAULTS.copy()
    
    if "--rnn" in sys.argv[1:]:
        split=sys.argv.index("--rnn")
        train_opt.apply(sys.argv[1:split])
        rnn_opt.apply(sys.argv[split+1:])
    else:
        train_opt.apply(sys.argv[1:])
        
    train_opt.B=train_opt.K*train_opt.Q
    
    rnn = torch.jit.script(PRNN(train_opt.L,**rnn_opt.__dict__))    
    
    beta1=0.9;beta2=0.999
    optimizer = torch.optim.Adam(
    rnn.parameters(), 
    lr=train_opt.lr, 
    betas=(beta1,beta2)
    )
    
    print(train_opt)
    
    mydir=setup_dir(train_opt)
    orig_stdout = sys.stdout
    
    rnn_opt.model_name=PRNN.__name__
    full_opt = Options(train=train_opt.__dict__,model=rnn_opt.__dict__)
    full_opt.save(mydir+"\\settings.json")
    
    f = open(mydir+'\\output.txt', 'w')
    sys.stdout = f
    try:
        reg_train(train_opt,(rnn,optimizer),printf=True,mydir=mydir)
    except Exception as e:
        print(e)
        sys.stdout = orig_stdout
        f.close()
        1/0
    sys.stdout = orig_stdout
    f.close()
