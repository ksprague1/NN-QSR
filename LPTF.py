from RNN_QSR import *
from PTF import *

class LPTF(Sampler):
    
    """
    Sampler class which uses a transformer for long term information and a smaller Sampler for short term information
    This can either be in the form of an RNN or a transformer (likely patched).
    
    The sequence is broken into 2D patches (4x4 by default), each patch is expanded to a tensor of size Nh (be repeating it),\
    then a positional encoding is added. You then apply masked self-attention to the patches num_layers times, with the final
    outputs fed in as the initial hidden state of an rnn.
    
    Going with 4x4 patches, you can use these patches as a sequence to get a factorized probability of the entire
    4x4 patch by feeding the 2x2 patches in one at a time and outputting a size 16 tensor 
    (probability of all possible next 2x2 patches) for each patch. The output is obtained by applying two FC layers to
    the hidden state of the rnn.
    
    
    Here is an example of how everything comes together
    
    Say you have a 16x16 input and Nh=128, this input is broken into 16 4x4 patches which are repeated 8 times and
    given a positional encoding. Masked self attention is done between the 16 patches (size Nh) for N layers, then
    16 RNNs are given the outputs in parallel as the hidden state. Now the original input is broken into 16 sets of 4 2x2
    patches. These length 4 sequences are given to the rnns (16 in parallel all sharing the same weights) and the outputs
    are then grouped together such that you end up with a length 64 sequence of vectors of size 16. this gives your probability.
    You can easily calculate it by taking the output (of 16) corresponding to each 2x2 patch and multiplying all 64 of them
    together (or adding them in logscale).
    
    """
    INFO = """
    
    Transformer based sampler where the input sequence is broken up into large 'patches' and the output is a sequence of conditional probabilities of all possible patches at position i given the previous 0 to i-1 patches. Each patch is projected into a token with an added positional encoding. The sequence of encoded patches is used as transformer input. This specific model is used for very large patches where doing a softmax over all possible patches is not feasable thus a subsampler must be used to factorize these probabilities.
    
    
    LPTF Optional arguments:
    
        L          (int)     -- The total number of atoms in your lattice.
    
        Nh         (int)     -- Transformer token size. Input patches are projected to match the token size.
                                Note: When using an RNN subsampler this Nh MUST match the rnn's Nh
    
        patch      (int)     -- Number of atoms input/predicted at once (patch size).
                                The Input sequence will have an effective length of L/patch
    
        _2D        (bool)    -- Whether or not to make patches 2D (Ex patch=2 and _2D=True give shape 2x2 patches)
        
        dropout    (float)   -- The amount of dropout to use in the transformer layers
        
        num_layers (int)     -- The number of transformer layers to use
        
        nhead     (int)     -- The number of heads to use in Multi-headed Self-Attention. This should divide Nh
        
        subsampler (Sampler) -- The inner model to use for probability factorization. This is set implicitly
                                by including --rnn or --ptf arguments.
    
    """
    DEFAULTS=Options(L=64,patch=1,_2D=False,Nh=128,dropout=0.0,num_layers=2,nhead=8)
    def __init__(self,subsampler,L,patch,_2D,Nh,dropout,num_layers,nhead,device=device, **kwargs):
        super(Sampler, self).__init__()
        #print(nhead)
        p=patch
        
        if _2D:
            L=int(L**0.5)
            self.pe = PE2D(Nh, L//p,L//p,device)
            self.patch=Patch2D(p,L)
            self.L = int(L**2//p**2)
            self.p=int(p**2)
        else:
            self.pe = PE1D(Nh,L//p,device)
            self.patch=Patch1D(p,L)
            self.L = int(L//p)
            self.p = int(p)
            

        self.tokenize=nn.Sequential(
                nn.Linear(self.p,Nh),
                nn.Tanh()
        )
            
            
        self.device=device
        #Encoder only transformer
        #misinterperetation on encoder made it so this code does not work
        self.transformer = FastMaskedTransformerEncoder(Nh=Nh,dropout=dropout,num_layers=num_layers,nhead=nhead)       
        
        self.subsampler = subsampler#subsample(**rnnargs)
        
        self.set_mask(self.L)
        
        self.to(device)
    def set_mask(self, L):
        # type: (int)
        """Initialize the self-attention mask"""
        self.L=L
        self.transformer.set_mask(L)
        self.pe.L=L

    @torch.jit.export
    def logprobability(self,input):
        # type: (Tensor) -> Tensor
        """Compute the logscale probability of a given state
            Inputs:
                input - [B,L,1] matrix of zeros and ones for ground/excited states
            Returns:
                logp - [B] size vector of logscale probability labels
        """
        
        if input.shape[1]//self.p!=self.L:
            self.set_mask(input.shape[1]//self.p)
        #pe should be sequence first [L,B,Nh]
        
        #shape is modified to [L//p,B,p]
        input = self.patch(input.squeeze(-1)).transpose(1,0)
        
        data=torch.zeros(input.shape,device=self.device)
        data[1:]=input[:-1]
        
        #[L//p,B,p] -> [L//p,B,Nh]
        encoded=self.pe(self.tokenize(data))
        #shape is preserved
        output = self.transformer(encoded)
        
        Lp,B,Nh=output.shape
        # [L//p,B,Nh] -> [1,L//p*B,Nh]
        h0 = output.view([1,Lp*B,Nh])
        rnn_input = input.reshape([Lp*B,self.p])
        # [L//p*B,p],[1,L//p*B,Nh] -> [L//p,B]
        logsubsample = self.subsampler.logprobability(rnn_input,h0).view([Lp,B])
        
        #[L//p,B] -> [B]
        logp=torch.sum(logsubsample,dim=0)
        return logp
    
    @torch.jit.export
    def sample(self,B,L,cache=None):
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
        
        #return (torch.rand([B,L,1],device=device)<0.5).to(torch.float32)
        #Sample set will have shape [L/p,B,p]
        #need one extra zero batch at the start for first pred hence input is [L+1,B,1] 
        input = torch.zeros([L+1,B,self.p],device=self.device)

        logp = torch.zeros([B],device=self.device)
        
        #with torch.no_grad():
        for idx in range(1,L+1):
            
            #pe should be sequence first [l,B,Nh]
            # multiply by 1 to copy the tensor
            encoded_input = self.pe(self.tokenize(input[:idx,:,:]*1))
                        
            #Get transformer output
            output,cache = self.transformer.next_with_cache(encoded_input,cache)
            #get state and probability by sampling from the subsample
            sample,logsubsample = self.subsampler.sample(B,self.p,output[-1].view([1,B,output.shape[-1]]))
            
            logp+=logsubsample
            #set input to the sample that was actually chosen
            input[idx] = sample.squeeze(-1)
            
        #remove the leading zero in the input    
        input=input[1:]
        #sample is repeated 16 times at 3rd index so we just take the first one
        return self.patch.reverse(input.transpose(1,0)).unsqueeze(-1),logp
    
    
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
            
        #switch sample into sequence-first
        sample = sample.transpose(1,0)
            
        #compute all of their logscale probabilities
        with torch.no_grad():

            data=torch.zeros(sample.shape,device=self.device)
            data[1:]=sample[:-1]
            
            #[L//p,B,p] -> [L//p,B,Nh]
            encoded=self.pe(self.tokenize(data))
            
            #add positional encoding and make the cache
            out,cache=self.transformer.make_cache(encoded)
            probs=torch.zeros([B,L],device=self.device)
            #expand cache to group L//D flipped states
            cache=cache.unsqueeze(2)

            #this line took like 1 hour to write I'm so sad
            #the cache has to be shaped such that the batch parts line up
                        
            cache=cache.repeat(1,1,L//D,1,1).transpose(2,3).reshape(cache.shape[0],L//self.p,B*L//D,cache.shape[-1])
            
            Lp,B,Nh=out.shape
            # [L//p,B,Nh] -> [1,L//p*B,Nh]
            h0 = out.view([1,Lp*B,Nh])
            rnn_input = sample.reshape([Lp*B,self.p])
            # [L//p*B,p],[1,L//p*B,Nh] -> [L//p,B]
            logsubsample0 = self.subsampler.logprobability(rnn_input,h0).view([Lp,B])

            for k in range(D):

                N = k*L//D
                #next couple of steps are crucial          
                #get the samples from N to N+L//D
                #Note: samples are the same as the original up to the Nth spin
                real = sflip[:,N:(k+1)*L//D]
                #flatten it out and set to sequence first
                tmp = real.reshape([B*L//D,L//self.p,self.p]).transpose(1,0)
                #set up next state predction
                fsample=torch.zeros(tmp.shape,device=self.device)
                fsample[1:]=tmp[:-1]
                # put sequence before batch so you can use it with your transformer
                tgt=self.pe(self.tokenize(fsample))
                #grab your transformer output
                out,_=self.transformer.next_with_cache(tgt,cache[:,:N//self.p],N//self.p)

                output = out[N//self.p:]
                
                #[(L-N)/p,B*L/D,Nh]
                Lp2,B2,Nh=output.shape
                
                # [(L-N)/p,B*L/D,Nh] -> [1,((L-N)/p)*(B*L/D),Nh]
                h0 = output.view([1,Lp2*B2,Nh])
                
                rnn_input = tmp[N//self.p:].reshape([Lp2*B2,self.p])
                # [?] -> [(L-N)/p,B*L//D]
                logsubsample = self.subsampler.logprobability(rnn_input,h0).view([Lp2,B2])

                #[(L-N)/p,B*L//D] -> [B,L/D]
                                
                #sum over (L-N)/p
                logp=torch.sum(logsubsample,dim=0).view([B,L//D])
                
                #sum over N/p
                logp+=torch.sum(logsubsample0[:N//self.p],dim=0).unsqueeze(-1)
                
                probs[:,N:(k+1)*L//D]=logp
                
        return probs
    
OptionManager.register("lptf",LPTF.DEFAULTS) 

if __name__=="__main__":        
    import sys
    print(sys.argv[1:])
    
    split0=split1=len(sys.argv)
    if "--lptf" in sys.argv[1:]:
        split0=sys.argv.index("--lptf")
    if "--rnn" in sys.argv[1:]:
        split1=sys.argv.index("--rnn")
        SMODEL=PRNN
    if "--ptf" in sys.argv[1:]:
        split1=sys.argv.index("--ptf")
        SMODEL=PTF

        
    #Initialize default options
    train_opt=TrainOpt(K=256,Q=1,dir="LPTF")
    lptf_opt=LPTF.DEFAULTS.copy()
    sub_opt=SMODEL.DEFAULTS.copy()
    
    
    # Update options with command line arguments
    train_opt.apply(sys.argv[1:split0])
    lptf_opt.apply(sys.argv[split0+1:split1])
    sub_opt.apply(sys.argv[split1+1:])
    
    #Add in extra info to options
    lptf_opt.model_name=LPTF.__name__
    sub_opt.model_name=SMODEL.__name__
    train_opt.B=train_opt.K*train_opt.Q
    if SMODEL==PTF:sub_opt.Nh=[sub_opt.Nh,train_opt.Nh]
    
    # Build models
    subsampler = SMODEL(**sub_opt.__dict__)
    lptf = torch.jit.script(LPTF(subsampler,train_opt.L,**lptf_opt.__dict__))    
    
    #Initialize optimizer
    beta1=0.9;beta2=0.999
    optimizer = torch.optim.Adam(
    lptf.parameters(), 
    lr=train_opt.lr, 
    betas=(beta1,beta2)
    )
    
    print(train_opt)
    mydir=setup_dir(train_opt)
    orig_stdout = sys.stdout
    
    full_opt = Options(train=train_opt.__dict__,model=lptf_opt.__dict__,submodel=sub_opt.__dict__)
    full_opt.save(mydir+"\\settings.json")
    
    f = open(mydir+'\\output.txt', 'w')
    sys.stdout = f
    try:
        reg_train(train_opt,(lptf,optimizer),printf=True,mydir=mydir)
    except Exception as e:
        print(e)
        sys.stdout = orig_stdout
        f.close()
        1/0
    sys.stdout = orig_stdout
    f.close()
    