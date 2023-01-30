from PTF import *
class PTFB(Sampler):
    """Note: logprobability IS normalized 
    
    Architexture wise this is how it works:
    
    You give it a (2D) state and it patches it into groups of 4 (think of a 2x2 cnn filter with stride 2). It then tells you
    the probability of each potential patch given all previous patches in your sequence using masked attention.
    
    This model has 16 outputs, which describes the probability distrubition for the nth patch when given the first n-1 patches
    
    """
    def __init__(self,L,p,_2D=False,device=device,Nh=128,dropout=0.0,num_layers=2,nhead=8, **kwargs):
        super(Sampler, self).__init__()
        #print(nhead)
        
        if type(Nh) is int:
            Nh = [Nh]*4
        else:
            Nh+=[int(L**2//p**2)*Nh[0]] if _2D else [int(L//p)*Nh[0]]
        
        if _2D:
            self.pe = PE2D(Nh[0], L//p,L//p,device)
            self.patch=Patch2D(p,L)
            self.L = int(L**2//p**2)
            self.p=int(p**2)
        else:
            self.pe = PE1D(Nh[0],L//p,device)
            self.patch=Patch1D(p,L)
            self.L = int(L//p)
            self.p = int(p)
            
        self.device=device

        #Encoder only transformer
        #misinterperetation on encoder made it so this code does not work
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=Nh[0], nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)   
        self.transformer.Nh=Nh[0]
        
        self.lin = nn.Sequential(
                nn.Linear(Nh[1],Nh[2]),
                nn.ReLU(),
                nn.Linear(Nh[0],1<<self.p),
                nn.Softmax(dim=-1)
            )
        
        self.lin0,self.lin1=self.lin[:2],self.lin[2:]
        
        
        self.set_mask(self.L)

        
        #create a tensor of all possible patches
        self.options=torch.zeros([1<<self.p,self.p],device=self.device)
        tmp=torch.arange(1<<self.p,device=self.device)
        for i in range(self.p):
            self.options[:,i]=(tmp>>i)%2
        
        
        self.to(device)
        
    def set_mask(self, L):
        # type: (int)
        """Initialize the self-attention mask"""
        # take the log of a lower triangular matrix
        self.L=L
        self.mask = torch.log(torch.tril(torch.ones([L,L],device=self.device))) 
        self.pe.L=L

    @torch.jit.export
    def logprobability(self,input,h0=None):
        # type: (Tensor,Optional[Tensor]) -> Tensor
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
        encoded=self.pe(data)
 
        if h0 is not None:
            
            L,B,Nh=encoded.shape
            h0 = self.lin0(h0)
            #[1,B,Nh0] -> [L,B,Nh]
            h=h0.reshape([1,B,Nh,L]).transpose(-1,0).squeeze(-1)
            #output is shape [L//p,B,Nh]
            output = self.transformer(tgt=encoded, memory=h, tgt_mask=self.mask)
            #output,_ = self.transformer.cross_with_cache(encoded,encoded,encoded,None,0)
            output=self.lin1(output)          
        else:
            #shape is preserved
            output = self.transformer(tgt=encoded, memory=encoded, tgt_mask=self.mask)
            # [L//p,B,Nh] -> [L//p,B,2^p]
            output = self.lin(output)
        
        
        
        
        #real is going to be a onehot with the index of the appropriate patch set to 1
        #shape will be [L//p,B,2^p]
        real=genpatch2onehot(input,self.p)
        
        #[L//p,B,2^p] -> [L//p,B]
        total = torch.sum(real*output,dim=-1)
        #[L//p,B] -> [B]
        logp=torch.sum(torch.log(total),dim=0)
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
        
        if h0 is not None:
            #h0 is shape [1,B,Nh0], Nh0=L*Nh
            h0 = self.lin0(h0)
            #[1,B,Nh0] -> [L,B,Nh]
            h0=h0.reshape([1,B,self.transformer.Nh,L]).transpose(-1,0).squeeze(-1)
        
        #return (torch.rand([B,L,1],device=device)<0.5).to(torch.float32)
        #Sample set will have shape [L/p,B,p]
        #need one extra zero batch at the start for first pred hence input is [L+1,B,1] 
        input = torch.zeros([L+1,B,self.p],device=self.device)

        logp = torch.zeros([B],device=self.device)
        
        #make cache initially an empty tensor
        #cache = torch.zeros([self.transformer.nl,0,B,self.transformer.Nh],device=self.device)
        
        #with torch.no_grad():
        for idx in range(1,L+1):
            
            #pe should be sequence first [l,B,Nh]
            # multiply by 1 to copy the tensor
            encoded_input = self.pe(input[:idx,:,:]*1)
                        
            
            #check out the probability of all 16 vectors
            if h0 is not None:
                h = h0[:idx,:,:]
                #output is shape [?,B,Nh]
                output = self.transformer(tgt=encoded_input, memory=h, tgt_mask=self.mask[:idx,:idx])
                #output,cache = self.transformer.cross_with_cache(encoded_input,encoded_input,encoded_input,cache)
                probs=self.lin1(output[-1,:,:]).view([B,1<<self.p])
            else:
                #Get transformer output
                output = self.transformer(tgt=encoded_input, memory=encoded_input, tgt_mask=self.mask[:idx,:idx])
                probs=self.lin(output[-1,:,:]).view([B,1<<self.p])

            #sample from the probability distribution
            indices = torch.multinomial(probs,1,False).squeeze(1)
            #extract samples
            sample = self.options[indices]
            
            onehot = nn.functional.one_hot(indices, num_classes=1<<self.p)
            logp+= torch.log(torch.sum(onehot*probs,dim=-1))
            
            
            #set input to the sample that was actually chosen
            input[idx] = sample
            
        #remove the leading zero in the input    
        input=input[1:]
        #sample is repeated 16 times at 3rd index so we just take the first one
        return self.patch.reverse(input.transpose(1,0)).unsqueeze(-1),logp
    
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
                probs[k*B*L//D:(k+1)*B*L//D] = self.logprobability(tmp[k*B*L//D:(k+1)*B*L//D],None)

        return probs.reshape([B,L])