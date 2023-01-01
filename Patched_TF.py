#!/usr/bin/env python
# coding: utf-8

# In[1]:


from RNN_QSR import *


# # Reshaping the 2D sequence tensor into chunks

# In[2]:


def patch(x,Lx):
    # type: (Tensor,int) -> Tensor
    """patch your sequence into chunks of 4"""
    #make the input 2D then break it into 2x2 chunks 
    #afterwards reshape the 2x2 chunks to vectors of size 4 and flatten the 2d bit
    return x.view([x.shape[0],Lx,Lx]).unfold(-2,2,2).unfold(-2,2,2).reshape([x.shape[0],Lx*Lx//4,4])

def unpatch(x,Lx):
    # type: (Tensor,int) -> Tensor
    """inverse function for patch"""
    # original sequence order can be retrieved by chunking twice more
    #in the x-direction you should have chunks of size 2, but in y it should
    #be chunks of size Ly//2
    return x.unfold(-2,Lx//2,Lx//2).unfold(-2,2,2).reshape([x.shape[0],Lx*Lx])


# In[3]:


class PE2D(nn.Module):
    #TODO: Positional encoding is wrong because the spins are at index i+1 when we sample and get probabilities
    def __init__(self, d_model, Lx,Ly,device,n_encode=None):
        
        
        super().__init__()
        assert (d_model%4==0)
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(Lx*Ly, d_model)
        
        if type(n_encode)==type(None):
            n_encode=3*d_model//4
        for pos in range(Lx*Ly):
            x=pos//Ly
            y=pos%Ly
            # Only going to fill 3/4 of the matrix so the
            # occupation values are preserved
            for i in range(0, n_encode, 4):
                
                #x direction encoding
                pe[pos, i] =                 math.sin(x / (10000 ** ((2 * i)/n_encode)))
                pe[pos, i + 1] =                 math.cos(x / (10000 ** ((2 * (i + 1))/n_encode)))
                #y direction encoding
                pe[pos, i+2] =                 math.sin(y / (10000 ** ((2 * i)/n_encode)))
                pe[pos, i + 3] =                 math.cos(y / (10000 ** ((2 * (i + 1))/n_encode)))
                
        self.pe = pe.unsqueeze(1).to(device)
        self.L=Lx*Ly
    
    def forward(self, x):
        return x.repeat(1,1,self.d_model//4) + self.pe[:x.shape[0]]


# In[4]:


@torch.jit.script
def patch2idx(patch):
    #moving the last dimension to the front
    patch=patch.unsqueeze(0).transpose(-1,0).squeeze(-1)
    out=torch.zeros(patch.shape[1:],device=patch.device)
    for i in range(4):
        out+=patch[i]<<i
    return out.to(torch.int64)

@torch.jit.script
def patch2onehot(patch):
    #moving the last dimension to the front
    patch=patch.unsqueeze(0).transpose(-1,0).squeeze(-1)
    out=torch.zeros(patch.shape[1:],device=patch.device)
    for i in range(4):
        out+=patch[i]<<i
    return nn.functional.one_hot(out.to(torch.int64), num_classes=16)


# In[5]:


class PatchTransformerBase(Sampler):#(torch.jit.ScriptModule):
    """
    Base class for the two patch transformer architectures 
    
    Architexture wise this is how a patched transformer works:
    
    You give it a (2D) state and it patches it into groups of 4 (think of a 2x2 cnn filter with stride 2). It then tells you
    the probability of each patch given it and all previous patches in your sequence using masked attention.
    
    Outputs should either be size 1 (the probability of the current patch which is input) or size 16 (for 2x2 patches where 
    the probability represented is of each potential patch)
    
    """
    def __init__(self,Lx,device=device,Nh=128,dropout=0.0,num_layers=2,nhead=8,outsize=1, **kwargs):
        super(PatchTransformerBase, self).__init__()
        #print(nhead)
        self.pe = PE2D(Nh, Lx,Lx,device)
        self.device=device
        #Encoder only transformer
        #misinterperetation on encoder made it so this code does not work
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=Nh, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        
        self.lin = nn.Sequential(
                nn.Linear(Nh,Nh),
                nn.ReLU(),
                nn.Linear(Nh,outsize),
                nn.Sigmoid() if outsize==1 else nn.Softmax(dim=-1)
            )
        
        self.Lx=Lx
        self.set_mask(Lx**2//4)
        
        self.options=torch.zeros([16,4],device=self.device)
        tmp=torch.arange(16,device=self.device)
        for i in range(4):
            self.options[:,i]=(tmp>>i)%2
        
        
        self.to(device)
        
    def set_mask(self, L):
        # type: (int)
        # take the log of a lower triangular matrix
        self.L=L
        self.mask = torch.log(torch.tril(torch.ones([L,L],device=self.device)))
        self.pe.L=L
        
        
    def forward(self, input):
        # input is shape [B,L,1]
        # add positional encoding to get shape [B,L,Nh]
        if input.shape[1]//4!=self.L:
            self.set_mask(input.shape[1]//4)
        #pe should be sequence first [L,B,Nh]
        input=self.pe(patch(input.squeeze(-1),self.Lx).transpose(1,0))
        output = self.transformer(input,self.mask)
        output = self.lin(output.transpose(1,0))
        return output
    
    def next_with_cache(self,tgt,cache=None,idx=-1):
        # type: (Tensor,Optional[Tensor],int) -> Tuple[Tensor,Tensor]
        """Efficiently calculates the next output of a transformer given the input sequence and 
        cached intermediate layer encodings of the input sequence
        
        Inputs:
            tgt - Tensor of shape [L,B,1]
            cache - Tensor of shape ?
            idx - index from which to start
            
        Outputs:
            output - Tensor of shape [?,B,1]
            new_cache - Tensor of shape ?
        """
        #HMMM
        output = tgt
        new_token_cache = []
        #go through each layer and apply self attention only to the last input
        for i,layer in enumerate(self.transformer.layers):
            
            tgt=output
            #have to merge the functions into one
            src = tgt[idx:, :, :]
            mask = None if idx==-1 else self.mask[idx:]

            # self attention part
            src2 = layer.self_attn(
                src,#only do attention with the last elem of the sequence
                tgt,
                tgt,
                attn_mask=mask,  # not needed because we only care about the last token
                key_padding_mask=None,
            )[0]
            #straight from torch transformer encoder code
            src = src + layer.dropout1(src2)
            src = layer.norm1(src)
            src2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(src))))
            src = src + layer.dropout2(src2)
            src = layer.norm2(src)
            #return src
            
            output = src#self.next_attn(output,layer,idx)
            new_token_cache.append(output)
            if cache is not None:
                #layers after layer 1 need to use a cache of the previous layer's output on each input
                output = torch.cat([cache[i], output], dim=0)

        #update cache with new output
        if cache is not None:
            new_cache = torch.cat([cache, torch.stack(new_token_cache, dim=0)], dim=1)
        else:
            new_cache = torch.stack(new_token_cache, dim=0)

        return output, new_cache
    
    def make_cache(self,tgt):
        output = tgt
        new_token_cache = []
        #go through each layer and apply self attention only to the last input
        for i, layer in enumerate(self.transformer.layers):
            output = layer(output,src_mask=self.mask)#self.next_attn(output,layer,0)
            new_token_cache.append(output)
        #create cache with tensor
        new_cache = torch.stack(new_token_cache, dim=0)
        return output, new_cache


# In[7]:


class PatchTransformerB(PatchTransformerBase):
    """Note: logprobability IS normalized 
    
    Architexture wise this is how it works:
    
    You give it a (2D) state and it patches it into groups of 4 (think of a 2x2 cnn filter with stride 2). It then tells you
    the probability of each potential patch given all previous patches in your sequence using masked attention.
    
    
    This model has 16 outputs, which describes the probability distrubition for the nth patch when given the first n-1 patches
    
    """
    def __init__(self,Lx,**kwargs):
        #only important bit is that outsize = 16
        super(PatchTransformerB, self).__init__(Lx,outsize=16,**kwargs)
    @torch.jit.export
    def logprobability(self,input):
        # type: (Tensor) -> Tensor
        """Compute the logscale probability of a given state
            Inputs:
                input - [B,L,1] matrix of zeros and ones for ground/excited states
            Returns:
                logp - [B] size vector of logscale probability labels
        """
        
        if input.shape[1]//4!=self.L:
            self.set_mask(input.shape[1]//4)
        #pe should be sequence first [L,B,Nh]
        
        #shape is modified to [L//4,B,4]
        input = patch(input.squeeze(-1),self.Lx).transpose(1,0)
        
        data=torch.zeros(input.shape,device=self.device)
        data[1:]=input[:-1]
        
        #[L//4,B,4] -> [L//4,B,Nh]
        encoded=self.pe(data)
        #shape is preserved
        output = self.transformer(encoded,self.mask)
        # [L//4,B,Nh] -> [L//4,B,16]
        output = self.lin(output)
        
        #real is going to be a onehot with the index of the appropriate patch set to 1
        #shape will be [L//4,B,16]
        real=patch2onehot(input)
        
        #[L//4,B,16] -> [L//4,B]
        total = torch.sum(real*output,dim=-1)
        #[L//4,B] -> [B]
        logp=torch.sum(torch.log(total+1e-10),dim=0)
        return logp   
    
    @torch.jit.export
    def sample(self,B,L,cache=None):
        # type: (int,int,Optional[Tensor]) -> Tensor
        """ Generates a set states
        Inputs:
            B (int)            - The number of states to generate in parallel
            L (int)            - The length of generated vectors
        Returns:
            samples - [B,L,1] matrix of zeros and ones for ground/excited states
        """
        #length is divided by four due to patching
        L=L//4
        
        #return (torch.rand([B,L,1],device=device)<0.5).to(torch.float32)
        #Sample set will have shape [B,L,1]
        #need one extra zero batch at the start for first pred hence input is [L+1,B,1] 
        input = torch.zeros([L+1,B,4],device=self.device)
         
        with torch.no_grad():
          for idx in range(1,L+1):
            
            #pe should be sequence first [L,B,Nh]
            encoded_input = self.pe(input[:idx,:,:])
                        
            #Get transformer output
            output,cache = self.next_with_cache(encoded_input,cache)
            #check out the probability of all 16 vectors
            probs=self.lin(output[-1,:,:]).view([B,16])

            #sample from the probability distribution
            indices = torch.multinomial(probs,1,False).squeeze(1)
            #extract samples
            sample = self.options[indices]
            
            #set input to the sample that was actually chosen
            input[idx] = sample
            
        #remove the leading zero in the input    
        input=input[1:]
        #sample is repeated 16 times at 3rd index so we just take the first one
        return unpatch(input.transpose(1,0),self.Lx).unsqueeze(-1)
    
    
    @torch.jit.export
    def _off_diag_labels(self,sample,B,L,grad,D=1):
        # type: (Tensor,int,int,bool,int) -> Tuple[Tensor, Tensor]
        """label all of the flipped states  - set D as high as possible without it slowing down runtime
        Parameters:
            sample - [B,L,1] matrix of zeros and ones for ground/excited states
            B,L (int) - batch size and sequence length
            D (int) - Number of partitions sequence-wise. We must have L%D==0 (D divides L)
            
        Outputs:
            
            sample - same as input
            probs - [B,L] matrix of probabilities of states with the jth excitation flipped
        """
        
        
        
        sample0=sample
        #sample is batch first at the moment
        sample = patch(sample.squeeze(-1),self.Lx)
        
        sflip = torch.zeros([B,L,L//4,4],device=self.device)
        #collect all of the flipped states into one array
        for j in range(L//4):
            #have to change the order of in which states are flipped for the cache to be useful
            for j2 in range(4):
                sflip[:,j*4+j2] = sample*1.0
                sflip[:,j*4+j2,j,j2] = 1-sflip[:,j*4+j2,j,j2]
            
        #switch sample into sequence-first
        sample = sample.transpose(1,0)
            
        #compute all of their logscale probabilities
        with torch.no_grad():
            

            data=torch.zeros(sample.shape,device=self.device)
            data[1:]=sample[:-1]
            
            #[L//4,B,4] -> [L//4,B,Nh]
            encoded=self.pe(data)
            
            #add positional encoding and make the cache
            out,cache=self.make_cache(encoded)

            probs=torch.zeros([B,L],device=self.device)
            #expand cache to group L//D flipped states
            cache=cache.unsqueeze(2)

            #this line took like 1 hour to write I'm so sad
            #the cache has to be shaped such that the batch parts line up
                        
            cache=cache.repeat(1,1,L//D,1,1).transpose(2,3).reshape(cache.shape[0],L//4,B*L//D,cache.shape[-1])

            pred0 = self.lin(out)
            #shape will be [L//4,B,16]
            real=patch2onehot(sample)
            #[L//4,B,16] -> [B,L//4]
            total0 = torch.sum(real*pred0,dim=-1).transpose(1,0)

            for k in range(D):

                N = k*L//D
                #next couple of steps are crucial          
                #get the samples from N to N+L//D
                #Note: samples are the same as the original up to the Nth spin
                real = sflip[:,N:(k+1)*L//D]
                #flatten it out and set to sequence first
                tmp = real.reshape([B*L//D,L//4,4]).transpose(1,0)
                #set up next state predction
                fsample=torch.zeros(tmp.shape,device=self.device)
                fsample[1:]=tmp[:-1]
                # put sequence before batch so you can use it with your transformer
                tgt=self.pe(fsample)
                #grab your transformer output
                out,_=self.next_with_cache(tgt,cache[:,:N//4],N//4)

                # grab output for the new part
                output = self.lin(out[N//4:].transpose(1,0))
                # reshape output separating batch from spin flip grouping
                pred = output.view([B,L//D,(L-N)//4,16])
                real = patch2onehot(real[:,:,N//4:])
                total = torch.sum(real*pred,dim=-1)
                #sum across the sequence for probabilities
                
                #print(total.shape,total0.shape)
                logp=torch.sum(torch.log(total+1e-10),dim=-1)
                logp+=torch.sum(torch.log(total0[:,:N//4]+1e-10),dim=-1).unsqueeze(-1)
                probs[:,N:(k+1)*L//D]=logp
                
        return sample0,probs
    @torch.jit.export
    def sample_with_labelsALT(self,B,L,grad=False,nloops=1):
        # type: (int,int,bool,int) -> Tuple[Tensor,Tensor,Tensor]
        sample,probs = self.sample_with_labels(B,L,grad,nloops)
        logsqrtp=probs.mean(dim=1)/2
        sumsqrtp = torch.exp(probs/2-logsqrtp.unsqueeze(1)).sum(dim=1)
        return sample,sumsqrtp,logsqrtp
    @torch.jit.export
    def sample_with_labels(self,B,L,grad=False,nloops=1):
        # type: (int,int,bool,int) -> Tuple[Tensor,Tensor]
        sample=self.sample(B,L,None)
        return self._off_diag_labels(sample,B,L,grad,nloops)


# In[8]:


if __name__=="__main__":
    
    import sys
    print(sys.argv[1:])
    op=Opt()
    op.apply(sys.argv[1:])
    op.B=op.K*op.Q
    print(op)
    Lx=int(op.L**0.5)
    trainsformer = torch.jit.script(PatchTransformerB(Lx,Nh=op.Nh,num_layers=2))
    sampleformer = torch.jit.script(PatchTransformerB(Lx,Nh=op.Nh,num_layers=2))
    beta1=0.9;beta2=0.999
    optimizer = torch.optim.Adam(
    trainsformer.parameters(), 
    lr=op.lr, 
    betas=(beta1,beta2)
    )
    if op.USEQUEUE:
        queue_train(op,(trainsformer,sampleformer,optimizer))
    else:
        print("Training. . .")
        reg_train(op,(trainsformer,optimizer))



