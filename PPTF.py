from PTF import *


import copy

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class PPTF(Sampler):
    """
    
    Transformer but old pxp patches are condensed into one token in a pyramid transformer scheme
    
    """
    def __init__(self,L,p,_2D=False,device=device,Nh=128,dropout=0.0,num_layers=2,nhead=8, **kwargs):
        super(PPTF, self).__init__()
        #print(nhead)

        if _2D:
            self.patch=Patch2D(p,L)
            self.L = int(L**2//p**2)
            self.p=int(p**2)
            #positional encodings and minipatches
            self.minipatch = Patch2D(2,p)
            self.pe = PE2D(Nh, L//2,L//2,device).pe
        else:
            
            self.patch=Patch1D(p,L)
            self.L = int(L//p)
            self.p = int(p)
            self.minipatch = Patch2D(4,p)
            self.pe = PE1D(Nh,L//4,device).pe
            
        self.device=device

        #Encoder only transformer
        #misinterperetation on encoder made it so this code does not work
        self.transformer = FastMaskedTransformerEncoder(Nh=Nh,dropout=dropout,num_layers=num_layers,nhead=nhead)       
        
        
        self.tokenize=nn.Linear(4,Nh)
        
        SR = nn.Sequential(
                nn.Flatten(start_dim=-2),
                nn.Linear(Nh*self.p//4,Nh),
                nn.LayerNorm(Nh)
        )
        
        self.SR = _get_clones(SR,num_layers)
        
        self.lin = nn.Sequential(
                nn.Linear(Nh,Nh),
                nn.ReLU(),
                nn.Linear(Nh,1<<4),
                nn.Softmax(dim=-1)
            )
        
        #self.lin0,self.lin1=self.lin[:2],self.lin[2:] 
        
        self.set_mask(self.L)

        
        #create a tensor of all possible patches
        self.options=torch.zeros([1<<4,4],device=self.device)
        tmp=torch.arange(1<<4,device=self.device)
        for i in range(4):
            self.options[:,i]=(tmp>>i)%2
        
        self.setpe_L()
        
        self.to(device)
        
    def set_mask(self, L):
        # type: (int)
        """Initialize the self-attention mask"""
        # take the log of a lower triangular matrix
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

        B = input.shape[0]
        
        #print(B,self.L)
        
        logp = torch.zeros([B],device=self.device)
        #shape is modified to [L//p,B,p]
        input = self.patch(input.squeeze(-1)).transpose(1,0)        

        bigCache = torch.zeros([self.transformer.nl,0,B,self.transformer.Nh],device=self.device)
        squished_input = torch.zeros([0,B,self.transformer.Nh],device=self.device)
        zero_token = torch.zeros([B,self.transformer.Nh],device=self.device)
        
        minip=self.p//4
        
        for i in range(self.L):
            
            cache = bigCache
            
            sample = torch.zeros([minip+1,B,4],device=self.device)
            
            sample[1:] = self.minipatch(input[i]).transpose(1,0)
            
            if i==self.L-1:
                encoded_input = self.tokenize(sample[:-1])+self.pe[i*minip:(i+1)*minip+1] 
                tokens = torch.cat([squished_input, encoded_input], dim=0)
                self.transformer.set_mask(tokens.shape[0])
                output,cache = self.transformer.next_with_cache(tokens,cache,-encoded_input.shape[0])
                output = self.lin(output[-encoded_input.shape[0]:])
            else:
                encoded_input = self.tokenize(sample)+self.pe[i*minip:(i+1)*minip+1]
                tokens = torch.cat([squished_input, encoded_input], dim=0)
                self.transformer.set_mask(tokens.shape[0])
                output,cache = self.transformer.next_with_cache(tokens,cache,-encoded_input.shape[0])
                output = self.lin(output[-encoded_input.shape[0]:-1])
            
            
            #real is going to be a onehot with the index of the appropriate patch set to 1
            #shape will be [L//p,B,2^p]
            real=genpatch2onehot(sample[1:],4)
                        
                
            #[L//p,B,2^p] -> [L//p,B]
            total = torch.sum(real*output,dim=-1)
            #[L//p,B] -> [B]
            logp+=torch.sum(torch.log(total),dim=0)
            
            
            new_token_cache = []
            #squish newly generated tokens into one token
            for si,SR in enumerate(self.SR):
                #[nl,?,B,Nh] -> [B,Nh]
                if si==0:
                    squish_token = SR(encoded_input[-minip:].transpose(1,0))
                    squished_input = torch.cat([squished_input,squish_token.unsqueeze(0)],dim=0)
                else:
                    squish_token = SR(cache[si-1,-minip:].transpose(1,0))
                    new_token_cache.append(squish_token)

            new_token_cache.append(zero_token)
            #add shape [nl,1,B,Nh]
            bigCache = torch.cat([bigCache,torch.stack(new_token_cache, dim=0).unsqueeze(1)],dim=1)
            
            
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
        
        
        #return (torch.rand([B,L,1],device=device)<0.5).to(torch.float32)
        #Sample set will have shape [L/p,B,p]
        #need one extra zero batch at the start for first pred hence input is [L+1,B,1] 
        input = torch.zeros([L,B,self.p],device=self.device)

        logp = torch.zeros([B],device=self.device)
        
        #make cache initially an empty tensor
        bigCache = torch.zeros([self.transformer.nl,0,B,self.transformer.Nh],device=self.device)
        
        squished_input = torch.zeros([0,B,self.transformer.Nh],device=self.device)
        
        zero_token = torch.zeros([B,self.transformer.Nh],device=self.device)
        
        #self.DEBUG_st = []
        
        minip=self.p//4
        #with torch.no_grad():
        for idx in range(L):
            
            sample = torch.zeros([minip+1,B,4],device=self.device)
            cache = bigCache
            offset=idx*minip
            for minidx in range(1,minip+1):

                #pe should be sequence first [l,B,Nh]
                # multiply by 1 to copy the tensor
                
                encoded_input = self.tokenize(sample[:minidx,:,:]*1)+self.pe[offset:offset+minidx]
                
                tokens = torch.cat([squished_input, encoded_input], dim=0)
                
                #Get transformer output
                output,cache = self.transformer.next_with_cache(tokens,cache)
                probs=self.lin(output[-1,:,:]).view([B,1<<4])

                #sample from the probability distribution
                indices = torch.multinomial(probs,1,False).squeeze(1)
                #extract samples
                sample[minidx] = self.options[indices]

                onehot = nn.functional.one_hot(indices, num_classes=1<<4)
                logp+= torch.log(torch.sum(onehot*probs,dim=-1))
            
            new_token_cache = []
            
            if idx<L-1:
                
                #make cache for the last sample in the large patch
                
                encoded_input = self.tokenize(sample*1)+self.pe[offset:offset+minip+1] 
                                
                tokens = torch.cat([squished_input, encoded_input], dim=0)
                output,cache = self.transformer.next_with_cache(tokens,cache)
                
                #self.DEBUG_st += [tokens]

                #squish the large patch into a single token
                for si,SR in enumerate(self.SR):
                    #[nl,?,B,Nh] -> [B,Nh]
                    
                    if si==0:
                        squish_token = SR(encoded_input[-minip:].transpose(1,0))
                        squished_input = torch.cat([squished_input,squish_token.unsqueeze(0)],dim=0)
                    else:
                        squish_token = SR(cache[si-1,-minip:].transpose(1,0))
                        new_token_cache.append(squish_token)
                        
                new_token_cache.append(zero_token)
                #add shape [nl,1,B,Nh]
                bigCache = torch.cat([bigCache,torch.stack(new_token_cache, dim=0).unsqueeze(1)],dim=1)
            
            #set input to the sample that was actually chosen
            input[idx] = self.minipatch.reverse(sample[1:].transpose(1,0))
        
        
        #remove the leading zero in the input    
        #sample is repeated 16 times at 3rd index so we just take the first one
        return self.patch.reverse(input.transpose(1,0)).unsqueeze(-1),logp
    
    
    def setpe_L(self):
        self.pe_L = torch.zeros([self.p//4+1,self.L,1,self.transformer.Nh],device=self.device)
        minip=self.p//4
        for i in range(self.L-1):
            self.pe_L[:,i]=self.pe[i*minip:(i+1)*minip+1] 
            
        self.pe_L[:-1,-1]=self.pe[-minip:]
        self.B=0
        
        #(N⋅num_heads,Q,K), where N is the batch size,
        #Q is the target sequence length, (number of queries)
        #K is the source sequence length. (number of keys)
        # For a binary mask, a True value indicates that the corresponding position is not allowed to attend. 
        
        #start with everything being allowed to attend
        self.mask0 = torch.zeros([self.L,self.p//4+1,self.L+self.p//4+1],dtype=torch.bool,device=self.device)
        
        for i in range(self.L):
            for k in range(self.L):
                #zeroth dim can only attend to previous ones in 2rd dim (up to L)
                #This can be considered attention between the big patches
                if i<=k:
                    self.mask0[i,:,k]=True
                    
        for j in range(self.p//4+1): 
            for n in range(self.p//4+1):
                #first dim can only attend to previous or equal ones in 2rd dim (after L)
                #this can be considered attention between small patches
                if j<n:
                    self.mask0[:,j,self.L+n]=True
        
    def set_batch_mask(self,B):
        self.B=B
        self.mask = self.mask0.unsqueeze(1).unsqueeze(2)
        
        head = self.transformer.nhead
        
        self.mask=self.mask.repeat(1,B,head,1,1)
        
        self.mask=self.mask.reshape([self.L*head*B,self.p//4+1,self.L+self.p//4+1])
        
        
    #@torch.jit.export
    def logprobability2(self,input):
        # type: (Tensor) -> Tensor
        """Compute the logscale probability of a given state
            Inputs:
                input - [B,L,1] matrix of zeros and ones for ground/excited states
            Returns:
                logp - [B] size vector of logscale probability labels
        """
        
        
        
        #shape is modified to [L//p,B,p]
        input = self.patch(input.squeeze(-1)).transpose(1,0)
        Lp,B,p=input.shape
        
        if B!= self.B:
            self.set_batch_mask(B)
        
        logp = torch.zeros([B],device=self.device)
            
        #[p/4,Lp,B,4]    
        patched = self.minipatch(input.reshape([Lp*B,p])).transpose(1,0).view([p//4,Lp,B,4])
        
        
        
        out = torch.zeros([p//4+1,Lp,B,4],device=self.device)
        
        out[1:]=patched
        #[p//4+1,Lp,B,Nh]
        out = self.tokenize(out)+self.pe_L
            
        #[B,Lp,p//4,Nh]        
        #self.DEBUG_st2 = []
            
        for i in range(len(self.transformer.transformer.layers)):
            #[p//4+1,Lp,B,Nh] -> [B,Lp,p//4,Nh] -> [B,Lp,Nh]
            squishedtokens = self.SR[i](out[1:].transpose(0,2))
                        
                   
            #[B,Lp,Nh] -> [Lp,Lp,B,Nh]
            squish_sequence = squishedtokens.transpose(1,0).unsqueeze(1).repeat(1,Lp,1,1)
        
            full_sequence = torch.cat([squish_sequence,out],dim=0)
            fs = full_sequence.reshape([full_sequence.shape[0],Lp*B,full_sequence.shape[-1]])
        
            layer=self.transformer.transformer.layers[i]
            
            #self.DEBUG_st2+=[fs]
            
            src = out.view(out.shape[0],Lp*B,out.shape[-1])
            # self attention part
            
            src2 = layer.self_attn(
                src,
                fs,
                fs,
                attn_mask=self.mask,  
                key_padding_mask=None,
            )[0]
            #straight from torch transformer encoder code
            src = src + layer.dropout1(src2)
            src = layer.norm1(src)
            src2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(src))))
            src = src + layer.dropout2(src2)
            src = layer.norm2(src)
            #return src
            out = src.view([p//4+1,Lp,B,out.shape[-1]])
        
        #[p/4,Lp,B,16]
        output = self.lin(out[:-1])
        
        #[p/4,Lp,B,4] -> [p/4,Lp,B,16]
        real=genpatch2onehot(patched,4)
        
        #[p/4,Lp,B,16] -> [p/4,Lp,B]
        total = torch.sum(real*output,dim=-1)
        

        logp = torch.sum(torch.log(total),dim=(0,1))
     
        return logp
    
    
    @torch.jit.export
    def logprobability3(self,input):
        # type: (Tensor) -> Tensor
        """Compute the logscale probability of a given state
            Inputs:
                input - [B,L,1] matrix of zeros and ones for ground/excited states
            Returns:
                logp - [B] size vector of logscale probability labels
        """
   
        #print(B,self.L)
        input = self.patch(input.squeeze(-1)).transpose(1,0) 
        Lp,B,p=input.shape
        
        logp = torch.zeros([B],device=self.device)
        #shape is modified to [L//p,B,p]
         
        
        
        #[p/4,Lp,B,4]    
        patched = self.minipatch(input.reshape([Lp*B,p])).transpose(1,0).view([p//4,Lp,B,4])
        out = torch.zeros([p//4+1,Lp,B,4],device=self.device)
        out[1:]=patched
        

        #[p//4+1,Lp,B,Nh]
        all_tokens = self.tokenize(out)+self.pe_L

        #[p//4+1,Lp,B,Nh] -> [B,Lp,p//4,Nh] -> [B,Lp,Nh] -> [Lp,B,Nh]
        squished_input = self.SR[0](all_tokens[1:].transpose(0,2)).transpose(1,0)

        bigCache = torch.zeros([self.transformer.nl,0,B,self.transformer.Nh],device=self.device)

        zero_token = torch.zeros([B,self.transformer.Nh],device=self.device)
        
        minip=self.p//4
        
        for i in range(self.L):
            
            cache = bigCache
            
            #[p//4+1,Lp,B,Nh] -> [p//4+1,B,Nh]
            encoded_input = all_tokens[:,i]
            tokens = torch.cat([squished_input[:i], encoded_input], dim=0)
            self.transformer.set_mask(tokens.shape[0])
            output,cache = self.transformer.next_with_cache(tokens,cache,-encoded_input.shape[0])
            output = self.lin(output[-encoded_input.shape[0]:-1])
            
            #real is going to be a onehot with the index of the appropriate patch set to 1
            #shape will be [p//4,B,2^4]
            real=genpatch2onehot(patched[:,i],4)
            #[L//p,B,2^p] -> [L//p,B]
            total = torch.sum(real*output,dim=-1)
            #[L//p,B] -> [B]
            logp+=torch.sum(torch.log(total),dim=0)
            
            new_token_cache = []
            #squish newly generated tokens into one token
            for si,SR in enumerate(self.SR):
                #[nl,?,B,Nh] -> [B,Nh]
                if si!=0:
                    squish_token = SR(cache[si-1,-minip:].transpose(1,0))
                    new_token_cache.append(squish_token)

            new_token_cache.append(zero_token)
            #add shape [nl,1,B,Nh]
            bigCache = torch.cat([bigCache,torch.stack(new_token_cache, dim=0).unsqueeze(1)],dim=1)
            
            
        return logp
    
    
    
    
    
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
        probs=torch.zeros([B,L],device=self.device)
        
        
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
            

            #print(B,self.L)

            logp0 = torch.zeros([self.L,B],device=self.device)
            #shape is modified to [L//p,B,p]
            input = sample        

            bigCache = torch.zeros([self.transformer.nl,0,B,self.transformer.Nh],device=self.device)
            squished_input = torch.zeros([0,B,self.transformer.Nh],device=self.device)
            zero_token = torch.zeros([B,self.transformer.Nh],device=self.device)

            minip=self.p//4

            for i in range(self.L):

                cache = bigCache

                sample = torch.zeros([minip+1,B,4],device=self.device)

                sample[1:] = self.minipatch(input[i]).transpose(1,0)

                if i==self.L-1:
                    encoded_input = self.tokenize(sample[:-1])+self.pe[i*minip:(i+1)*minip+1] 
                    tokens = torch.cat([squished_input, encoded_input], dim=0)
                    self.transformer.set_mask(tokens.shape[0])
                    output,cache = self.transformer.next_with_cache(tokens,cache,-encoded_input.shape[0])
                    output = self.lin(output[-encoded_input.shape[0]:])
                else:
                    encoded_input = self.tokenize(sample)+self.pe[i*minip:(i+1)*minip+1]
                    tokens = torch.cat([squished_input, encoded_input], dim=0)
                    self.transformer.set_mask(tokens.shape[0])
                    output,cache = self.transformer.next_with_cache(tokens,cache,-encoded_input.shape[0])
                    output = self.lin(output[-encoded_input.shape[0]:-1])


                #real is going to be a onehot with the index of the appropriate patch set to 1
                #shape will be [L//p,B,2^p]
                real=genpatch2onehot(sample[1:],4)


                #[L//p,B,2^p] -> [L//p,B]
                total = torch.sum(real*output,dim=-1)
                #[L//p,B] -> [B]
                logp0[i]=torch.sum(torch.log(total),dim=0)


                new_token_cache = []
                #squish newly generated tokens into one token
                for si,SR in enumerate(self.SR):
                    #[nl,?,B,Nh] -> [B,Nh]
                    if si==0:
                        squish_token = SR(encoded_input[-minip:].transpose(1,0))
                        squished_input = torch.cat([squished_input,squish_token.unsqueeze(0)],dim=0)
                    else:
                        squish_token = SR(cache[si-1,-minip:].transpose(1,0))
                        new_token_cache.append(squish_token)

                new_token_cache.append(zero_token)
                #add shape [nl,1,B,Nh]
                bigCache = torch.cat([bigCache,torch.stack(new_token_cache, dim=0).unsqueeze(1)],dim=1)
            
            
            #get the big cache in there
            bigCache0 = bigCache.unsqueeze(2)
            bigCache0=bigCache0.repeat(1,1,L//D,1,1).transpose(2,3).reshape(bigCache0.shape[0],L//self.p,B*L//D,bigCache0.shape[-1])
            
            #get the squished input in there
            squished_input0 = squished_input.unsqueeze(1)
            squished_input0 = squished_input0.repeat(1,L//D,1,1).transpose(1,2).reshape(L//self.p,B*L//D,squished_input0.shape[-1])
            
            zero_token = torch.zeros([B*L//D,self.transformer.Nh],device=self.device)
            
            for k in range(D):

                N = k*L//D
                #next couple of steps are crucial          
                #get the samples from N to N+L//D
                #Note: samples are the same as the original up to the Nth spin
                real = sflip[:,N:(k+1)*L//D]
                #flatten it out and set to sequence first
                tmp = real.reshape([B*L//D,L//self.p,self.p]).transpose(1,0)
                #set up next state predction

                #reuse old cache
                bigCache       = bigCache0[:,:N//self.p]
                squished_input = squished_input0[:N//self.p]
                
                logp = torch.zeros([B*L//D],device=self.device)
                
                for i in range(N//self.p,self.L):

                    cache = bigCache
                    sample = torch.zeros([minip+1,B*L//D,4],device=self.device)
                    sample[1:] = self.minipatch(tmp[i]).transpose(1,0)

                    if i==self.L-1:
                        encoded_input = self.tokenize(sample[:-1])+self.pe[i*minip:(i+1)*minip+1] 
                        tokens = torch.cat([squished_input, encoded_input], dim=0)
                        self.transformer.set_mask(tokens.shape[0])
                        output,cache = self.transformer.next_with_cache(tokens,cache,-encoded_input.shape[0])
                        output = self.lin(output[-encoded_input.shape[0]:])
                    else:
                        encoded_input = self.tokenize(sample)+self.pe[i*minip:(i+1)*minip+1]
                        tokens = torch.cat([squished_input, encoded_input], dim=0)
                        self.transformer.set_mask(tokens.shape[0])
                        output,cache = self.transformer.next_with_cache(tokens,cache,-encoded_input.shape[0])
                        output = self.lin(output[-encoded_input.shape[0]:-1])


                    #real is going to be a onehot with the index of the appropriate patch set to 1
                    #shape will be [L//p,B,2^p]
                    real=genpatch2onehot(sample[1:],4)


                    #[L//p,B,2^p] -> [L//p,B]
                    total = torch.sum(real*output,dim=-1)
                    #[L//p,B] -> [B]
                    logp+=torch.sum(torch.log(total),dim=0)


                    new_token_cache = []
                    #squish newly generated tokens into one token
                    for si,SR in enumerate(self.SR):
                        #[nl,?,B,Nh] -> [B,Nh]
                        if si==0:
                            squish_token = SR(encoded_input[-minip:].transpose(1,0))
                            squished_input = torch.cat([squished_input,squish_token.unsqueeze(0)],dim=0)
                        else:
                            squish_token = SR(cache[si-1,-minip:].transpose(1,0))
                            new_token_cache.append(squish_token)

                    new_token_cache.append(zero_token)
                    #add shape [nl,1,B,Nh]
                    bigCache = torch.cat([bigCache,torch.stack(new_token_cache, dim=0).unsqueeze(1)],dim=1)
                
                

                #[(L-N)/p,B*L//D] -> [B,L/D]
                                
                #sum over (L-N)/p
                logp=logp.view([B,L//D])
                
                #sum over N/p
                logp+=torch.sum(logp0[:N//self.p],dim=0).unsqueeze(-1)
                
                probs[:,N:(k+1)*L//D]=logp
                
        return probs
    