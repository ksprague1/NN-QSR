from RNN_QSR import *

class PE2D(nn.Module):
    """Positional Encoder for a 2D sequence"""
    #TODO: Positional encoding is wrong because the spins are at index i+1 when we sample and get probabilities
    def __init__(self, d_model, Lx,Ly,device,n_encode=None):
        """
        Inputs:
            Lx (int) -- Size in the x dimension (axis 1)
            Ly (int) -- Size in the y dimension (axis 2)
        """
        super().__init__()
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
                pe[pos, i] = \
                math.sin(x / (10000 ** ((2 * i)/n_encode)))
                pe[pos, i + 1] = \
                math.cos(x / (10000 ** ((2 * (i + 1))/n_encode)))
                #y direction encoding
                pe[pos, i+2] = \
                math.sin(y / (10000 ** ((2 * i)/n_encode)))
                pe[pos, i + 3] = \
                math.cos(y / (10000 ** ((2 * (i + 1))/n_encode)))
                
        self.pe = pe.unsqueeze(0).to(device)
        self.L=Lx*Ly
    
    def forward(self, x):
        """Adds a positional encoding (batch first)"""
        return x + self.pe[:,:self.L,:]
    
    
    
class SlowTransformer(Sampler):
    """A transformer sampler which uses masked attention to calculate probabilities of a given state.
    Here each 'spin' is given a positional encoding and self-attention is calculated across each spin and all  previous spins
    """
    def __init__(self,Lx,Ly,device=device,Nh=128,decoder=False,dropout=0.0,num_layers=3,nhead=8, **kwargs):
        """
        Parameters:
            Lx,Ly (int) -- Sequence dimensions
            Nh (int) -- size of the input vector at each sequence element (same as d_model)
            decoder (bool) -- whether to use a TF decoder or encoder. Using a decoder isn't currently implemented
        """
        super(SlowTransformer, self).__init__(device=device)
        
        self.pe = PE2D(Nh, Lx,Ly,device)
        
        if decoder:
            #Decoder only transformer
            self.decoder_layer = nn.TransformerDecoderLayer(d_model=Nh, nhead=nhead, dropout=dropout)
            self.transformer = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        else:
            #Encoder only transformer
            #misinterperetation on encoder made it so this code does not work
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=Nh, nhead=nhead, dropout=dropout)
            self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        
        self.lin = nn.Sequential(
                nn.Linear(Nh,Nh),
                nn.ReLU(),
                nn.Linear(Nh,1),
                nn.Sigmoid()
            )
        
        
        self.set_mask(Lx*Ly)
        self.to(device)
        
    def set_mask(self, L):
        """Initialize the self-attention mask"""
        # take the log of a lower triangular matrix
        self.L=L
        self.mask = torch.log(torch.tril(torch.ones([L,L],device=self.device)))
        self.pe.L=L

    def forward(self, input):
        
        # input is shape [B,L,1]
        # add positional encoding to get shape [B,L,Nh]
        if input.shape[1]!=self.L:
            self.set_mask(input.shape[1])
        
        input=self.pe(input).transpose(1,0)
        output = self.transformer(input,self.mask)
        output = self.lin(output.transpose(1,0))
        return output
    
    
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
        
        #Sample set will have shape [B,L,1]
        #need one extra zero batch at the start for first pred hence input is [N,L+1,1] 
        input = torch.zeros([B,L+1,1],device=self.device)
        
        self.set_mask(L)
        #sampling can be done without gradients
        with torch.no_grad():
          for idx in range(1,L+1):
            #run the rnn on shape [B,1,1]   
            #encode the input to the proper shape
            encoded_input = input[:,:idx,:]+self.pe.pe[:,:idx,:]
                        
            #Get transformer output
            output = self.transformer(encoded_input.transpose(1,0),self.mask[:idx,:idx])
            #if probs[i]=1 then there should be a 100% chance that sample[i]=1
            #if probs[i]=0 then there should be a 0% chance that sample[i]=1
            #stands that we generate a random uniform u and take int(u
            probs=self.lin(output.transpose(1,0)[:,-1,:])
            sample = (torch.rand([B,1],device=device)<probs).to(torch.float32)
            input[:,idx,:]=sample
        #input's first entry is zero to get a predction for the first atom
        return input[:,1:,:]
    
    
    
    
class FastTransformer(SlowTransformer):
    """Same architecture as SlowTransformer (weights can be shared) but with improvements to sampling and labelling which
    can give a 2x performance boost"""
    def _off_diag_labels(self,sample,B,L,grad,D=1): #_off_diag_labels
        """label all of the flipped states  - set D as high as possible without it slowing down runtime
        Parameters:
            sample - [B,L,1] matrix of zeros and ones for ground/excited states
            B,L (int) - batch size and sequence length
            D (int) - Number of partitions sequence-wise. We must have L%D==0 (D divides L)
            
        Outputs:
            
            sample - same as input
            probs - [B,L] matrix of probabilities of states with the jth excitation flipped
        """
        sflip = torch.zeros([B,L,L,1],device=self.device)
        #collect all of the flipped states into one array
        for j in range(L):
            #get all of the states with one spin flipped
            sflip[:,j] = sample*1.0
            sflip[:,j,j] = 1-sflip[:,j,j]
        #compute all of their logscale probabilities
        with torch.no_grad():
            #prepare sample to be used as cache
            B,L,one=sample.shape
            dsample=torch.zeros([B,L,one],device=self.device)
            dsample[:,1:,:]=sample[:,:-1,:]

            #add positional encoding and make the cache
            out,cache=self.make_cache(self.pe(dsample).transpose(1,0))

            probs=torch.zeros([B,L],device=self.device)
            #expand cache to group L//D flipped states
            cache=cache.unsqueeze(2)

            #this line took like 1 hour to write I'm so sad
            #the cache has to be shaped such that the batch parts line up
            cache=cache.repeat(1,1,L//D,1,1).transpose(2,3).reshape(2,L,B*L//D,cache.shape[-1])

            pred0 = self.lin(out.transpose(1,0))
            ones = sample*pred0
            zeros=(1-sample)*(1-pred0)
            total0 = ones+zeros

            for k in range(D):

                N = k*L//D
                #next couple of steps are crucial          
                #get the samples from N to N+L//D
                #Note: samples are the same as the original up to the Nth spin
                real = sflip[:,N:(k+1)*L//D]
                #flatten it out 
                tmp = real.reshape([B*L//D,L,1])
                #set up next state predction
                fsample=torch.zeros(tmp.shape,device=self.device)
                fsample[:,1:,:]=tmp[:,:-1,:]
                # put sequence before batch so you can use it with your transformer
                tgt=self.pe(fsample).transpose(1,0)
                #grab your transformer output
                out,_=self.next_with_cache(tgt,cache[:,:N],N)

                # self.lin actually does some repeated work but it's probably
                # negligable compared to the time attention takes
                output = self.lin(out[N:].transpose(1,0))
                # reshape output separating batch from spin flip grouping
                pred = output.view([B,L//D,L-N,1])
                real=real[:,:,N:]
                ones = real*pred
                zeros=(1-real)*(1-pred)
                total = ones+zeros
                #sum across the sequence for probabilities
                logp=torch.sum(torch.log(total+1e-10),dim=2).squeeze(2)
                logp+=torch.sum(torch.log(total0[:,:N]+1e-10),dim=1)
                probs[:,N:(k+1)*L//D]=logp
                
        return sample,probs
    def next_attn(_,tgt,layer,i=-1):
        """Calculates self attention with tgt and the last elem of tgt
        Inputs: 
            tgt - Tensor of shape [L+1,B,1]
            layer - TransformerDecoderLayer
            i - index of the first bit we want self-attention from
        Outputs:
            Tensor of shape [1,B,1]
        """
        src = tgt[i:, :, :]
        mask = None if i==-1 else _.mask[i:]
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
        return src
    
    def next_with_cache(self,tgt,cache=None,idx=-1):
        """Efficiently calculates the next output of a transformer given the input sequence and 
        cached intermediate layer encodings of the input sequence
        
        Inputs:
            tgt - Tensor of shape [L,B,1]
            cache - Tensor of shape [I,L,B,Nh]
            idx - index from which to start
            
        Outputs:
            output - Tensor of shape [L+c,B,1]
            new_cache - Tensor of shape [I,L+c,B,Nh]
        """
        output = tgt
        new_token_cache = []
        #go through each layer and apply self attention only to the last input
        for i, layer in enumerate(self.transformer.layers):
            output = self.next_attn(output,layer,idx)
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
    
    def set_mask(self, L):
        # take the log of a lower triangular matrix
        self.L=L
        self.mask = torch.log(torch.tril(torch.ones([L,L],device=self.device)))
        self.pe.L=L
        self.pe_t = self.pe.pe.transpose(1,0)
    
    
    def sample(self,B,L):
        """ Generates a set states
        Inputs:
            B (int)            - The number of states to generate in parallel
            L (int)            - The length of generated vectors
        Returns:
            samples - [B,L,1] matrix of zeros and ones for ground/excited states
        """
        #return (torch.rand([B,L,1],device=device)<0.5).to(torch.float32)
        #Sample set will have shape [B,L,1]
        #need one extra zero batch at the start for first pred hence input is [L+1,B,1] 
        #transformers don't do batch first so to save a bunch of transpose calls 
        input = torch.zeros([L+1,B,1],device=self.device)
        #self.set_mask(L)
        
        cache=None
        with torch.no_grad():
          for idx in range(1,L+1):
            #run the rnn on shape [B,1,1]   
            #encode the input to the proper shape
            encoded_input = input[:idx,:,:]+self.pe_t[:idx,:,:]
                        
            #Get transformer output
            output,cache = self.next_with_cache(encoded_input,cache)
            #if probs[i]=1 then there should be a 100% chance that sample[i]=1
            #if probs[i]=0 then there should be a 0% chance that sample[i]=1
            #stands that we generate a random uniform u and take int(u<probs) as our sample
            probs=self.lin(output[-1,:,:])
            sample = (torch.rand([B,1],device=device)<probs).to(torch.float32)
            input[idx,:,:]=sample
        #input's first entry is zero to get a predction for the first atom
        #print(".",end="")
        return input.transpose(1,0)[:,1:,:]
    
    
if __name__=="__main__":
    
    import sys
    print(sys.argv[1:])
    op=Opt()
    op.apply(sys.argv[1:])
    op.B=op.K*op.Q
    print(op)
    Lx=Ly=int(op.L**0.5)
    trainsformer = FastTransformer(Lx,Lx,Nh=op.Nh,num_layers=2)
    sampleformer= FastTransformer(Lx,Lx,Nh=op.Nh,num_layers=2)
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