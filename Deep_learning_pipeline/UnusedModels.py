# File with unused functions


class distanceLayer(nn.Module):
    
    """
    Distance processing layer placed at the end of the architecture encoder
    
    Params:
    
        - x: tensor to be processed in the distance processing layer
    
    """
    
    
    def __init__(self, ch_in, ch_out):
        
        super(distanceLayer, self).__init__()
        
        self.cat = Concat()
        
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=params.kernel_size,stride=1,padding=params.padding,bias=True)
        
        self.bn = EncoderNorm_2d(ch_out)
        
    
    def forward(self,x):
        
        distance_maps = utilities.distanceTransform(x, 'net')
        
        distance_maps = torch.tanh(distance_maps)
        
        concat = self.cat(x,distance_maps.float())
        
        conv_out = self.bn(self.conv(concat))
        
        return conv_out
    

    
    
class UNet_with_ResidualsFourLayers(nn.Module):
    
    """
    U-Net with residuals architecture, extracted from Bratt et al., 2019 paper, with four layers
    
    
    """

    def __init__(self):
        
        super(UNet_with_ResidualsFourLayers, self).__init__()

        self.cat = Concat()
        
        self.pad = addRowCol()
        
        # Decide on number of input channels
        
        if params.sum_work and 'both' in params.train_with:
            
            in_chan = 7 # Train with magnitude + phase + sum of both along time + MIP of both along time
        
        elif (params.sum_work and not('both' in params.train_with)):
            
            in_chan = 3 # Train with magnitude or phase, sum in time and MIP in time
            
        elif (not(params.sum_work) and 'both' in params.train_with):
            
            in_chan = 2 # Train with magnitude + phase (no sum)
            
        elif not(params.sum_work) and not('both' in params.train_with):
        
            in_chan = 1 # Train magnitude or phase (no sum)

        self.conv1 = nn.Conv2d(in_chan, params.base, params.kernel_size, padding=params.padding)
        
        if params.normalization is not None:
                    
            self.bn1 = EncoderNorm_2d(params.base)

        self.drop = nn.Dropout2d(params.dropout)

        self.conv2 = nn.Conv2d(params.base, params.base, params.kernel_size, padding=params.padding)
        
        if params.normalization is not None:
        
            self.bn2 = EncoderNorm_2d(params.base)
                

        self.Rd1 = Res_Down(params.base, params.base*2, params.kernel_size, params.padding)
        self.Rd2 = Res_Down(params.base*2, params.base*4, params.kernel_size, params.padding)
        self.Rd3 = Res_Down(params.base*4, params.base*4, params.kernel_size, params.padding)
        #self.Rd4 = Res_Down(params.base*8, params.base*8, params.kernel_size, params.padding)
        
        self.fudge = nn.ConvTranspose2d(params.base*4, params.base*4, params.kernel_size, stride = (1,1),\
                padding = params.padding)

        
        #self.Ru3 = Res_Up(params.base*8, params.base*8, params.kernel_size, params.padding)
        self.Ru2 = Res_Up(params.base*4, params.base*4, params.kernel_size, params.padding)
        self.Ru1 = Res_Up(params.base*4, params.base*2, params.kernel_size, params.padding)
        self.Ru0 = Res_Up(params.base*2, params.base, params.kernel_size, params.padding)

        

#        self.Ru3 = Res_Up(512,512)
#        self.Ru2 = Res_Up(512,256)
#        self.Ru1 = Res_Up(256,128)
#        self.Ru0 = Res_Up(128,64)

        self.Rf = Res_Final(params.base, len(params.class_weights), params.kernel_size, params.padding)


    def forward(self, x):
        
        if params.normalization is not None:
        
            out = F.relu(self.bn1(self.conv1(x)))

            e0 = F.relu(self.bn2(self.conv2(out)))
            
        else:
        
            out = F.relu(self.conv1(x))

            e0 = F.relu(self.conv2(out))

        e1 = self.Rd1(e0)
        e2 = self.drop(self.Rd2(e1))
        e3 = self.drop(self.Rd3(e2))
        #e4 = self.Rd4(e3)


        #d3 = self.Ru3(e4)
        d2 = self.Ru2(e3)
        
        if d2.shape[2] != e2.shape[2]:
            
            e2 = self.pad(e2)
        
        #d2 = self.Ru2(self.cat(d3[:,(params.base*4):],e3[:,(params.base*4):]))
        d1 = self.Ru1(self.cat(d2[:,(params.base*2):],e2[:,(params.base*2):]))
        
        if d1.shape[2] != e1.shape[2]:
        
            e1 = self.pad(e1)
            
        d0 = self.Ru0(self.cat(d1[:,params.base:],e1[:,params.base:]))
        
        
        if d0.shape[2] != e0.shape[2]:
        
            e0 = self.pad(e0)

        out = self.Rf(self.cat(e0[:,(params.base//2):],d0[:,(params.base//2):]))



        return out
    
    
class pretrainedEncoder(nn.Module):
    
    """
    Pretrained encoder with ResNet18 weights from ImageNet
    
    """
    
    def __init__(self):
        
        super(pretrainedEncoder, self).__init__()
        
        self.resnet = models.resnet18(pretrained=True).cuda()
        
        
    def forward(self,x):
        
        modules = list(self.resnet.children())[:(params.num_layers - 8)]
        
        #resnet = nn.Sequential(*modules)
        
        inter_results = []
        
        layersInterest = [2]
        
        for i in range(len(modules)):
            
            for param in modules[i].parameters():
            
                param.requires_grad = False # Do not need to train this part of the model
            
            x = modules[i](x)
            
            if i in layersInterest:
                
                inter_results.append(x)
        
        return x, inter_results
    

    
class pretrainingPreprocessing(nn.Module):
    
    """
    Preprocesses the data so that it can be properly processed in the pretrained ResNet18
    
    """
    
    def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        
        super(pretrainingPreprocessing, self).__init__()
    
        self.mean = torch.from_numpy(np.array(mean, dtype=np.float32)).float().to("cuda:0")

        self.std = torch.from_numpy(np.array(std, dtype=np.float32)).float().to("cuda:0")
        
        self.mean = self.mean.unsqueeze(0).unsqueeze(-1) #add a dimenstion for batch and (width*height)
        
        self.std = self.std.unsqueeze(0).unsqueeze(-1)
        
        
        # Decide on number of input channels
        
        if params.sum_work and 'both' in params.train_with:
            
            in_chan = 7 # Train with magnitude + phase + sum of both along time + MIP of both along time
        
        elif (params.sum_work and not('both' in params.train_with)):
            
            in_chan = 3 # Train with magnitude or phase, sum in time and MIP in time
            
        elif (not(params.sum_work) and 'both' in params.train_with):
            
            in_chan = 2 # Train with magnitude + phase (no sum)
            
        elif not(params.sum_work) and not('both' in params.train_with):
        
            in_chan = 1 # Train magnitude or phase (no sum)
        
        self.convertconv = nn.Conv2d(in_chan, 3, 1, padding=0) # Turn input without three or one channel into 3 channel-input
    
    def forward(self,x):
        
        # Pretrained encoder requires 3 channels. Modify input so that it has 3 channels

        if x.shape[1] == 1:
            
            x = x.repeat(1, 3, 1, 1) 
        
        else:
            
            if x.shape[1] != 3:
                
                x = self.convertconv(x)
        
        # Tensor normalization
        
        h, w = x.shape[2:]
        
        norm_tensor = x.view(x.shape[0], x.shape[1], -1).cuda() #batch x channel x (height*width)
        
        norm_tensor = norm_tensor - self.mean # Make image mean zero
        
        norm_tensor = norm_tensor / self.std # Make std = 1
        
        norm_tensor = norm_tensor.view(x.shape[0], x.shape[1], h, w) #back to batch x chan x w x h
        
        return norm_tensor
    

    
class UNet_with_ResidualsPretrained(nn.Module):
    
    """
    UNet with skip connections pretrained on ResNet18 (ImageNet) weights in the encoder
    
    """
    
    def __init__(self):
    
        super(UNet_with_ResidualsPretrained, self).__init__()
            
        self.preprocess = pretrainingPreprocessing()

        self.pool = nn.MaxPool2d(2, 2)
        
        self.cat = Concat()
        
        if params.normalization is not None:
        
            self.bn = EncoderNorm_2d(params.base)

        self.drop = nn.Dropout2d(params.dropout)
        
        self.encoder = pretrainedEncoder()
        
        self.pad = addRowCol()
        
        self.fudge = nn.ConvTranspose2d(params.base, params.base, params.kernel_size, stride = (2,2),\
                padding = params.padding)
        
        #self.Ru2 = Res_Up(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)), params.kernel_size, params.padding)
        
        #self.Ru1 = Res_Up(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 2)), params.kernel_size, params.padding)
        
        self.Ru0 = Res_Up(params.base, params.base, params.kernel_size, params.padding)
        
        self.fudge2 = nn.ConvTranspose2d(params.base, params.base, params.kernel_size, stride = (2,2), padding = params.padding)
        
        self.Rf = Res_Final(params.base, len(params.class_weights), params.kernel_size, params.padding)
        
   

    def forward(self,x):
        
        # Preprocess the data so that it can be inputted to the pretrained encoder
        
        x = self.preprocess(x)

        
        # Pretrained encoder
        
        e, inter = self.encoder(x)
        
        #print(e.shape, inter[-1].shape, inter[-2].shape)

        #e = self.fudge(e)
        
        #e = self.pad(e)
        
        #print(e.shape)
        
        # Decoder

        #d2 = self.Ru2(e)
        
        #print(d2.shape, inter[-1].shape)

        #if d2.shape[2] != inter[-1].shape[2]:
        
            #d2 = self.pad(d2)
        
        
        #d1 = self.Ru1(self.cat(e[:,(params.base*2):],inter[-1]))
        
        #if d1.shape[2] != inter[-2].shape[2]:
        
            #d1 = self.pad(d1)
        

        d0 = self.Ru0(e)
        
        # Final layer
        
        if d0.shape[2] != inter[-1].shape[2]:
            
            d0 = self.pad(d0)

        out = self.fudge2(self.cat(inter[-1][:,(params.base//2):],d0[:,(params.base//2):]))
        
        out = self.pad(out)
        
        out = self.Rf(out)


        return out
    

    
def connectedComponentLoss(output, target, weight = 1):

    """
    Compute a loss based on the connected components of the network output and the target

    Params:

        - output: network output (torch.tensor)

        - target: mask (torch.tensor)

    """

    cc_output = connectedComponents(output)
    
    cc_target = connectedComponents(target)
    
    return weight*(np.sum(np.array(cc_output) - np.array(cc_target)))**2


def focal_cc_loss(output, target, weight = 0.1):
    
    """
    Combination of focal Dice overlap loss with Connected Components Loss
    
    Params:
    
        - output: result from the network
        
        - target: ground truth result
        
        - weight: weight for the Connected Components Loss
        
    Returns:
        
        - Combined Focal + Connected Component loss
    
    
    """
    
    focal = focal_loss(output, target)
    
    cc = connectedComponentLoss(output, target, 0.1)
    
    return focal + cc





def tversky_loss_scalar(output, target):
    
    """
    Computes Tversky loss.
    
    Params:
        
        - output: result from the network
        
        - target: ground truth result
        
    Returns:
        
        - Dice loss
        
    """
    
    output = F.log_softmax(output, dim=1)[:,0]
    
    numerator = torch.sum(target * output)
    
    denominator = target * output + params.loss_beta * (1 - target) * output + (1 - params.loss_beta) * target * (1 - output)

    return 1 - (numerator) / (torch.sum(denominator))


def focal_tversky_loss(output, target):
    
    """
    Computes Focal-Tversky loss.
    
    Params:
        
        - output: result from the network
        
        - target: ground truth result
        
    Returns:
        
        - Dice loss
        
    """
    
    tversky_loss = abs(tversky_loss_scalar(output, target))
    
    return torch.pow(tversky_loss, params.loss_gamma)


def distanceTransform(tensor, key):
    
    """
    Compute distance transform of tensor, normalizing it from -1 to +1
    
    Params:
    
        - tensor: input tensor
        
        - key: variable specifying if the computation is inside the network (loss/net)
        
    Output:
    
        - trans: distance transform output
    
    """
    
    # Disallow gradients from this section
    
    #with torch.no_grad():
    
    if key == 'net' or key == 'NET' or key == 'Net':
        
        numpy_tensor = tensor.detach().cpu().numpy()
        
    elif key == 'loss' or key == 'LOSS' or key == 'Loss':
        
        numpy_tensor = tensor.cpu().numpy()

    batch_len = numpy_tensor.shape[0]

    if key == 'net' or key == 'NET' or key == 'Net':

        classes = numpy_tensor.shape[1]

    elif key == 'loss' or key == 'LOSS' or key == 'Loss':

        classes = 1

    trans_numpy = np.zeros(numpy_tensor.shape)

    trans_numpy_inv = np.zeros(numpy_tensor.shape)

    for batch_ind in range(batch_len):

        for cl in range(classes):

            if key == 'net' or key == 'NET' or key == 'Net':

                if len(numpy_tensor.shape) == 5:

                    numpy_tensor[batch_ind, cl, :, :, :] = (numpy_tensor[batch_ind, cl, :, :, :] - np.amin(numpy_tensor[batch_ind, cl, :, :, :]))/(np.amax(numpy_tensor[batch_ind, cl, :, :, :]) - np.amin(numpy_tensor[batch_ind, cl, :, :, :]))

                    trans_numpy[batch_ind, cl, :, :, :] = scipy.ndimage.morphology.distance_transform_edt(numpy_tensor[batch_ind, cl, :, :, :])


                elif len(numpy_tensor.shape) == 4:

                    numpy_tensor[batch_ind, cl, :, :] = (numpy_tensor[batch_ind, cl, :, :] - np.amin(numpy_tensor[batch_ind, cl, :, :]))/(np.amax(numpy_tensor[batch_ind, cl, :, :]) - np.amin(numpy_tensor[batch_ind, cl, :, :]))


                    trans_numpy[batch_ind, cl, :, :] = scipy.ndimage.morphology.distance_transform_edt(numpy_tensor[batch_ind, cl, :, :])

                trans_numpy_final = np.copy(trans_numpy)  

            elif key == 'loss' or key == 'LOSS' or key == 'Loss':

                if len(numpy_tensor.shape) == 4:

                    numpy_tensor[batch_ind, :, :, :] = (numpy_tensor[batch_ind, :, :, :] - np.amin(numpy_tensor[batch_ind, :, :, :]))/(np.amax(numpy_tensor[batch_ind, :, :, :]) - np.amin(numpy_tensor[batch_ind, :, :, :]))

                    numpy_tensor_inv = np.abs(1 - numpy_tensor) # Inverted tensor

                    trans_numpy[batch_ind, :, :, :] = scipy.ndimage.morphology.distance_transform_edt(numpy_tensor[batch_ind, :, :, :])

                    trans_numpy_inv[batch_ind, :, :, :] = scipy.ndimage.morphology.distance_transform_edt(numpy_tensor_inv[batch_ind, :, :, :])

                elif len(numpy_tensor.shape) == 3:

                    numpy_tensor[batch_ind, :, :] = (numpy_tensor[batch_ind, :, :] - np.amin(numpy_tensor[batch_ind, :, :]))/(np.amax(numpy_tensor[batch_ind, :, :]) - np.amin(numpy_tensor[batch_ind, :, :]))

                    numpy_tensor_inv = np.abs(1 - numpy_tensor) # Inverted tensor

                    trans_numpy[batch_ind, :, :] = scipy.ndimage.morphology.distance_transform_edt(numpy_tensor[batch_ind, :, :])

                    trans_numpy_inv[batch_ind, :, :] = scipy.ndimage.morphology.distance_transform_edt(numpy_tensor_inv[batch_ind, :, :])

                trans_numpy_final = trans_numpy - trans_numpy_inv


    # Normalization from -1 to +1
    
    if np.sum(trans_numpy_final.flatten()) > 0:
        
        warnings.filterwarnings("ignore")

        trans_numpy_final = (trans_numpy_final - np.amin(trans_numpy_final))/(np.amax(trans_numpy_final) - np.amin(trans_numpy_final))

        trans_numpy_final = 2*trans_numpy_final - 1

        trans = torch.tensor(trans_numpy_final, requires_grad = True, device = 'cuda:0')
        
    else:
        
        # Avoid empty distance transforms
        
        trans = torch.ones(numpy_tensor.shape, requires_grad = True, device = 'cuda:0')
        

 
        
    return trans
                        

    
def distance_loss(output, target):
    
    """
    Computes Distance loss to reduce the impact of false positives and refine vessel boundaries.
    
    Params:
        
        - output: result from the network
        
        - target: ground truth result
        
    Returns:
        
        - Distance loss
        
    """
    
    output = torch.argmax(output, 1)
    
    output_dist = distanceTransform(output, 'loss')
    
    #output_dist = torch.tensor(output_dist, requires_grad = True) # Allow for backprop
    
    target_dist = distanceTransform(target, 'loss')
    
    l1_loss = nn.L1Loss()
    
    return l1_loss(output_dist, target_dist)



def focal_distance_loss(output, target, iteration, total):
    
    
    """
    Computes Focal + Distance loss to improve result overlap while reducing the impact of false positives and refine vessel boundaries. The distance loss will be weighted with a scheduling process
    
    Params:
        
        - output: result from the network
        
        - target: ground truth result
        
        - iteration: iteration number
        
        - total: total number of training iterations
        
    Returns:
        
        - Distance loss
        
    """
    
    focal = focal_loss(output, target)
    
    distance = distance_loss(output, target)
    
    
    if iteration < total//2:
        
        # Before reaching half of the iterations, increasing weight
        
        distance_weight = 2*iteration/total
        
    else:
        
        # After half of iterations, weight of 1. Same importance as focal loss
        
        distance_weight = 1

        
   
    return focal + distance_weight*distance



class FullyRUNet(nn.Module):
    
    """
    Fully recurrent U-Net, with all layers as recurrent LSTM or GRU cells
    
    """
    
    def __init__(self):
        
        super(FullyRUNet, self).__init__()

        self.cat = Concat()
        
        self.pad = addRowCol3d()
        
        self.Maxpool = nn.MaxPool3d(kernel_size=2,stride=2)
        
        # Decide on number of input channels
        
        if 'both' in params.train_with:

            in_chan = 2

        else:

            in_chan = 1
                
            
        # Encoder
            
        if (params.rnn_position == 'full' or params.rnn_position == 'encoder') and params.rnn == 'LSTM':

            self.Down1 = ConvLSTM(in_chan, params.base*(2**(params.num_layers - 3)), (5, 5), 1, True, True, False)

            self.Down2 = ConvLSTM(params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 2)), (5, 5), 1, True, True, False)

            self.Down3 = ConvLSTM(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 1)), (5, 5), 1, True, True, False)

        elif (params.rnn_position == 'full' or params.rnn_position == 'encoder') and params.rnn == 'GRU':

            if torch.cuda.is_available():

                dtype = torch.cuda.FloatTensor # computation in GPU

            else:

                dtype = torch.FloatTensor

            self.Down1 = ConvGRU(in_chan, params.base*(2**(params.num_layers - 3)), (5, 5), 1, dtype, True, True, False)

            self.Down2 = ConvGRU(params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 2)), (5, 5), 1, dtype, True, True, False)

            self.Down3 = ConvGRU(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 1)), (5, 5), 1, dtype, True, True, False)

        else:

            self.Down1 = nn.Conv3d(in_chan, params.base*(2**(params.num_layers - 3)), kernel_size=5, stride=1,padding=0)

            self.Down2 = nn.Conv3d(params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 2)), kernel_size=5, stride=1,padding=0)

            self.Down3 = nn.Conv3d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 1)), kernel_size=5, stride=1,padding=0)
            
        self.bn1 = nn.InstanceNorm3d(params.base*(2**(params.num_layers - 3)))
        
        self.bn2 = nn.InstanceNorm3d(params.base*(2**(params.num_layers - 2)))
        
        self.bn3 = nn.InstanceNorm3d(params.base*(2**(params.num_layers - 1)))
        
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        
        self.drop = nn.Dropout3d(params.dropout)
        
        self.up_conv3 = nn.ConvTranspose3d(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 2)), params.kernel_size, stride = 2,padding=params.padding, output_padding = 1)
        
        self.up_conv2 = nn.ConvTranspose3d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 3)), params.kernel_size, stride = 2,padding=params.padding, output_padding = 1)
        
        self.Conv_1x1 = nn.Conv3d(params.base*(2**(params.num_layers - 3)), len(params.class_weights), kernel_size=1, stride=1,padding=0)
            
        
        # Decoder
        
        if (params.rnn_position == 'full' or params.rnn_position == 'decoder') and params.rnn == 'LSTM':
            
            self.Up3 = ConvLSTM(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 2)), (5, 5), 1, True, True, False)
            
            self.Up2 = ConvLSTM(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 3)), (5, 5), 1, True, True, False)
            
        elif (params.rnn_position == 'full' or params.rnn_position == 'decoder') and params.rnn == 'GRU':
            
            if torch.cuda.is_available():
                
                dtype = torch.cuda.FloatTensor # computation in GPU
                
            else:
                
                dtype = torch.FloatTensor
            
            self.Up3 = ConvGRU(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 2)), (5, 5), 1, dtype, True, True, False)
            
            self.Up2 = ConvGRU(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 3)), (5, 5), 1, dtype, True, True, False)
            
        else:
            
            self.Up3 = nn.Conv3d(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 2)), kernel_size=5, stride=1,padding=0)

            self.Up2 = nn.Conv3d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 3)), kernel_size=5, stride=1,padding=0)

        self.Att3 = Attention_block3d(F_g=params.base*(2**(params.num_layers - 2)),F_l=params.base*(2**(params.num_layers - 2)),F_int=params.base*(2**(params.num_layers - 3)))
        
        self.Att2 = Attention_block3d(F_g=params.base*(2**(params.num_layers - 3)),F_l=params.base*(2**(params.num_layers - 3)),F_int=int(params.base*(2**(params.num_layers - 4))))
                           
                           
    def forward(self,x):
                           
        # Reshape input: (B, C, H, W, T) --> (B, C, T, H, W)

        x = x.view(x.shape[0], x.shape[1], x.shape[-1], x.shape[-3], x.shape[-2])
                           
        # encoding path 
        
        if params.rnn_position == 'full' or params.rnn_position == 'encoder':
            
            x = x.view(x.shape[0], x.shape[2], x.shape[1], x.shape[3], x.shape[4])
        
            x1,_ = list(self.Down1(x))
            
            x1 = x1[0].view(x1[0].shape[0], x1[0].shape[2], x1[0].shape[1], x1[0].shape[3], x1[0].shape[4])
            
            x1_out = self.leaky_relu(self.bn1(x1))
            
            x1 = self.Maxpool(x1_out)
            
            x1 = x1.view(x1.shape[0], x1.shape[2], x1.shape[1], x1.shape[3], x1.shape[4])
            
            x2,_ = list(self.Down2(x1))
            
            x2 = x2[0].view(x2[0].shape[0], x2[0].shape[2], x2[0].shape[1], x2[0].shape[3], x2[0].shape[4])
            
            x2_out = self.drop(self.leaky_relu(self.bn2(x2)))
            
            x2 = self.Maxpool(x2_out)
            
            x2 = x2.view(x2.shape[0], x2.shape[2], x2.shape[1], x2.shape[3], x2.shape[4])
            
            x3,_ = list(self.Down3(x2))
            
            x3 = x3[0].view(x3[0].shape[0], x3[0].shape[2], x3[0].shape[1], x3[0].shape[3], x3[0].shape[4])
            
            x3 = self.drop(self.leaky_relu(self.bn3(x3)))
            
        else:
            
            x1_out = self.leaky_relu(self.bn1(self.Down1(x)))
            
            x1 = self.Maxpool(x1_out)
            
            x2_out = self.drop(self.leaky_relu(self.bn2(self.Down2(x1))))
            
            x2 = self.Maxpool(x2_out)
            
            x3 = self.drop(self.leaky_relu(self.bn3(self.Down3(x2))))
            
        
        # Decoding path
        
        d3 = self.up_conv3(x3)
        
        if d3.shape[2] != x2_out.shape[2]:
            
            d3 = self.pad(d3, x2_out.shape)
            
        
        x2 = self.Att3(g=d3,x=x2_out)
        
        if params.rnn_position == 'full' or params.rnn_position == 'decoder':
            
            cat2 = self.cat(x2,d3)
            
            cat2 = cat2.view(cat2.shape[0], cat2.shape[2], cat2.shape[1], cat2.shape[3], cat2.shape[4])
            
            d3,_ = list(self.Up3(cat2))
            
            d3 = d3[0].view(d3[0].shape[0], d3[0].shape[2], d3[0].shape[1], d3[0].shape[3], d3[0].shape[4])
            
            d3 = self.leaky_relu(self.bn2(d3))
            
            d2 = self.up_conv2(d3)
        
            if d2.shape[2] != x1_out.shape[2]:

                d2 = self.pad(d2, x1_out.shape)
                
            #if params.rnn_position == 'full':
                
                #x1_out = x1_out.view(x1_out.shape[0], x1_out.shape[2], x1_out.shape[1], x1_out.shape[3], x1_out.shape[4])

            x1 = self.Att2(g=d2,x=x1_out)
            
            cat1 = self.cat(x1,d2)

            cat1 = cat1.view(cat1.shape[0], cat1.shape[2], cat1.shape[1], cat1.shape[3], cat1.shape[4])
            
            d2,_ = list(self.Up2(cat1))
            
            d2 = d2[0].view(d2[0].shape[0], d2[0].shape[2], d2[0].shape[1], d2[0].shape[3], d2[0].shape[4])
            
            d2 = self.leaky_relu(self.bn1(d2))
            
        else:
            
            d3 = self.Up3(self.cat(x2,d3))
            
            d3 = self.leaky_relu(self.bn2(d3))
        
            d2 = self.up_conv2(d3)
        
            if d2.shape[2] != x1_out.shape[2]:

                d2 = self.pad(d2)

            x1 = self.Att2(g=d2,x=x1)
            
            d2 = self.Up3(self.cat(x1,d2))
            
            d2 = self.leaky_relu(self.bn2(d2))
            
        
        d1 = self.Conv_1x1(d2)
        
        d1 = d1.view(d1.shape[0], d1.shape[1], d1.shape[3], d1.shape[4], d1.shape[2])
       
        return d1

    
    



    
    
class UNetRNNDown(nn.Module):
    
    """
    Encoder layers of U-Net with convLSTMs
    
    """
    
    def __init__(self, in_chan, out_chan):
        
        super(UNetRNNDown, self).__init__()
        
        if params.rnn == 'LSTM':

            self.conv1 = ConvLSTM(in_chan, out_chan, (params.kernel_size, params.kernel_size), 1, True, True, False)
            
        elif params.rnn == 'GRU':
            
            if torch.cuda.is_available():
                
                dtype = torch.cuda.FloatTensor # computation in GPU
                
            else:
                
                dtype = torch.FloatTensor
            
            self.conv1 = ConvGRU(in_chan, out_chan, (params.kernel_size, params.kernel_size), 1, dtype, True, True, False)
            
        else:
             
            print('Wrong recurrent cell introduced. Please introduce a valid name for recurrent cell')
            
            
        
        if params.normalization is not None:
            
            self.bn1 = EncoderNorm_2d(out_chan)
            
        #self.drop = nn.Dropout2d(params.dropout)

        self.conv2 = nn.Conv2d(out_chan, out_chan, params.kernel_size + 1, stride = (2,2), padding = params.padding)
        
        if params.normalization is not None:
            
            self.bn2 = EncoderNorm_2d(out_chan)


    def forward(self, x):
        
        # Reshape x to enter the convLSTM cell: (BxT,C,H,W) --> (B,T,C,H,W)
        
        x = x.view(params.batch_size, x.shape[0]//params.batch_size, x.shape[-3], x.shape[-2], x.shape[-1])
        
        # Introduce x into convLSTM
        
        x,_ = list(self.conv1(x))
        
        # Reshape x again to original shape
        
        x = x[0].view(x[0].shape[0]*x[0].shape[1],x[0].shape[-3], x[0].shape[-2], x[0].shape[-1])
        
        if params.normalization is not None:
        
            x = self.bn2(self.conv2(F.leaky_relu(self.bn1(x))))
            
        else:
            
            x = self.conv2(F.leaky_relu(x))

        return F.leaky_relu(x)
    
    
class UNetRNNUp(nn.Module):
    
    """
    Decoder layers of U-Net with convLSTMs or convGRUs
    
    """
    
    def __init__(self, in_chan, out_chan):
        
        super(UNetRNNUp, self).__init__()
        
        if params.rnn == 'LSTM':

            self.conv1 = ConvLSTM(in_chan, out_chan, (params.kernel_size, params.kernel_size), 1, True, True, False)
            
        elif params.rnn == 'GRU':
            
            if torch.cuda.is_available():
                
                dtype = torch.cuda.FloatTensor # computation in GPU
                
            else:
                
                dtype = torch.FloatTensor
            
            self.conv1 = ConvGRU(in_chan, out_chan, (params.kernel_size, params.kernel_size), 1, dtype, True, True, False)
            
        else:
             
            print('Wrong recurrent cell introduced. Please introduce a valid name for recurrent cell')
        
        if params.normalization is not None:
            
            self.bn1 = EncoderNorm_2d(out_chan)
            
        #self.drop = nn.Dropout2d(params.dropout)

        self.conv2 = nn.ConvTranspose2d(out_chan, out_chan, params.kernel_size + 1, stride = (2,2), padding=(1,1))
        
        if params.normalization is not None:
            
            self.bn2 = EncoderNorm_2d(out_chan)


    def forward(self, x):
        
        # Reshape x to enter the convLSTM cell: (BxT,C,H,W) --> (B,T,C,H,W)
        
        x = x.view(params.batch_size, x.shape[0]//params.batch_size, x.shape[-3], x.shape[-2], x.shape[-1])
        
        # Introduce x into convLSTM
        
        x,_ = list(self.conv1(x))
        
        # Reshape x again to original shape
        
        x = x[0].view(x[0].shape[0]*x[0].shape[1],x[0].shape[-3], x[0].shape[-2], x[0].shape[-1])
        
        if params.normalization is not None:
        
            x = self.bn2(self.conv2(F.leaky_relu(self.bn1(x))))

            
        else:
            
            x = self.conv2(F.leaky_relu(x))

        return F.leaky_relu(x)
    
    
    
class UNetRNN(nn.Module):
    
    """
    U-Net with convLSTM or convGRU operators, to process 2D+time information
    
    """
    
    
    def __init__(self):
        
        super(UNetRNN, self).__init__()
        
        self.cat = Concat()
        
        self.pad = addRowCol()
        
        if 'both' in params.train_with:
            
            channels = 2
            
        else:
            
            channels = 1
        
        self.conv1 = nn.Conv2d(channels, params.base, params.kernel_size, padding=params.padding)
        
        if params.normalization is not None:
                    
            self.bn1 = EncoderNorm_2d(params.base)

        self.drop = nn.Dropout2d(params.dropout)

        self.conv2 = nn.Conv2d(params.base, params.base, params.kernel_size, padding=params.padding)
        
        if params.normalization is not None:
        
            self.bn2 = EncoderNorm_2d(params.base)
            
        if params.rnn_position == 'encoder' or params.rnn_position == 'full': 
            
            self.Rd1 = UNetRNNDown(params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 2)))

            self.Rd2 = UNetRNNDown(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 1)))

            self.Rd3 = UNetRNNDown(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)))
            
        else:
            
            self.Rd1 = Res_Down(params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 2)), params.kernel_size, params.padding)
            self.Rd2 = Res_Down(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 1)), params.kernel_size, params.padding)
            self.Rd3 = Res_Down(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)), params.kernel_size, params.padding)
            
        
        
        if params.rnn_position == 'decoder' or params.rnn_position == 'full':
            
            self.Ru3 = UNetRNNUp(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)))

            self.Ru2 = UNetRNNUp(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 2)))

            self.Ru1 = UNetRNNUp(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 3)))
            
        else:
                
            self.Ru3 = Res_Up(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)), params.kernel_size, params.padding)

            self.Ru2 = Res_Up(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 2)), params.kernel_size, params.padding)

            self.Ru1 = Res_Up(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 3)), params.kernel_size, params.padding)
                
        self.Rf = Res_Final(params.base, len(params.class_weights), params.kernel_size, params.padding)
        
        
    def forward(self, x):
        
        # Reshape input: (B, C, H, W, T) --> (B*T, C, H, W)
        
        x = x.view(x.shape[0]*x.shape[-1], x.shape[1], x.shape[-3], x.shape[-2])
        
        # First convolutional layer
        
        if params.normalization is not None:
        
            out = F.relu(self.bn1(self.conv1(x)))

            e0 = F.relu(self.bn2(self.conv2(out)))
            
        else:
        
            out = F.relu(self.conv1(x))

            e0 = F.relu(self.conv2(out))
            
        # Encoder layers: convRNN + conv2D

        e1 = self.Rd1(e0)
        
        e2 = self.drop(self.Rd2(e1))
        
        e3 = self.drop(self.Rd3(e2))
    
        # Decoder layers: conv2Dtranspose + conv2d
        
        d3 = self.Ru3(e3)

        if d3.shape[2] != e2.shape[2]:
                    
            e2 = self.pad(e2)
                
        d2 = self.Ru2(self.cat(d3[:,(params.base*(2**(params.num_layers - 2))):],e2[:,(params.base*(2**(params.num_layers - 2))):]))
        
        if d2.shape[2] != e1.shape[2]:
            
            e1 = self.pad(e1)
            
        d1 = self.Ru1(self.cat(d2[:,(params.base*(2**(params.num_layers - 3))):],e1[:,(params.base*(2**(params.num_layers - 3))):]))
                
        # Final output layer
        
        if d1.shape[2] != e0.shape[2]:
        
            e0 = self.pad(e0)

        out = self.Rf(self.cat(e0[:,(params.base//2):],d1[:,(params.base//2):]))
        
        # Reshape output to original dimensions for loss function computation with respect to corresponding mask
        
        out = out.view(params.batch_size, out.shape[1], out.shape[2], out.shape[3], out.shape[0]//params.batch_size)

        
        return out
    
    
    
    
class conv3dnorm(nn.Module):
    
    """
    Provide a 3D convolution followed by a normalization and a PReLU
    
    Params:
    
        - in_chan: input number of channels
        
        - out_chan: output number of channels
        
        - kernel: kernel size
        
        - stride: stride
        
        - padding: padding
        
        - activation: activation function to use ('prelu', 'relu' or 'elu')
        
        - transpose: flag indicating if the convolution is transpose or not (for upsampling)
    
    """
    
    def __init__(self, in_chan, out_chan, kernel, stride, padding, activation, transpose):
        
        super(conv3dnorm, self).__init__()
        
        if transpose:
            
            self.conv = nn.ConvTranspose3d(in_chan, out_chan, kernel, stride, padding)
            
        else:
        
            self.conv = nn.Conv3d(in_chan, out_chan, kernel, stride, padding)
        
        self.norm = nn.InstanceNorm3d(out_chan)
        
        if activation == 'prelu':
        
            self.activ = nn.PReLU(out_chan)
            
        elif activation == 'relu':
            
            self.activ = nn.ReLU()
            
        elif activation == 'elu':
            
            self.activ = nn.ELU()
            
        else:
            
            print('Unrecognized activation function. Please provide a valid key for activation: "elu", "prelu" or "relu"')
            
        
    def forward(self,x):    
        
        conv = self.conv(x)
        
        norm_conv = self.norm(conv)
        
        act_norm_conv = self.activ(norm_conv)
        
        return act_norm_conv
    
        
    
class AttentionVNet(nn.Module):
    
    """
    VNet from Milletari et al. 2016. Process time dimension as a third spatial dimension
    
    Include attention gates from Oktay et al. 2018
    
    """
    
    def __init__(self):
        
        super(AttentionVNet, self).__init__()
    
        self.cat = Concat()

        self.pad = addRowCol3d()

        self.drop = nn.Dropout3d(p = 0.5)

        # Decide on number of input channels

        if 'both' in params.train_with:

            in_chan = 2

        else:

            in_chan = 1

        # Recommended convolutional parameters: base = 16, kernel = 5x5x5, padding = 2x2x2, stride = 1

        # Downsampling layer 1

        self.conv1 = conv3dnorm(in_chan, params.base*(2**(params.num_layers - 3)), params.kernel_size, 1, params.padding, 'prelu', False)
        
        self.prelu1 = nn.PReLU(params.base*(2**(params.num_layers - 3)))
        
        self.down1 = conv3dnorm(params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 2)), 2, 2, 0, 'prelu', False)
        
        # Downsampling layer 2

        self.conv2_1 = conv3dnorm(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.kernel_size, 1, params.padding, 'prelu', False)
        
        self.conv2_2 = conv3dnorm(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.kernel_size, 1, params.padding, 'prelu', False)
        
        self.prelu2 = nn.PReLU(params.base*(2**(params.num_layers - 2)))
        
        self.down2 = conv3dnorm(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 1)), 2, 2, 0, 'prelu', False)
        
        # Downsampling layer 3
        
        self.conv3_1 = conv3dnorm(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)), params.kernel_size, 1, params.padding, 'prelu', False)
        
        self.conv3_2 = conv3dnorm(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)), params.kernel_size, 1, params.padding, 'prelu', False)
        
        self.conv3_3 = conv3dnorm(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)), params.kernel_size, 1, params.padding, 'prelu', False)
        
        self.prelu3 = nn.PReLU(params.base*(2**(params.num_layers - 1)))
        
        # Upsampling layer 2
        
        self.up2 = conv3dnorm(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 2)), 2, 2, 0, 'prelu', True)
        
        self.Att2 = Attention_block3d(F_g=params.base*(2**(params.num_layers - 2)),F_l=params.base*(2**(params.num_layers - 2)),F_int=params.base*(2**(params.num_layers - 3)))
        
        self.conv2_1up = conv3dnorm(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 2)), params.kernel_size, 1, params.padding, 'prelu', False)
        
        self.conv2_2up = conv3dnorm(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.kernel_size, 1, params.padding, 'prelu', False)
        
        self.prelu2up = nn.PReLU(params.base*(2**(params.num_layers - 2)))
       
        # Upsampling layer 1
        
        self.up1 = conv3dnorm(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 3)), 2, 2, 0, 'prelu', True)
        
        self.Att1 = Attention_block3d(F_g=params.base*(2**(params.num_layers - 3)),F_l= params.base*(2**(params.num_layers - 3)),F_int=int(params.base*(2**(params.num_layers - 4))))
        
        self.conv1_1up = conv3dnorm(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 3)), params.kernel_size, 1, params.padding, 'prelu', False)
        
        self.prelu1up = nn.PReLU(params.base*(2**(params.num_layers - 3)))
        
        self.conv1_2up = conv3dnorm(params.base*(2**(params.num_layers - 3)), len(params.class_weights), 1, 1, 0, 'prelu', False)
        
        
    def forward(self, x):
        
        # Tensor reshaping
            
        x = x.view(x.shape[0], x.shape[1], x.shape[-1], x.shape[2], x.shape[-2])
        
        # Downsampling layer 1

        x1 = self.conv1(x)
        
        x_cat = x.clone()
        
        for i in range(params.base - 1):

            x_cat = torch.cat((x_cat, x), dim = 1)

        x1 = self.prelu1(torch.add(x_cat, x1))
        
        x_down1 = self.down1(x1)
        
        # Downsampling layer 2
        
        x2_1 = self.conv2_1(self.drop(x_down1))
        
        x2_2 = self.conv2_2(x2_1)
        
        x2 = self.prelu2(torch.add(x_down1,x2_2))
        
        x_down2 = self.down2(x2)
        
        # Downsampling layer 3
        
        x3_1 = self.conv3_1(self.drop(x_down2))
        
        x3_2 = self.conv3_2(x3_1)
        
        x3_3 = self.conv3_3(x3_2)
        
        x3 = self.prelu3(torch.add(x_down2,x3_3))
        
        # Upsampling layer 2
        
        x_up2 = self.up2(x3)
        
        if x_up2.shape[2] != x2.shape[2]:
        
            x_up2 = self.pad(x_up2, x2.shape)
        
        x2 = self.Att2(g=x_up2,x=x2)
        
        x_up2_1 = self.conv2_1up(self.drop(self.cat(x2, x_up2)))
        
        x_up2_2 = self.conv2_2up(x_up2_1)
        
        x_up2 = self.prelu2up(torch.add(self.drop(x_up2), x_up2_2))
        
        # Upsampling layer 1
        
        x_up1 = self.up1(x_up2)
        
        if x_up1.shape[2] != x1.shape[2]:
        
            x_up1 = self.pad(x_up1, x1.shape)
        
        x1 = self.Att1(g = x_up1, x = x1)
        
        x_up1_1 = self.conv1_1up(self.drop(self.cat(x1, x_up1)))
        
        x_up1 = self.prelu1up(torch.add(self.drop(x_up1), x_up1_1))
        
        out = self.conv1_2up(x_up1)
        
        out = out.view(out.shape[0], out.shape[1], out.shape[-2], out.shape[-1], out.shape[2])
        
        return out
    
    
    
class RecurrentVNet(nn.Module):
    
    """
    VNet from Milletari et al. 2016. Process time dimension as a third spatial dimension
    
    Include recurrent connections in the encoder-decoder skip connections, as in Payer et al. 2018
    
    """
    
    def __init__(self):
        
        super(RecurrentVNet, self).__init__()
    
        self.cat = Concat()

        self.pad = addRowCol3d()

        self.drop = nn.Dropout3d(p = 0.5)

        # Decide on number of input channels

        if 'both' in params.train_with:

            in_chan = 2

        else:

            in_chan = 1

        # Recommended convolutional parameters: base = 16, kernel = 5x5x5, padding = 2x2x2, stride = 1

        # Downsampling layer 1

        self.conv1 = conv3dnorm(in_chan, params.base*(2**(params.num_layers - 3)), params.kernel_size, 1, params.padding, 'prelu', False)
        
        self.prelu1 = nn.PReLU(params.base*(2**(params.num_layers - 3)))
        
        self.down1 = conv3dnorm(params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 2)), 2, 2, 0, 'prelu', False)
        
        # Downsampling layer 2

        self.conv2_1 = conv3dnorm(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.kernel_size, 1, params.padding, 'prelu', False)
        
        self.conv2_2 = conv3dnorm(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.kernel_size, 1, params.padding, 'prelu', False)
        
        self.prelu2 = nn.PReLU(params.base*(2**(params.num_layers - 2)))
        
        self.down2 = conv3dnorm(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 1)), 2, 2, 0, 'prelu', False)
        
        # Downsampling layer 3
        
        self.conv3_1 = conv3dnorm(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)), params.kernel_size, 1, params.padding, 'prelu', False)
        
        self.conv3_2 = conv3dnorm(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)), params.kernel_size, 1, params.padding, 'prelu', False)
        
        self.conv3_3 = conv3dnorm(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)), params.kernel_size, 1, params.padding, 'prelu', False)
        
        self.prelu3 = nn.PReLU(params.base*(2**(params.num_layers - 1)))
        
        # Recurrent layers
        
        if params.rnn is not None:
            
            if params.rnn == 'LSTM':
                
                self.rnn2 = ConvLSTM(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)), (params.kernel_size, params.kernel_size), 1, True, True, False)
            
                self.rnn1 = ConvLSTM(params.base*(2**(params.num_layers - 2)), int(params.base*(2**(params.num_layers - 2))), (params.kernel_size, params.kernel_size), 1, True, True, False)

            elif params.rnn == 'GRU':

                if torch.cuda.is_available():

                    dtype = torch.cuda.FloatTensor # computation in GPU

                else:

                    dtype = torch.FloatTensor

                self.rnn2 = ConvGRU(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)), (params.kernel_size, params.kernel_size), 1, dtype, True, True, False)

                self.rnn1 = ConvGRU(params.base*(2**(params.num_layers - 2)), int(params.base*(2**(params.num_layers - 2))), (params.kernel_size, params.kernel_size), 1, dtype, True, True, False)
            
            else:

                print('Wrong recurrent cell. Please type "LSTM" or "GRU" as possible names for a recurrent cell')
            
            
            
        else:
            
            print('A certain type of RNN should be specified, as "GRU" or "LSTM"')
        
        # Upsampling layer 2
        
        self.up2 = conv3dnorm(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 2)), 2, 2, 0, 'prelu', True)
        
        self.conv2_1up = conv3dnorm(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 2)), params.kernel_size, 1, params.padding, 'prelu', False)
        
        self.conv2_2up = conv3dnorm(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.kernel_size, 1, params.padding, 'prelu', False)
        
        self.prelu2up = nn.PReLU(params.base*(2**(params.num_layers - 2)))
       
        # Upsampling layer 1
        
        self.up1 = conv3dnorm(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 3)), 2, 2, 0, 'prelu', True)
        
        self.conv1_1up = conv3dnorm(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 3)), params.kernel_size, 1, params.padding, 'prelu', False)
        
        self.prelu1up = nn.PReLU(params.base*(2**(params.num_layers - 3)))
        
        self.conv1_2up = conv3dnorm(params.base*(2**(params.num_layers - 3)), len(params.class_weights), 1, 1, 0, 'prelu', False)
        
        
    def forward(self, x):
        
        # Tensor reshaping
            
        x = x.view(x.shape[0], x.shape[1], x.shape[-1], x.shape[2], x.shape[-2])
        
        # Downsampling layer 1

        x1 = self.conv1(x)
        
        x_cat = x.clone()
        
        for i in range(params.base - 1):

            x_cat = torch.cat((x_cat, x), dim = 1)

        x1 = self.prelu1(torch.add(x_cat, x1))
        
        x_down1 = self.down1(x1)
        
        # Downsampling layer 2
        
        x2_1 = self.conv2_1(self.drop(x_down1))
        
        x2_2 = self.conv2_2(x2_1)
        
        x2 = self.prelu2(torch.add(x_down1,x2_2))
        
        x_down2 = self.down2(x2)
        
        # Downsampling layer 3
        
        x3_1 = self.conv3_1(self.drop(x_down2))
        
        x3_2 = self.conv3_2(x3_1)
        
        x3_3 = self.conv3_3(x3_2)
        
        x3 = self.prelu3(torch.add(x_down2,x3_3))
        
        # Upsampling layer 2
        
        x_up2 = self.up2(x3)
        
        if x_up2.shape[2] != x2.shape[2]:
        
            x_up2 = self.pad(x_up2, x2.shape)
        
        cat2 = self.drop(self.cat(x_up2, x2))
        
        cat2 = cat2.view(cat2.shape[0], cat2.shape[2], cat2.shape[1], cat2.shape[-2], cat2.shape[-1])
        
        x2, _ = self.rnn2(cat2)
        
        x2 = x2[0].view(x2[0].shape[0], x2[0].shape[2], x2[0].shape[1], x2[0].shape[-2], x2[0].shape[-1])
        
        x_up2_1 = self.conv2_1up(x2)
        
        x_up2_2 = self.conv2_2up(x_up2_1)
        
        x_up2 = self.prelu2up(torch.add(self.drop(x_up2), x_up2_2))
        
        # Upsampling layer 1
        
        x_up1 = self.up1(x_up2)
        
        if x_up1.shape[2] != x1.shape[2]:
        
            x_up1 = self.pad(x_up1, x1.shape)
        
        cat1 = self.drop(self.cat(x1, x_up1))
        
        cat1 = cat1.view(cat1.shape[0], cat1.shape[2], cat1.shape[1], cat1.shape[-2], cat1.shape[-1])
        
        x1, _ = self.rnn1(cat1)
        
        x1 = x1[0].view(x1[0].shape[0], x1[0].shape[2], x1[0].shape[1], x1[0].shape[-2], x1[0].shape[-1])
        
        x_up1_1 = self.conv1_1up(x1)
        
        x_up1 = self.prelu1up(torch.add(self.drop(x_up1), x_up1_1))
        
        out = self.conv1_2up(x_up1)
        
        out = out.view(out.shape[0], out.shape[1], out.shape[-2], out.shape[-1], out.shape[2])
        
        return out