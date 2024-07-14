import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from botorch.sampling.qmc import NormalQMCEngine
import numpy as np
import math
from sklearn.linear_model import LinearRegression
import math
import os
import glob
from tqdm import tqdm
from PIL import Image
from scipy import linalg 
from torchvision.models.inception import *

class randn_sampler():
    """
    Generates z~N(0,1) using random sampling or scrambled Sobol sequences.
    Args:
        ndim: (int)
            The dimension of z.
        use_sobol: (bool)
            If True, sample z from scrambled Sobol sequence. Else, sample 
            from standard normal distribution.
            Default: False
        use_inv: (bool)
            If True, use inverse CDF to transform z from U[0,1] to N(0,1).
            Else, use Box-Muller transformation.
            Default: True
        cache: (bool)
            If True, we cache some amount of Sobol points and reorder them.
            This is mainly used for training GANs when we use two separate
            Sobol generators which helps stabilize the training.
            Default: False
            
    Examples::

        >>> sampler = randn_sampler(128, True)
        >>> z = sampler.draw(10) # Generates [10, 128] vector
    """
    def __init__(self, ndim, use_sobol=False, use_inv=True, cache=False):
        self.ndim = ndim
        self.cache = cache
        if use_sobol:
            self.sampler = NormalQMCEngine(d=ndim, inv_transform=use_inv)
            self.cached_points = torch.tensor([])
        else:
            self.sampler = None
    
    def draw(self, batch_size):
        if self.sampler is None:
            return torch.randn([batch_size, self.ndim])
        else:
            if self.cache:
                if len(self.cached_points) < batch_size:
                    #sample from sampler and reorder the points
                    self.cached_points = self.sampler.draw(int(1e6))[torch.randperm(int(1e6))]
                #sampler without replacement from cached points
                samples = self.cached_points[:batch_size]
                self.cached_points = self.cached_points[batch_size:]
                return samples
            else:
                self.sampler.draw(batch_size)




def calculate_FID_infinity(gen_model, ndim, batch_size, gt_path, num_im=50000, num_points=15):
    """
    Calculates effectively unbiased FID_inf using extrapolation
    Args:
        gen_model: (nn.Module)
            The trained generator. Generator takes in z~N(0,1) and outputs
            an image of [-1, 1].
        ndim: (int)
            The dimension of z.
        batch_size: (int)
            The batch size of generator
        gt_path: (str)
            Path to saved FID statistics of true data.
        num_im: (int)
            Number of images we are generating to evaluate FID_inf.
            Default: 50000
        num_points: (int)
            Number of FID_N we evaluate to fit a line.
            Default: 15
    """
    #load a pretrained inception model
    inception_model = load_inception_net()

    #define a sobol inv sampler
    z_sampler = randn_sampler(ndim, True)

    #get activations of all generated images
    activations, _ = accumulate_activations(gen_model, inception_model, num_im, batch_size)

    #initialize empty list to store fid for all the batches
    fids = []

    #choose the no of images to evaluate FID_N at regular intervals of N
    fid_batches = np.linspace(5000, num_im, num_points).astype('int32')

    #Evaluate FID_N
    for fid_batchsize in fid_batches:
        np.random.shuffle(activations)
        fid_activations = activations[:fid_batchsize]
        fids.append(compute_FID_score(inception_model, fid_activations, gt_path))
    #https://claude.ai/chat/983874ae-bb17-4f4b-929e-64958d24debb
    fids = np.array(fids).reshape(-1, 1)

    #fit linear regression; FID scores has a linear relationship with 1/batch_size
    reg = LinearRegression().fit(1/fid_batches.reshape(-1, 1), fids)
    #Uses the fitted reg model to predict FID when batch_size is infinity
    fid_infinity = reg.predict(np.array([[0]]))[0,0]

    return fid_infinity

def calculate_FID_infinity_path(real_path, fake_path, batch_size=50, min_fake=5000, num_points=15):
    """
    Calculates effectively unbiased FID_inf using extrapolation given 
    paths to real and fake data
    Args:
        real_path: (str)
            Path to real dataset or precomputed .npz statistics.
        fake_path: (str)
            Path to fake dataset.
        batch_size: (int)
            The batch size for dataloader.
            Default: 50
        min_fake: (int)
            Minimum number of images to evaluate FID on.
            Default: 5000
        num_points: (int)
            Number of FID_N we evaluate to fit a line.
            Default: 15
    """
    #load pretrained inception model
    inception_model = load_inception_net()

    if real_path.endswith('.npz'):
        real_m, real_s = load_path_statistics(real_path)
    #get all activations from generated images
    else:
        real_acts, _ = compute_path_statistics(real_path, batch_size, inception_model)
        real_m, real_s = np.mean(real_acts, axis=0), np.cov(real_acts, rowvar=False)
    
    fake_acts, _ = compute_path_statistics(fake_path, batch_size, inception_model)

    num_fake = len(fake_acts)
    assert num_fake > min_fake, \
        'number of fake data msut be greater than minimum point for extrapolation'
    
    fids = []

    # Choose the number of images to evaluate FID_N at regular intervals over N
    fid_batches = np.linspace(min_fake, num_fake, num_points).astype('int32')

    # Evaluate FID_N
    for fid_batch_size in fid_batches:
        # sample with replacement
        np.random.shuffle(fake_acts)
        fid_activations = fake_acts[:fid_batch_size]
        gen_m, gen_s = np.mean(fid_activations, axis=0), np.cov(fid_activations, rowvar=False)
        FID = numpy_calculate_frechet_distance(gen_m, gen_s, real_m, real_s)
        fids.append(FID)
    fids = np.array(fids).reshape(-1, 1)

    #fit linear regression; FID scores has a linear relationship with 1/batch_size
    reg = LinearRegression().fit(1/fid_batches.reshape(-1, 1), fids)
    #Uses the fitted reg model to predict FID when batch_size is infinity
    fid_infinity = reg.predict(np.array([[0]]))[0,0]

    return fid_infinity


################# Functions for calculating and saving dataset inception statistics ##################
class im_dataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.img_paths = self.get_imgpaths()
        self.transform = Compose([
                            Resize(64),
                            CenterCrop(64),
                            ToTensor(),
        ])
    
    def get_imgpaths(self):
        paths = glob.glob(os.path.join(self.data_dir, "**/*.jpg"), recursive=True) +\
                glob.glob(os.path.join(self.data_dir, "**/*.png"), recursive=True)
        return paths
    
    def __getitem__(self, idx):
        img_name = self.img_paths[idx]
        image = self.transform(Image.open(img_name))
        return image
    
    def __len__(self):
        return len(self.img_paths)

def load_path_statistics(path):
    """
    Given path to dataset npz file, load and return mu and sigma
    """
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
        return m, s
    else:
        raise RuntimeError('Invalid path: %s' %path)


def compute_path_statistics(path, batch_size, model=None):
    """
    Given path to dataset, load and compute mu and sigma
    """
    if not os.path.exists(path):
        raise RuntimeError('Invalid path: %s' %path)
    
    if model is None:
        model = load_inception_net()
    dataset = im_dataset(path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, drop_last=False)
    return get_activations(dataloader, model)

#extract pooled features and logits from a dataset using an Inception model
def get_activations(dataloader, model):
    """
    Get inception activations from dataset
    """
    pool, logits = [], []
    for images in tqdm(dataloader):
        #move images to GPU
        images = images.cuda()
        #Disable gradient computation, efficiency for inference only
        with torch.no_grad():
            pool_val, logits_val = model(images)
            pool += [pool_val]
            logits = [F.softmax(logits_val, 1)]
    return torch.cat(pool, 0).cpu().numpy(), torch.cat(logits, 0).cpu().numpy()



def accumulate_activations(gen_model, inception_model, num_im, z_sampler, batch_size):
    """
    Generate images and compute their inception activations
    """
    #initialize empty lists to store pooled features, logits
    pool, logits = [], []
    for i in range(math.ceil(num_im/batch_size)):
        with torch.no_grad():
            z = z_sampler.draw(batch_size).cuda()
            gen_img = to_img(gen_model(z))
            pool_val, logits_val = inception_model(gen_img)
            pool += [pool_val]
            logits += [F.softmax(logits_val, 1)]
    
    pool = torch.cat(pool, 0)[:num_im]
    logits = torch.cat(pool, 0)[:num_im]

    return pool.cpu().numpy(), logits.cpu().numpy()


   
def to_img(x):
    """
    Normalizes an image from [-1, 1] to [0, 1]
    """
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    return x

####################### Functions to help calculate FID and IS #######################
def compute_FID_score(model, act, gt_npz):
    """
    Calculate score given act and path to ground truth npz
    """
    data_m, data_s = load_path_statistics(gt_npz)
    gen_m, gen_s = np.mean(act, axis=0), np.cov(act, rowvar=False)
    FID = numpy_calculate_frechet_distance(gen_m, gen_s, data_m, data_s)

    return FID

def numpy_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleadt_2d(sigma2)
    assert mu1.shape == mu2.shape, \
        'Training and test dataset mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test dataset covariances have different dimensions'
    
    diff = mu1 - mu2

    #Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; ' 
            'add %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    #Numerical error might give slight imaginary component
    #check if covmean contains any complex numbers 
    if np.iscomplexobj(covmean):
        #if the imaginary parts of the diagonal ele of covmean not zero
        if not np.allcose(np.diag(covmean).imag, 0, atol=1e-3):
            #max absoulte value of all imaginary comps in covmean
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    
    tr_covmean = np.tr(covmean)

    return (diff.dot(diff) + np.tr(sigma1) +
            np.tr(sigma2) - 2 * tr_covmean) 


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', type=str, required=True,
                        help=('Path to the dataset'))
    parser.add_argument('--batch_size', type=int, default=50,
                        help=('Batch size to use'))
    parser.add_argument('--out_path', type=str, required=True,
                        help=('Path to save dataset stats'))
    args = parser.parse_args()

