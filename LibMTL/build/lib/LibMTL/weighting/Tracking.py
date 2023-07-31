import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting

from scipy.optimize import minimize

class Tracking(AbsWeighting):
    r"""Tracking (placrholder).
    

    Args:
        beta (float, default=0.1): learning rate for tracking variable
        sigma2 (float, default=0.5): decay rate for lr of tracking variable
        delta (float, default=0.1): learning rate for lambda parameter 
        sigma3 (float, default=0.5): decay rate for lr of lambda parametr

    .. warning::
            Tracking is not supported by representation gradients, i.e., ``rep_grad`` must be ``False``.

    """
    def __init__(self):
        super(Tracking, self).__init__()
        self.step = 1
    
    def init_param(self):
        self.y = 0
        self.lambd = 1/self.task_num*torch.ones([self.task_num, ])

    def projection_simplex_sort(self, v, z=1):
        n_features = v.shape[0]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - z
        ind = np.arange(n_features) + 1
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        w = np.maximum(v - theta, 0)
        return w       
    
    def _record_aux_data(self, data, beta, delta, sigma2, sigma3, seed):
        filename = f"log_aux_data_beta-{beta}_delta-{delta}_sigma2-{sigma2}_sigma3-{sigma3}_seed-{seed}"
        with open(filename, 'a+') as f:
            f.write(data+'\n')
        
    def backward(self, losses, large_losses=None, **kwargs):
        if large_losses is not None: # Calculating large batch (true) gradient for jacobian
            self._compute_grad_dim()
            grads = self._compute_grad(losses, mode='autograd')
            A = grads@torch.transpose(grads ,0, 1)

            eigmin = torch.min(torch.linalg.eigvals(A).float())
            eigmax = torch.max(torch.linalg.eigvals(A).float())
            kappa = eigmax/eigmin
            
            return eigmin, kappa
        
        else:
            beta, delta, sigma2, sigma3, seed = kwargs['beta'], kwargs['delta'], kwargs['sigma2'], kwargs['sigma3'], kwargs['seed']
            if self.rep_grad:
                raise ValueError('No support method Tracking with representation gradients (rep_grad=True)')
    #             per_grads = self._compute_grad(losses, mode='backward', rep_grad=True)
    #             grads = per_grads.reshape(self.task_num, self.rep.size()[0], -1).sum(1)
            else:
                self._compute_grad_dim()
                grads = self._compute_grad(losses, mode='backward')
            
            self.y = self.y - beta/self.step**sigma2*(self.y - torch.transpose(grads ,0, 1))
            # print(self.lambd.device, self.y.device)
            self.lambd = self.projection_simplex_sort(self.lambd - delta/self.step**sigma3*2*torch.transpose(self.y.cpu(), 0, 1)@(self.y.cpu()@self.lambd))
            self.step += 1
            new_grads = (self.y@self.lambd.to(self.device) )

            # record aux data here -----------------------------------------------
            if (self.step-1)==1 or (self.step-1)%50==0: 
                losses_data = ""
                grads_data = ""
                for i in range(self.task_num):
                    losses_data += f"{losses[i].cpu().detach().numpy()} "
                    grads_data += f"{np.linalg.norm(grads[i,:].cpu().detach().numpy())} "        

                multi_grad_data = f"{np.linalg.norm(new_grads.cpu().detach().numpy())}"

                G2 = grads.cpu().detach().numpy()@grads.cpu().detach().numpy().T
                Y2 = self.y.cpu().detach().numpy().T@self.y.cpu().detach().numpy()

                eigsG2, _ = np.linalg.eig(G2)
                eigsY2, _ = np.linalg.eig(Y2)
                kG2 = np.max(eigsG2)/np.min(eigsG2)
                kY2 = np.max(eigsY2)/np.min(eigsY2)

                data = f"{self.step} "+losses_data+grads_data+f"{kG2} {kY2} "+multi_grad_data
                self._record_aux_data(data, beta, delta, sigma2, sigma3, seed)
            # ---------------------------------------------------------------------
            
            self._reset_grad(new_grads)
            return self.lambd
