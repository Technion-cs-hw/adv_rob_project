import torch
from tqdm import tqdm
import math
class Attacker:
    def __init__(self,model,loss_fn,device,epsilon=8/256,optim = None,batch_size = 16,num_iterations = 100,lr = 0.01):
        self.device = device
        self.dtype = torch.float32
        
        self.model = model
        self.loss_fn = loss_fn
        self.optim = None

        self.eps = epsilon
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.lr = lr
    
    def random_initialization(self,x):
        return torch.empty(x.shape, dtype=self.dtype, device=self.device).uniform_(-1, 1) * self.eps
        

    def project(self, perturbation):
        pert = torch.clamp(perturbation, -self.eps, self.eps)
        #pert.clamp_(self.pert_lb, self.pert_ub)
        return pert
    '''
    def normalize_grad(self, grad):
        return grad.sign()
       

    def step(self, pert, grad):
        grad = self.normalize_grad(grad)
        pert += grad
        return self.project(pert)

    def test_pert(self, x, y, pert):
        with torch.no_grad():
            output = self.model.forward(x + pert)
            loss = self.loss_fn(output,)
            return output, loss

    def eval_pert_untargeted(self, x, y, pert):
        with torch.no_grad():
            output, loss = self.test_pert(x, y, pert)
            succ = torch.argmax(output, dim=1) != y
            return loss, succ

    def eval_pert_targeted(self, x, y, pert):
        with torch.no_grad():
            output, loss = self.test_pert(x, y, pert)
            succ = torch.argmax(output, dim=1) == y
            return loss, succ

    def update_best(self, best_crit, new_crit, best_ls, new_ls):
        improve = new_crit > best_crit
        best_crit[improve] = new_crit[improve]
        for idx, best in enumerate(best_ls):
            new = new_ls[idx]
            best[improve] = new[improve]
    '''
    def perturb(self, x, w=0):
        '''
        n = x.shape[0]
        for i in range(math.floor(n/self.batch_size)):
            batch = x[i*self.batch_size:(i+1)*self.batch_size]
        '''
        delta = torch.nn.parameter.Parameter(self.random_initialization(x))
        optim = torch.optim.Adam([delta],lr = self.lr)
        x = x.to(self.device)
        
        for i in tqdm(range(self.num_iterations)):
            #out_clear = self.model(x)
            pert_image = torch.clamp(x+delta,0.0,1.0)
            stacked = torch.cat([x,pert_image],dim=0)
            
            _, _, out, _ = self.model(stacked) # prediction, pred_probs, logits, cross_enc_output
            out_clear = out[:out.shape[0]//2]
            out_pert = out[out.shape[0]//2:]
            
            pert_logits_norm = torch.norm(out_pert,p=2,dim=-1).mean()

            loss = -self.loss_fn(out_clear,out_pert).mean(dim=0)
            #torch.autograd.set_detect_anomaly(True)
            loss.backward(retain_graph=True)
            optim.step()

            #TODO: CHANGE THIS RANDOM SHIT
            #delta_prev = delta
            delta = self.project(delta).detach().clone()
            delta.requires_grad = True
            optim = torch.optim.Adam([delta],lr = self.lr)
            delta = delta.detach().to('cpu')
            x = x.detach().to('cpu')
            
        return delta,x

    def perturb2(self, x, y):
        '''
        n = x.shape[0]
        for i in range(math.floor(n/self.batch_size)):
            batch = x[i*self.batch_size:(i+1)*self.batch_size]
        '''
        delta = torch.nn.parameter.Parameter(self.random_initialization(x)).to(self.device)
        optim = torch.optim.Adam([delta],lr = self.lr)
        x = x.to(self.device)
        
        for i in tqdm(range(self.num_iterations)):
            _, _, out_clear, _ = self.model(x) # prediction, pred_probs, logits, cross_enc_output
            pert_image = torch.clamp(x+delta,0.0,1.0)

            _, _, out_pert, _ = self.model(pert_image) # prediction, pred_probs, logits, cross_enc_output
            
            l1 = out_clear.shape[0]
            l2 = out_pert.shape[0]
            ll = min(l1,l2)

            l1 = out_clear.shape[1]
            l2 = out_pert.shape[1]
            lll = min(l1,l2)

            loss = -self.loss_fn(out_clear[:ll,:lll],out_pert[:ll,:lll]).mean(dim=0)
            #torch.autograd.set_detect_anomaly(True)
            loss.backward(retain_graph=True)
            optim.step()

            #delta_prev = delta
            dulta = self.project(delta).detach()
            del delta
            dulta.requires_grad = True
            delta = dulta
            optim = torch.optim.Adam([delta],lr = self.lr)
        #x = x.detach().to('cpu')
        del x
        delta = delta.detach().to('cpu')
        #torch.cuda.empty_cache()    
        return delta
    def perturbBLIP(self, x):
        delta = torch.nn.parameter.Parameter(self.random_initialization(x)).to(self.device)
        optim = torch.optim.Adam([delta],lr = self.lr)
        x = x.to(self.device)
        
        for i in range(self.num_iterations):
            
            out_clear = self.model(x) # prediction, pred_probs, logits, cross_enc_output
            pert_image = torch.clamp(x+delta,0.0,1.0)

            out_pert = self.model(pert_image) # prediction, pred_probs, logits, cross_enc_output

            l1 = out_clear.shape[1]
            l2 = out_pert.shape[1]
            lll = min(l1,l2)
            
            loss = self.loss_fn(out_clear[:,:lll],out_pert[:,:lll]).mean(dim=0)            
            
            loss.backward(retain_graph=False)
            optim.step()


            dulta = self.project(delta).detach()
            del delta
            dulta.requires_grad = True
            delta = dulta
            optim = torch.optim.Adam([delta],lr = self.lr)
        x = x.detach().to('cpu')
        del x
        
        
        #torch.cuda.empty_cache()    
        return delta.detach().to('cpu')

    def perturbExpNet(self, x):
        delta = torch.nn.parameter.Parameter(self.random_initialization(x)).to(self.device)
        optim = torch.optim.Adam([delta],lr = self.lr)
        x = x.to(self.device)
        
        for i in range(self.num_iterations):
            
            _, _, out_clear, _ = self.model(x) # prediction, pred_probs, logits, cross_enc_output
            pert_image = torch.clamp(x+delta,0.0,1.0)

            _, _, out_pert, _ = self.model(pert_image) # prediction, pred_probs, logits, cross_enc_output
     
            l1 = out_clear.shape[0]
            l2 = out_pert.shape[0]
            ll = min(l1,l2)

            l1 = out_clear.shape[1]
            l2 = out_pert.shape[1]
            lll = min(l1,l2)
            
            loss = self.loss_fn(out_clear[:ll,:lll],out_pert[:ll,:lll]).mean(dim=0)            
            
            loss.backward(retain_graph=False)
            optim.step()


            dulta = self.project(delta).detach()
            del delta
            dulta.requires_grad = True
            delta = dulta
            optim = torch.optim.Adam([delta],lr = self.lr)
        x = x.detach().to('cpu')
        del x
        
        
        #torch.cuda.empty_cache()    
        return delta.detach().to('cpu')

    def perturbBLIPClassifier(self,x,y):
        delta = torch.nn.parameter.Parameter(self.random_initialization(x)).to(self.device)
        x = x.to(self.device)
        y=y.to(self.device)
        optim = torch.optim.Adam([delta],lr = self.lr)
        
        for i in range(self.num_iterations):
            
            #_, _, out_clear, _ = self.model(x) # prediction, pred_probs, logits, cross_enc_output
            pert_image = torch.clamp(x+delta,0.0,1.0)
            
            _, out_pert = self.model(pert_image.to(self.device)) # prediction, pred_probs, logits, cross_enc_output
            

            loss = -self.loss_fn(out_pert,y)#[4]#.max(dim=0)[0]            
            
            loss.backward(retain_graph=False)
            optim.step()


            dulta = self.project(delta).detach()
            del delta
            dulta.requires_grad = True
            delta = dulta
            optim = torch.optim.Adam([delta],lr = self.lr)
        x = x.detach().to('cpu')
        del x
        
        return delta.detach().to('cpu')
    
    def perturbCLIPClassifier(self,x,y):
        delta = torch.nn.parameter.Parameter(self.random_initialization(x)).to(self.device)
        x = x.to(self.device)
        y=y.to(self.device)
        optim = torch.optim.Adam([delta],lr = self.lr)
        
        for i in range(self.num_iterations):
            
            #_, _, out_clear, _ = self.model(x) # prediction, pred_probs, logits, cross_enc_output
            pert_image = torch.clamp(x+delta,0.0,1.0)
            
            out_pert = self.model(pert_image.to(self.device)) # prediction, pred_probs, logits, cross_enc_output
            

            loss = -self.loss_fn(out_pert,y)#[4]#.max(dim=0)[0]            
            
            loss.backward(retain_graph=False)
            optim.step()


            dulta = self.project(delta).detach()
            del delta
            dulta.requires_grad = True
            delta = dulta
            optim = torch.optim.Adam([delta],lr = self.lr)
        x = x.detach().to('cpu')
        del x
        
        return delta.detach().to('cpu')
    def perturbExpNetTargeted(self, x, token_id):
        delta = torch.nn.parameter.Parameter(self.random_initialization(x)).to(self.device)
        optim = torch.optim.Adam([delta],lr = self.lr)
        x = x.to(self.device)
        
        _, _, out_logits, _ = self.model(x)
        target_logit = torch.zeros(1,out_logits.shape[-1],device=self.device)
        target_logit[:,token_id] = 1
        for i in range(self.num_iterations):
            
            #_, _, out_clear, _ = self.model(x) # prediction, pred_probs, logits, cross_enc_output
            pert_image = torch.clamp(x+delta,0.0,1.0)
            
            _, _, out_pert, _ = self.model(pert_image) # prediction, pred_probs, logits, cross_enc_output
            
           
            loss = -self.loss_fn(target_logit.expand(out_pert.shape[0],-1)[4],out_pert[4])#[4]#.max(dim=0)[0]            
            
            loss.backward(retain_graph=False)
            optim.step()


            dulta = self.project(delta).detach()
            del delta
            dulta.requires_grad = True
            delta = dulta
            optim = torch.optim.Adam([delta],lr = self.lr)
        x = x.detach().to('cpu')
        del x
        
        
        #torch.cuda.empty_cache()    
        return delta.detach().to('cpu')
        

    def perturb3(self, x, y, w=0):
        '''
        n = x.shape[0]
        for i in range(math.floor(n/self.batch_size)):
            batch = x[i*self.batch_size:(i+1)*self.batch_size]
        '''
        delta = torch.nn.parameter.Parameter(self.random_initialization(x)).to(self.device)
        optim = torch.optim.Adam([delta],lr = self.lr)
        x = x.to(self.device)
        
        for i in tqdm(range(self.num_iterations), leave = False):
            _, _, out_clear, enc_clear = self.model(x) # prediction, pred_probs, logits, cross_enc_output
            pert_image = torch.clamp(x+delta,0.0,1.0)

            _, _, out_pert, enc_pert = self.model(pert_image) # prediction, pred_probs, logits, cross_enc_output
     
            l1 = out_clear.shape[0]
            l2 = out_pert.shape[0]
            ll = min(l1,l2)

            l1 = out_clear.shape[1]
            l2 = out_pert.shape[1]
            lll = min(l1,l2)
            
            out_l = self.loss_fn(out_clear[:ll,:lll],out_pert[:ll,:lll]).mean(dim=0)
            #enc_l = self.loss_fn(enc_clear, enc_pert).mean(dim=0)
            pert_logits_norm = torch.norm(out_pert,p=2,dim=-1).mean()
            
            loss = -(out_l + w*pert_logits_norm)
            #print(f"Logits loss: {out_l}"
            #      f"\nWeighted {w} Confidence (Norm2): {w*pert_logits_norm}")
 
            loss.backward(retain_graph=True)
            optim.step()

            dulta = self.project(delta).detach()
            del delta
            dulta.requires_grad = True
            delta = dulta
            optim = torch.optim.Adam([delta],lr = self.lr)
        #x = x.detach().to('cpu')
        del x
        #print(loss)
        #torch.cuda.empty_cache()    
        return delta.detach().to('cpu'), pert_logits_norm.detach().to('cpu')
            