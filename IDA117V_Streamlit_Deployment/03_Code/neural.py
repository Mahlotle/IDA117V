"""Small NumPy neural classifiers. Original implementation, not RTDL code.

MLP: x -> ReLU(W0 x) -> ReLU(W1 h) -> sigmoid.
Residual: x -> ReLU(W0 x) -> ReLU(h + W1 ReLU(W2 h)) -> sigmoid.
Weighted binary cross entropy, Adam, L2, dropout and early stopping.
"""
import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score, log_loss

class TabularNeural(ClassifierMixin, BaseEstimator):
    def __init__(self, architecture='mlp', width=16, lr=0.003, l2=0.01,
                 dropout=0.15, epochs=250, batch_size=32, patience=30,
                 balanced=True, random_state=42):
        self.architecture=architecture; self.width=width; self.lr=lr
        self.l2=l2; self.dropout=dropout; self.epochs=epochs
        self.batch_size=batch_size; self.patience=patience
        self.balanced=balanced; self.random_state=random_state

    def _init(self, d):
        r=self.rng; w=self.width
        shapes={'W0':(d,w),'b0':(w,), 'W1':(w,w),'b1':(w,),
                'Wo':(w,1),'bo':(1,)}
        if self.architecture=='residual': shapes.update(W2=(w,w),b2=(w,))
        self.params={k:(r.normal(0,np.sqrt(2/s[0]),s) if k.startswith('W') else np.zeros(s)) for k,s in shapes.items()}
        self.m={k:np.zeros_like(v) for k,v in self.params.items()}
        self.v={k:np.zeros_like(v) for k,v in self.params.items()}; self.step=0

    def _forward(self,X,training=False):
        p=self.params; a=X@p['W0']+p['b0']; h=np.maximum(a,0)
        mask=np.ones_like(h)
        if training and self.dropout: mask=(self.rng.random(h.shape)>=self.dropout)/(1-self.dropout)
        hd=h*mask
        if self.architecture=='residual':
            z=hd@p['W2']+p['b2']; q=np.maximum(z,0)
            b=hd+q@p['W1']+p['b1']
        else: z=q=None; b=hd@p['W1']+p['b1']
        o=np.maximum(b,0); prob=expit(o@p['Wo']+p['bo']).ravel()
        return prob,(X,a,h,mask,hd,z,q,b,o)

    def _grad(self,X,y):
        prob,c=self._forward(X,True); X,a,h,mask,hd,z,q,b,o=c
        sw=self.class_weights[y.astype(int)]
        dz=((prob-y)*sw/sw.sum())[:,None]; p=self.params
        g={'Wo':o.T@dz,'bo':dz.sum(0)}; db=(dz@p['Wo'].T)*(b>0)
        if self.architecture=='residual':
            g['W1']=q.T@db; g['b1']=db.sum(0)
            dq=(db@p['W1'].T)*(z>0)
            g['W2']=hd.T@dq;g['b2']=dq.sum(0)
            dh=db+dq@p['W2'].T
        else:
            g['W1']=hd.T@db;g['b1']=db.sum(0);dh=db@p['W1'].T
        da=dh*mask*(a>0);g['W0']=X.T@da;g['b0']=da.sum(0)
        for k in g:
            if k.startswith('W'):g[k]+=self.l2*p[k]/len(y)
        return g

    def fit(self,X,y,eval_set=None):
        X=np.asarray(X,float);y=np.asarray(y,int)
        self.classes_=np.array([0,1]);self.n_features_in_=X.shape[1]
        self.rng=np.random.default_rng(self.random_state);self._init(X.shape[1])
        counts=np.bincount(y,minlength=2)
        self.class_weights=len(y)/(2*np.maximum(counts,1)) if self.balanced else np.ones(2)
        self.history_=[];best=np.inf;wait=0;best_params=None
        for epoch in range(self.epochs):
            order=self.rng.permutation(len(y))
            for start in range(0,len(y),self.batch_size):
                ix=order[start:start+self.batch_size];g=self._grad(X[ix],y[ix]);self.step+=1
                for k in g:
                    self.m[k]=.9*self.m[k]+.1*g[k];self.v[k]=.999*self.v[k]+.001*g[k]**2
                    self.params[k]-=self.lr*(self.m[k]/(1-.9**self.step))/(np.sqrt(self.v[k]/(1-.999**self.step))+1e-8)
            tp=self._forward(X)[0];row={'epoch':epoch+1,'train_loss':log_loss(y,tp,labels=[0,1]),'train_macro_f1':f1_score(y,tp>=.5,average='macro',zero_division=0)}
            if eval_set is not None:
                vx,vy=eval_set;vp=self._forward(np.asarray(vx,float))[0]
                loss=log_loss(vy,vp,labels=[0,1]);row.update(val_loss=loss,val_macro_f1=f1_score(vy,vp>=.5,average='macro',zero_division=0))
                if loss<best-1e-5:
                    best=loss;wait=0;best_params={k:v.copy() for k,v in self.params.items()};self.best_epoch_=epoch+1
                else:wait+=1
            self.history_.append(row)
            if eval_set is not None and wait>=self.patience:break
        if best_params is not None:self.params=best_params
        else:self.best_epoch_=len(self.history_)
        self.n_iter_=len(self.history_);return self

    def predict_proba(self,X):
        p=self._forward(np.asarray(X,float))[0];return np.column_stack([1-p,p])

    def predict(self,X):return (self.predict_proba(X)[:,1]>=.5).astype(int)
