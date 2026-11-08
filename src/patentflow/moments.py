"""Order-invariant weighted exponential moments for timestamp transactions."""
import numpy as np


class DecayedMoments:
    def __init__(self,n_nodes,dimension,half_life_days):
        self.numerator=np.zeros((n_nodes,dimension),dtype=np.float32)
        self.mass=np.zeros(n_nodes,dtype=np.float32)
        self.last_day=np.full(n_nodes,-1,dtype=np.int32)
        self.half_life=half_life_days

    def mean(self,nodes):
        return self.numerator[nodes]/np.maximum(self.mass[nodes,None],1e-12)

    def update(self,nodes,values,weights,day):
        nodes=np.asarray(nodes)
        unique,inverse=np.unique(nodes,return_inverse=True)
        if (self.last_day[unique]>=day).any():
            raise ValueError('Each timestamp must be committed once after all reads')
        decay=np.exp2(-(day-self.last_day[unique])/self.half_life).astype(np.float32)
        decay=np.where(self.last_day[unique]>=0,decay,0)
        sums=np.zeros((len(unique),values.shape[1]),dtype=np.float32)
        masses=np.zeros(len(unique),dtype=np.float32)
        np.add.at(sums,inverse,values*weights[:,None])
        np.add.at(masses,inverse,weights)
        self.numerator[unique]=self.numerator[unique]*decay[:,None]+sums
        self.mass[unique]=self.mass[unique]*decay+masses
        self.last_day[unique]=day

    def snapshot(self,day):
        mean=self.numerator/np.maximum(self.mass[:,None],1e-12)
        age=np.where(self.last_day>=0,day-self.last_day,0)
        mass=self.mass*np.exp2(-age/self.half_life)
        return mean,mass.astype(np.float32),age.astype(np.int32)
