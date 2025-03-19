import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder 


class LinearRegression:
    def __init__(self):
        self.betas=None
    def update_betas(self,X,y):
        self.betas= np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    
    def fit(self,X,y): 
        m,n=X.shape
        bias_term=np.ones((m,1))
        X=np.c_[bias_term,X]
        self.update_betas(X,y)
        self.y_mean=np.mean(y)
    def predict(self,X): 
        m=X.shape[0]
        bias=np.ones((m,1))
        X=np.c_[bias,X]
        return   X.dot(self.betas)  
    def score(self,X,y):
        m,n=X.shape
        sst=np.sum((y-self.y_mean)**2)
        y_predicted=self.predict(X)
        sse=np.sum((y-y_predicted)**2)
        r_spuared=1-(sse/sst)
        return  r_spuared
    def SSE(self,X,y):
        y_predicted=self.predict(X)
        return (y-y_predicted)**2

class RidgeRegression(LinearRegression):
    def __init__(self,lambdaa=.1):
        super().__init__()
        self.lambdaa=lambdaa

    def ubdate_betas(self,X,y):
        identity_matrix=np.identity(X.shape[1])
        identity_matrix[0][0]=0
        self.betas=np.linalg.inv(X.T.dot(X)+self.lambdaa*identity_matrix).dot(X.T).dot(y)




class GradientDescentRegressor:
    def __init__(self,learning_rate=.1,epochs=1000,type="batch",batch_size=30,penalty=None,lambdaa=.5,l1_ratio=.5,random_state=42):
        self.learning_rate=learning_rate
        self.type=type
        self.epochs=epochs
        self.batch_size=batch_size
        self.penalty=penalty
        self.lambdaa=lambdaa
        self.l1_ratio=l1_ratio

        self.weights=None####
        np.random.seed(random_state)

    def fit(self,X,y) :
        m,n=X.shape
        bias_term=np.ones((m,1))
        X=np.c_[bias_term,X]
        self.weights=np.zeros(n+1)
        
        for i in range (self.epochs):
            if self.type=="batch":
                 gradient=self._compute_gradient(X,y)
            elif self.type=="mini batch":
                indices=np.random.choice(m,self.batch_size,replace=False)
                gradient=self._compute_gradient(X[indices],y[indices])
            elif self.type=="stochastic":
                 index=np.random.choice(m)
                 gradient=self._compute_gradient(X[index],y[index])
            else:
                raise TypeError("only batch, mini batch and stochastic are supported")
            
            self.weights=self.weights-self.learning_rate/(2*m)*gradient
        self.y_mean=np.mean(y)

            
    def _compute_gradient(self,X,y):
        gradient=-2*X.T.dot(y)+2*X.T.dot(X).dot(self.weights)
        if self.penalty is not None:
            if self.penalty=="L2":
                penalty=2*self.lambdaa*self.weights
            elif self.penalty=="L1":
                penalty=self.lambdaa*np.sign(self.weights)
            elif self.penalty=="Elastic":
                l1_benality=2*self.lambdaa*self.l1_ratio*self.weights
                l2_benality=self.lambdaa*(1-self.l1_ratio)*np.sign(self.weights)  
                penalty=l1_benality+l2_benality
            else :
                raise ValueError("penalty can be None, l1, l2 or elastic net")    
            gradient[1:]+=penalty[1:]
        return gradient
    
    def predict(self,X):
        bias=np.ones((X.shape[0],1))
        X=np.c_[bias,X]
        return X.dot(self.weights)
    
    def score(self,X,y):
        m,n=X.shape
        sst=np.sum((y-self.y_mean)**2)
        y_predicted=self.predict(X)
        sse=np.sum((y-y_predicted)**2)
        r_spuared=1-(sse/sst)
        return  r_spuared


        


        
           