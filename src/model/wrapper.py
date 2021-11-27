from abc import ABC, abstractmethod
class AbstractModel():
    
    def __init__(self, Clf, **kwargs):
        self.clf = Clf(**kwargs)
    
    @abstractmethod
    def fit(self, X, y, **kwargs):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        pass
    
    @abstractmethod
    def uncertainty(self, X):
        pass  
    
    
class SklearnWrapper(AbstractModel):
    def __init__(self, Clf, **kwargs):
        super().__init__(Clf, **kwargs)

    def fit(self, X, y, **kwargs):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)
    
    def predict_proba(self, X):
        return self.clf.predict_proba(X)
    
    def uncertainty(self, X):
        return self.clf.decision_function(X)    