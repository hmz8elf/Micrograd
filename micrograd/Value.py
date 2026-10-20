
class Value :
    def __init__(self,data,_children=[],label='',_op=''):
        self.data = data
        self.label = label
        self.grad = 0
        self._prev = set(_children)
        self._op =_op
       

    def __repr__(self):
        if self.label == '' :
            print(f'Value(data={self.data}, grad={self.grad})')
        else :
            print(f'Value(data={self.data},label={self.label}, grad={self.grad})')

    def __add__(self,other) :
        other = other if isinstance(other,Value) else Value(other)
        out  = Value(self.data + other.data , (self,other),'+')

        return out

    def __mul__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data , (self,other),'*')
        return out

    def __pow__(self,other) :
        assert isinstance(other,(int,float)) , "Erreur : only supporting int/float powers"
        out = Value(self.data ** other ,(self,),f'**{other}')
        return out

    def __neg__(self): 
        return self * -1

    def __radd__(self, other): 
        return self + other

    def __sub__(self, other): 
        return self + (-other)

    def __rsub__(self, other): 
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    

