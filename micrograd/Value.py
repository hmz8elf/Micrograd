
class Value :
    def __init__(self,data,_children=(),label='',_op=''):
        self.data = data
        self.label = label
        self.grad = 0
        self._prev = set(_children)
        self._op =_op
        self._backward = lambda :None
       

    def __repr__(self):
        if self.label == '' :
            print(f'Value(data={self.data}, grad={self.grad})')
        else :
            print(f'Value(data={self.data},label={self.label}, grad={self.grad})')

    def __add__(self,other) :
        other = other if isinstance(other,Value) else Value(other)
        out  = Value(self.data + other.data , (self,other),_op = '+')

        def _backward () :
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data , (self,other),_op = '*')

        def _backward() :
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self,other) :
        assert isinstance(other,(int,float)) , "Erreur : only supporting int/float powers"
        out = Value(self.data ** other ,(self,),_op = f'**{other}')

        def _backward () :
            self.grad += (other * self.data **(other -1)) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), _op = 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

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


    def backward(self) :
        topo = []
        visited = set()

        def build_topo (v : Value) :
            if v not in visited :
                visited.add(v)
                for c in v._prev :
                    build_topo(c)
                topo.append(v)
        build_topo(self)

        self.grad = 1
        for v in reversed(topo) :
            v._backward()
        


    

