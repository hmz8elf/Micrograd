from graphviz import Digraph
from micrograd import Value

"""
This class is used to plot DAG 
of reccurence of opéretions on Value
"""

class DAG :
    def __init__(self,x : Value) :
        self.value = x

    def trace(self) :
        nodes, edges = set(), set()
        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))
                    build(child)
        build(self.value)
        return nodes, edges

    def draw_dot(self, format='png', rankdir='LR'):
        """
        format: png | svg | ...
        rankdir: TB (top to bottom graph) | LR (left to right)
        """
        assert rankdir in ['LR', 'TB']
        nodes, edges = self.trace(self.value)
        dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
        
        for n in nodes:
            dot.node(name=str(id(n)), label = "{ data %.4f | grad %.4f }" % (n.data, n.grad), shape='record')
            if n._op:
                dot.node(name=str(id(n)) + n._op, label=n._op)
                dot.edge(str(id(n)) + n._op, str(id(n)))
        
        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)
        
        return dot
