from micrograd import Value
from DAG import DAG

x = Value(1,label='x')
a = Value(3,label='a')
b = Value(5,label='b')
y= a*x + b
y.label ='y'

y.backward()
D = DAG(y)

D.draw_dot()
dot = D.draw_dot()
dot.render('graphe_y', view=True, cleanup=True)