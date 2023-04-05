import matplotlib
import matplotlib.pyplot as plt 
matplotlib.use('TkAgg')
no = []

while True:
    try:
       line = input()
       line = line.split(' ')

       if len(line) != 6:
           continue
       
       no.append( float(line[-1]) )
    except EOFError:
        break

plt.plot(range(len(no)), no)
plt.show()
