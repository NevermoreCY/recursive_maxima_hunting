import numpy as np
import matplotlib.pyplot as plt

x_axis = [50,100,200,500,1000]

# peak

base = [0.4231,0.4012,0.3673,0.3230,0.2969]
PCA = []
PLS = []
MH = []
RMH = [0.2834,0.2446,0.2101,0.1822,0.1622 ]

plt.plot(x_axis,base, marker='s' , color='#C0C0C0')
plt.scatter(x_axis,base,s=100, marker='s' , label="Base", color='silver',linewidths=1,edgecolors="black")
plt.plot(x_axis,RMH, marker='s' , color='#ADD8E6')
plt.scatter(x_axis,RMH,s=100, marker='o' , label="RMH", color='#ADD8E6',linewidths=1,edgecolors="black")
plt.legend()
plt.title("Peak 1 ")
plt.ylabel("Classification Error")
plt.show()
plt.close()

# peak 2

base = [0.2402,0.1920,0.1520,0.1107,0.0899]
PCA = []
PLS = []
MH = []
RMH = [0.1408,0.1010,0.0705,0.0464,0.0371]

plt.plot(x_axis,base, marker='s' , color='#C0C0C0')
plt.scatter(x_axis,base,s=100, marker='s' , label="Base", color='silver',linewidths=1,edgecolors="black")
plt.plot(x_axis,RMH, marker='s' , color='#ADD8E6')
plt.scatter(x_axis,RMH,s=100, marker='o' , label="RMH", color='#ADD8E6',linewidths=1,edgecolors="black")
plt.legend()
plt.title("Peak 2 ")
plt.ylabel("Classification Error")
plt.show()
plt.close()


# square

base = [0.2090,0.1912,0.1796,0.1681,0.1627]
PCA = []
PLS = []
MH = []
RMH = []

plt.plot(x_axis,base, marker='s' , color='#C0C0C0')
plt.scatter(x_axis,base,s=100, marker='s' , label="Base", color='silver',linewidths=1,edgecolors="black")
plt.plot(x_axis,RMH, marker='s' , color='#ADD8E6')
plt.scatter(x_axis,RMH,s=100, marker='o' , label="RMH", color='#ADD8E6',linewidths=1,edgecolors="black")
plt.legend()
plt.title("Square")
plt.ylabel("Classification Error")
plt.show()
plt.close()

# sin

base = [0.2375,0.2039,0.1771,0.1572,0.1507]
PCA = []
PLS = []
MH = []
RMH = [0.2267,0.1932,0.1713,0.1567, 0.1488]

plt.plot(x_axis,base, marker='s' , color='#C0C0C0')
plt.scatter(x_axis,base,s=100, marker='s' , label="Base", color='silver',linewidths=1,edgecolors="black")
plt.plot(x_axis,RMH, marker='s' , color='#ADD8E6')
plt.scatter(x_axis,RMH,s=100, marker='o' , label="RMH", color='#ADD8E6',linewidths=1,edgecolors="black")
plt.legend()
plt.title("Sin")
plt.ylabel("Classification Error")
plt.show()
plt.close()
