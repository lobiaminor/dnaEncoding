from matplotlib import pylab
import plotly.plotly as py

log = plt.figure()

data = ((3,1000), (10,3), (100,30), (500, 800), (50,1))

pylab.xlabel("Image")
pylab.ylabel("Entropy")
pylab.title("Haar vs CDF 9/7 wavelet transform")
pylab.gca().set_yscale('log')

dim = len(data[0])
w = 0.75
dimw = w / dim

x = pylab.arange(len(data))
for i in range(len(data[0])) :
    y = [d[i] for d in data]
    b = pylab.bar(x + i * dimw, y, dimw, bottom=0.001)
pylab.gca().set_xticks(x + w / 2)
pylab.gca().set_ylim( (0.001,1000))

plot_url = py.plot_mpl(log, filename='mpl-log')