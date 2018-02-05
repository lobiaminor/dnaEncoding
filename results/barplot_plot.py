from matplotlib import pylab
import plotly.plotly as py

log = plt.figure()

data = ((3.74,2.02,2.03), (3.58,2.09,1.95), (3.64,2.20,1.79), (3.53,2.30,1.89), (3.72,2.13,1.89), (3.36,2.20,1.91), (2.88,2.34,1.83), (3.79,1.97,1.93), (3.72,2.11,1.96), (3.65,2.10,1.97), (3.65,2.17,2.06), (3.86,2.03,2.10))

pylab.xlabel("Image")
pylab.ylabel("Entropy")
pylab.title("Haar vs CDF 9/7 wavelet transform")
pylab.gca().set_yscale('log')

dim = len(data[0])
w = 0.75
dimw = w / dim

x = ('lake', 'woman_blonde', 'woman_darkhair', 'cameraman', 'lena', 'jetplane', 'house', 'peppers_gray', 'livingroom', 'pirate', 'mandril_gray', 'walkbridge')
for i in range(len(data[0])) :
    y = [d[i] for d in data]
    b = pylab.bar(x + i * dimw, y, dimw, bottom=0.001)

plot_url = py.plot_mpl(log, filename='mpl-log')