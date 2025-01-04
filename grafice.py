import numpy as np
import matplotlib.pyplot as plt
from seaborn import kdeplot,scatterplot

def plot_distributie(z:np.ndarray,y,clase,k=0):
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Plot distributie dupa axa discriminanta "+str(k+1),
                 fontdict={"fontsize":16,"color":"b"})
    ax.set_xlabel("z"+str(k+1))
    kdeplot(x=z[:,k],hue=y,hue_order=clase,ax=ax,warn_singular=False,fill=True)


def scatter_scoruri(z:np.ndarray,y,zg:np.ndarray,clase,k1=0,k2=1,etichete=None):
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Plot distributie dupa axe discriminante",
                 fontdict={"fontsize":16,"color":"b"})
    ax.set_xlabel("z"+str(k1+1))
    ax.set_ylabel("z"+str(k2+1))
    scatterplot(x=z[:,k1],y=z[:,k2],hue=y,hue_order=clase,ax=ax,legend=True)
    scatterplot(x=zg[:,k1],y=zg[:,k2],hue=clase,
                ax=ax,legend=False,s=200,marker="s",alpha=0.5)
    if etichete is not None:
        n = z.shape[0]
        for i in range(n):
            ax.text(z[i,k1],z[i,k2],etichete[i])


def show():
    plt.show()
