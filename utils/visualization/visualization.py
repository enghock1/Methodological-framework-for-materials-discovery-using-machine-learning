import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def hist_predicted_output(sample, 
                          feature_name, 
                          class_name, 
                          bin_size=100, 
                          density=False, 
                          unit_value=None, 
                          tickxloc=None, 
                          tickyloc=None, 
                          savefig=False, 
                          showfig=True):
    
    
    fig, axs = plt.subplots(1,2, tight_layout=True, figsize = (10,4))
    fig.subplots_adjust(wspace=4)
    
    for i, n2 in enumerate(range(2)):
        xname = feature_name[i]
        datax = sample[:,i]

        maxx = datax.max()           
        minx = datax.min()
        nbins = np.linspace(minx,maxx,bin_size)

        cls0_datax = datax[sample[:,-1]==1].reshape((-1,1))
        cls1_datax = datax[sample[:,-1]==-1].reshape((-1,1))

        h = axs[n2].hist(cls0_datax, bins=nbins, color='#ff5000', alpha=0.6, label=class_name[0], density=density)
        h = axs[n2].hist(cls1_datax, bins=nbins, color='#0060ff', alpha=0.6, label=class_name[1], density=density)
        axs[n2].set_facecolor('xkcd:white')
        axs[n2].set_xlabel(xname,fontsize = 30)
        axs[n2].set_xlim([cls0_datax.min(), cls0_datax.max()])

        if tickxloc is not None:
            axs[n2].xaxis.set_major_locator(ticker.FixedLocator(tickxloc[i]))
        if tickyloc is not None:
            axs[n2].yaxis.set_major_locator(ticker.FixedLocator(tickyloc[i]))
            #axs[n2].yaxis.set_major_locator(ticker.MaxNLocator(4, integer=True))

        for tick in axs[n2].xaxis.get_major_ticks():
            tick.label.set_fontsize(27)

        for tick in axs[n2].yaxis.get_major_ticks():
            tick.label.set_fontsize(27)

        #if n2 == 0:
            #axs[n2].set_ylim([0,1])

        if n2 == 0:
            axs[n2].legend(facecolor='white', fontsize=24, loc=1)

        if n2 == 0:
            axs[n2].set_ylabel('Number of Material',fontsize = 30)

        if unit_value != None:
            if n2 == 1:
                if showfig == True:
                    axs[n2].set_title('unit ' + str(unit_value), fontsize = 35)
                    
                else:
                    String = ('unit ' + str(unit_value) + 
                              ' ' + class_name[0] + ': ' + cls0_datax.shape[0] + 
                              ' ' + class_name[1] + ': ' + cls1_datax.shape[0])
                    
                    axs[n2].set_title(String, fontsize = 35, y=1.1)
                    
        i += 1

    if savefig==True:
        unit_value_x = str(unit_value[0])
        unit_value_y = str(unit_value[1])
        fig.savefig('HOPO_' + '_unit_' + unit_value_x + '_' + unit_value_y + '.png')
    
    if showfig==False:
        plt.close()