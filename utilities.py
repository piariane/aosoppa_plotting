import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import copy
print(matplotlib.__version__)

###########################
# Read data               #
###########################


def read_data(energy, weight, reference, total_methods, requested_methods, include_weights):
    data = np.loadtxt('total_energy', usecols=range(3,3+len(total_methods),1))
    cc3_exci = np.loadtxt(reference)
    
    total_dict = dict()
    for i,j in enumerate(total_methods):
        total_dict[j] = data[:,i]
    
    # Initialize dictionaries
    exci_dict = dict()
    error_dict = dict()
    abs_error_dict = dict()
    
    # Sort methods into dictionary
    for i in requested_methods:
        exci_dict[i] = total_dict[i]
    
    # Calculate devatioan and abs deviation from cc3
    for i in requested_methods:
        error_dict[i] = exci_dict[i] - cc3_exci
        abs_error_dict[i] = abs(exci_dict[i] - cc3_exci)

    if (include_weights):
    
        data2 = np.loadtxt('total_weight', usecols=range(3,3+len(total_methods),1))
        
        total_weight = dict()
        for i,j in enumerate(total_methods):
            total_weight[j] = data2[:,i]

        weight_dict = dict()
        for i in requested_methods:
            weight_dict[i] = total_weight[i]

        return exci_dict, error_dict, abs_error_dict, weight_dict

    else:

        return exci_dict, error_dict, abs_error_dict


# dc stuff
def read_dc_contributions(dc_123_labels, filename):
    dc_labels = ['ph(0)', 'ph(2)', '2p2h(2)', 'sum']
#    dc_123_labels = ['ph(0)', 'ph(0) + ph(2)', 'ph(0) + ph(2) + 2p2h(2)']
#    dc_123_labels = ['ph(0)', '+ ph(2)', '+ 2p2h(2)']
#    
#    filename = 'shrpad_exci_contribution'

    dc_contributions = np.loadtxt(filename, usecols=[2,3,4])
    dc_sum = np.loadtxt(filename, usecols=[1])
    dc_sum = np.atleast_2d(dc_sum).T
    dc_total = np.concatenate((dc_contributions, dc_sum), axis=1)
    
    dc_ph0 = np.loadtxt(filename, usecols=[2])
    dc_ph2 = np.loadtxt(filename, usecols=[3])
    dc_2p2h2 = np.loadtxt(filename, usecols=[4])
    
    dc_1 = dc_ph0
    dc_2 = dc_ph0 + dc_ph2
    dc_3 = dc_ph0 + dc_ph2 + dc_2p2h2
    
    dc_1 = np.atleast_2d(dc_1).T
    dc_2 = np.atleast_2d(dc_2).T
    dc_3 = np.atleast_2d(dc_3).T
    
    dc_123 = np.concatenate((dc_1,(np.concatenate((dc_2,dc_3), axis=1))), axis=1)
    
    # Convert au to ev
    dc_total = dc_total*27.211385
    dc_123 = dc_123*27.211385
    
    # Initialize dictionaries
    dc_dict = dict()
    dc_123_dict = dict()
    dc_error_dict = dict()
    abs_dc_error_dict = dict()
    dc_123_error_dict = dict()
    abs_dc_123_error_dict = dict()
    
    # Sort dc results into dictionary
    for i,j in enumerate(dc_labels):
        dc_dict[j] = dc_total[:,i]
    
    # Sort dc results into dictionary
    for i,j in enumerate(dc_123_labels):
        dc_123_dict[j] = dc_123[:,i]
    
    # Calculate devatioan and abs deviation from cc3
    cc3_exci = np.loadtxt('cc3_exci')
    for i in dc_123_labels:
        dc_123_error_dict[i] = dc_123_dict[i] - cc3_exci
        abs_dc_123_error_dict[i] = abs(cc3_exci - dc_123_dict[i])
    
    # Calculate devatioan and abs deviation from cc3
    for i in dc_labels:
        dc_error_dict[i] = dc_dict[i] - cc3_exci
        abs_dc_error_dict[i] = abs(cc3_exci - dc_dict[i])

    return dc_123_dict, dc_123_error_dict, abs_dc_123_error_dict


#######################
#     Statistics      #
#######################

def statistics(exci_labels, error_dict, save_or_show, filename):
    cats = np.atleast_2d(np.array([['Count', 'Mean', 'Abs mean', 'Std. dev.', 'Max. dev.', 'Min. dev.', 'Abs. max.']], dtype=object))
    stats = np.zeros((7,len(exci_labels)))
    mets = np.atleast_2d([' ']+exci_labels)
    for i in range(len(exci_labels)):
        # Count
        stats[0,i] = len(error_dict[exci_labels[i]])
        # Mean 
        stats[1,i] = sum(error_dict[exci_labels[i]]) / stats[0,i]
        # Abs mean 
        stats[2,i] = sum(abs(error_dict[exci_labels[i]])) / stats[0,i]
        # Std dev
        stats[3,i] = np.std(error_dict[exci_labels[i]])
        # Max
        stats[4,i] = max(error_dict[exci_labels[i]])
        # Min
        stats[5,i] = min(error_dict[exci_labels[i]])
        # Abs max
        stats[6,i] = max(stats[4,i], abs(stats[5,i]))
    
    if save_or_show == 'save':
        stats = np.round(stats, decimals=2)
        stats = np.concatenate((cats.T, stats), axis=1)
        stats = np.concatenate((mets, stats), axis=0)
        np.savetxt(filename, stats, delimiter=' & ', fmt='%-10s', newline=' \\\\\n')
    if save_or_show == 'show':
        print(stats)
    return stats

def statistics_weights(exci_labels, error_dict):
    stats = np.zeros((5,len(exci_labels)))
    for i in range(len(exci_labels)):
        # Count
        stats[0,i] = len(error_dict[exci_labels[i]])
        # Mean 
        stats[1,i] = sum(error_dict[exci_labels[i]]) / stats[0,i]
        # Std dev
        stats[2,i] = np.std(error_dict[exci_labels[i]])
        # Min
        stats[3,i] = min(error_dict[exci_labels[i]])
        # Max
        stats[4,i] = max(error_dict[exci_labels[i]])
    
    np.savetxt('singlet_weight_stats.tex', stats, delimiter=' & ', fmt='%2.2f', newline=' \\\\\n')
    return stats



def plot_seaborn(labels, labels2, errors, fsize):
    
    # Format into dictionary
    methods_list = []
    error_list = []
    
    for key,key2 in zip(labels,labels2):
        for data in errors[key]:
            methods_list.append(key2)
            error_list.append(data)

    method_error = dict()
    method_error["method"] = methods_list
    method_error["error"] = error_list
    
    dot_size = fsize / 3.0
    line_size = fsize / 20.
    # Save as panda data frame
    my_data_formatted = pd.DataFrame(method_error)
    
    g = sns.stripplot(x="method", y="error", data=my_data_formatted, jitter=True,
#                      palette="gist_earth", split=True, linewidth=1, edgecolor='gray',
                      color="white", split=True, linewidth=line_size, edgecolor='black',
                      size=dot_size, alpha=1)
    return g

def figure_size(medium):

    if medium == 'A4':
        # Fig size
        fig_width = 427. / 72.27
        fig_height = 0.4 * fig_width
    
        # Font size
        fsize = 10

    elif medium == 'half_A4':
        # Fig size
        fig_width = (427. / 72.27)*0.5
        fig_height = 2.0 * 0.4 * fig_width
    
        # Font size
        fsize = 10

    elif medium == 'ppt':
        # Fig size
        fig_width = 10
        fig_height = 5
    
        # Font size
        fsize = 20

    else:
        print('Figure size choice not correct')

    return fig_width, fig_height, fsize

def include_table():
    # Include table with stats:
    rows = ['Mean', 'Abs. mean', 'Std. dev.', 'Max.']
    columns = exci_labels
    the_table = plt.table(cellText=stat, rowLabels=rows, colLabels=columns, loc='bottom', cellLoc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(16)
    the_table.scale(1, 2.1)

    # Layout stuff
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, bottom=0.3)
    plt.xticks([])

def plot_distributions(exci_labels, error_dict, stat, singlet_or_triplet, filename, figuresize):

    # Delete count row and adjust number of decimals
    stat = np.delete(stat, 0, 0)
    stat = np.around(stat, decimals=2, out=None)

    # Get figure and font size for either A4 or ppt
    # figuresize options: A4, half_A4, slide
    fig_width, fig_height, fsize = figure_size(figuresize)

    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plt.figure(num=1)

    sns.set_style('whitegrid')

    # Include histogram underneath
#    df2 = pd.DataFrame(stat.T, index=exci_labels, columns=['Mean', 'Abs. mean', 'Std. dev.', 'Max.'])
    df2 = pd.DataFrame(stat.T, index=exci_labels, columns=['Mean', 'Std. dev.', 'Min.', 'Max.'])
    ax = df2.plot(ax=fig.gca(),kind='bar', position=0.5)
#    legend = ax.legend(loc=1, fontsize=fsize)
    legend = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                       ncol=4, mode="expand", borderaxespad=0.)
    ax.tick_params(axis='x', labelsize=fsize)
    ax.tick_params(axis='y', labelsize=fsize)
    ax.set_ylabel('Statistical measures (bars) [eV]', size=fsize)
    ax.xaxis.grid(False)
#    ax.set_ylim(-1,11)
    ax.axhline(0, color='gray')
    if singlet_or_triplet == 'singlet':
        ax.set_title('117 singlet excited states', size=fsize)
    if singlet_or_triplet == 'triplet':
        ax.set_title('55 triplet excited states', size=fsize)
    plt.xticks(rotation='horizontal')

    sns.set_style({'axes.grid': False})

    g = ax.twinx()
    g = plot_seaborn(exci_labels, error_dict, fsize)
    g.set_ylabel('Distribution of errors (dots) [eV]', size=fsize)
    g.set_label(' ', size=fsize)
#    g.set_ylim(-5,7)
    g.set_ylim(40,140)
    g.axhline(0, color='gray')

    legend.set_frame_on('True')
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('white')

    # Remove left and right spines
    for i in ['left', 'right', 'top', 'bottom']:
        ax.spines[i].set_visible(False)
        g.spines[i].set_visible(False)

    plt.tight_layout()
#    plt.show()
    name = filename
    plt.savefig(name)

def plot_distributions_2(exci_labels, error_dict, stat_nostat, stat, singlet_or_triplet, save_or_show, filename, figuresize):

    nr_states = len(error_dict[exci_labels[0]])

    stat_measures = ['Mean', 'Abs mean', 'Std. dev.', 'Abs. max.']
    # Delete count row and adjust number of decimals
    if stat_nostat == 'stat':
        new_stat = np.zeros((len(stat_measures), len(exci_labels)))
        k = 0
        stat = np.delete(stat, 0, 0)
        for i in range(len(stat)):
            for j in stat_measures:
                if j == stat[i,0]:
                    new_stat[k,:] = stat[i,1:]
                    k = k+1
        stat = np.around(new_stat, decimals=2, out=None)

    # Get figure and font size for either A4 or ppt
    # figuresize options: A4, half_A4, slide
    fig_width, fig_height, fsize = figure_size(figuresize)

    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    #plt.figure(num=1)

    sns.set_style('whitegrid')

    colors = sns.color_palette(palette='Set1', n_colors=4)
    ecolors = ['black', 'black', 'black', 'black', 'black', 'black']

#    h = g.twinx()

    # Include histogram underneath
    if stat_nostat == 'stat':
        #new_xticks = ['RPA', '\nRPA(D)', 'HRPA', '\nHRPA(D)', 's-HRPA(D)', '\nSOPPA', 'SOPPA(CC2)', '\nSOPPA(CCSD)']
        #new_xticks = ['RPA', '\nRPA(D)', 'HRPA', '\nHRPA(D)', 's-HRPA(D)', '\nSOPPA']
        new_xticks = ['RPA', 'RPA(D)', 'HRPA', 'HRPA(D)', 's-HRPA(D)', 'SOPPA']
        df2 = pd.DataFrame(new_stat.T, index=exci_labels, columns=stat_measures)
#        df2 = pd.DataFrame(new_stat.T, index=new_xticks, columns=stat_measures)
        h = df2.plot(ax=fig.gca(),kind='bar', position=0.5, color=colors, edgecolor=ecolors, linewidth=0.5, zorder=2)
        # Add hatches
        matplotlib.rcParams.keys()
        matplotlib.rcParams['hatch.linewidth'] = 0.5
        bars = h.patches
        patterns =('////','++++','\\\\\\\\','xxx')
        hatches = [p for p in patterns for i in range(len(df2))]
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
## Put values on bars
#        for p in ax.patches:
#            ax.annotate(str(p.get_height()), (p.get_x()*1.005 - p.get_width()*0.5, p.get_height() * 1.005), fontsize=fsize*(1./2.))
        if figuresize == 'half_A4':
            legend = h.legend(bbox_to_anchor=(-0.085, 1.02, 1.20, .102), loc='upper center', ncol=2, fontsize=fsize, borderaxespad=-1., handletextpad=0.10, columnspacing=0.05)
        if figuresize == 'A4':
            #legend = h.legend(bbox_to_anchor=(-0.085, 1.02, 1.20, .102), loc='lower center', ncol=4, fontsize=fsize, borderaxespad=-1., handletextpad=0.5, columnspacing=1.0)
            legend = h.legend(
                              #bbox_to_anchor=(-0.085, 1.02, 1.20, .102), 
                              #bbox_to_anchor=(0.7, 1.08), 
                              bbox_to_anchor=(0.96, 0.84), 
                              loc=1, 
                              ncol=2, 
                              fontsize=fsize*0.9, 
                              borderaxespad=-1., 
                              handletextpad=0.5, 
                              columnspacing=0.5)
        if figuresize == 'ppt':
            legend = h.legend(bbox_to_anchor=(-0.085, 1.02, 1.20, .102), loc='upper center', ncol=4, fontsize=fsize, borderaxespad=-1., handletextpad=0.5, columnspacing=1.0)
        h.tick_params(axis='x', labelsize=fsize, pad=-10)
        h.set_yticks([0, 2, 4, 6, 8, 10])
        h.tick_params(axis='y', labelsize=fsize, length=0)
        plt.xticks(rotation='horizontal')
        #h.set_ylabel('Statistical measures (bars) [eV]', size=fsize)
        h.set_ylabel('Statistics (bars) [eV]', size=fsize)
        h.yaxis.set_label_position("right")
        h.xaxis.grid(False)
        h.set_ylim(-1.2, 11)
#        ax.set_ylim(-11,24)
        h.axhline(0, color='gray', zorder=1)
        h.axhline(4, color='gray', zorder=1)
    
#        if singlet_or_triplet == 'triplet':
#            #ax.text(0.8, 11.1, '8.8, 13.58, 13.76', fontsize=fsize*0.65)
#            if nr_states != 50:
#                ax.text(1., 11.1, '13.76', fontsize=fsize*0.65)

        legend.set_frame_on('True')
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('white')
        legend.remove()

    
        g = h.twinx()

        h.yaxis.tick_right()
        sns.set_style({'axes.grid': False})

    # Plot distributions
    #g = plot_seaborn(exci_labels, new_xticks, error_dict, fsize)
    g = plot_seaborn(exci_labels, exci_labels, error_dict, fsize)
    g.yaxis.tick_left()
    #g.set_ylabel('Distribution of errors (dots) [eV]', size=fsize)
    g.set_ylabel('Errors (dots) [eV]', size=fsize)
    g.yaxis.set_label_position("left")
    g.set_yticks([-4, -2, 0, 2, 4, 6])
    g.tick_params(axis='y', which='both', labelsize=fsize, length=0)
    #g.set_xlabel(' ', size=fsize/10.)
    g.tick_params(axis='x', labelsize=fsize, length=0)
    #g.tick_params(labelbottom='off')
    g.set_ylim(-5.2,7)
    #g.set_ylim(-5,15)
    #g.axhline(0, color='gray')

    g.grid(False)

    if singlet_or_triplet == 'singlet':
        ax.set_title('117 singlet excited states', size=fsize, color='black')
    if singlet_or_triplet == 'triplet':
        ax.set_title(str(nr_states)+' triplet excited states', size=fsize, color='black')
    plt.xticks(rotation='horizontal')

    plt.tight_layout(pad=0.5)


    # Remove left and right spines
    for i in ['left', 'right', 'top', 'bottom']:
        h.spines[i].set_visible(False)
        g.spines[i].set_visible(False)

    if save_or_show == 'show':
        plt.show()
    if save_or_show == 'save':
        name = filename
        plt.savefig(name)

def plot_reg(methods, exci_dict,error_dict, method1, method2, y1, y2, titlename, save_or_show, figuresize, filename):

    # Get figure and font size for either A4 or ppt
    # figuresize options: A4, half_A4, slide
    fig_width, fig_height, fsize = figure_size(figuresize)
    dot_size = (fsize / 3.)*5
    line_size = (fsize / 20.)*2
    edge_size = (fsize / 20.)/2.

    sns.set_style('whitegrid')
    fig, ax0 = plt.subplots(figsize=(fig_width, fig_height))

#    colors = cm.Set1
    colors = sns.color_palette(palette='Set1', n_colors=6)

#    keys_list = list(exci_dict.keys())
    keys_list = methods

    marker_list = ['o', 's', '>', 'd', '^', 'p']

    for i,j in zip(method1,method2):
        idx = method1.index(i)
        idx2 = keys_list.index(i)
#        ax = sns.regplot(exci_dict[j], error_dict[i], marker='o-', color=colors[idx2], ci=None, truncate=False, label=i, scatter_kws={'s':dot_size, 'edgecolors':'black', 'linewidth':line_size, 'zorder':2}, line_kws={'lw':line_size})
#        ax = sns.regplot(exci_dict[j], error_dict[i], color=colors[idx2], ci=None, truncate=False, label=i, scatter_kws={'s':dot_size, 'edgecolors':'black', 'linewidth':line_size, 'zorder':2, 'marker':'o'}, line_kws={'lw':line_size})
        ax = sns.regplot(exci_dict[j], error_dict[i], color=colors[idx2], ci=None, truncate=False, label=i, scatter_kws={'s':dot_size, 'edgecolors':'black', 'linewidth':edge_size, 'zorder':2}, marker=marker_list[idx2], line_kws={'lw':line_size})
    if y1!=0:
        ax.set_ylim(y1, y2)
#    ax.set_xlim(86, 100)
#    ax.set_xlim(-4, 4)
    ax.tick_params(labelsize=fsize)
    
    ax.axhline(0, color='gray', zorder=1)

#    ax.set_xlabel('Excitation energy (eV)', size=fsize)
    ax.set_xlabel(method2[0]+' '+titlename, size=fsize)
    ax.set_ylabel('Model - CC3 (eV)', size=fsize)
    
#    ax.set_ylim(-2,6)
#    ax.figure.set_size_inches(fig_width, fig_height)
    plt.tight_layout()
    plt.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.2)

#    box = ax.get_position()
#    ax.set_position([box.x0, box.y0, box.width, box.height * 0.9])
#    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.40), ncol=len(method1), fontsize=0.8*fsize)

#    legend = ax.legend(loc=2,fontsize=fsize)
#    legend = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=len(method1), mode="expand", borderaxespad=0., fontsize=fsize, labelspacing=3, handletextpad=0.1)
    legend = ax.legend(bbox_to_anchor=(-0.2, 1.02, 1.20, .102), loc='upper center', ncol=len(method1), fontsize=fsize, borderaxespad=-1., handletextpad=0.005, columnspacing=0.05)
#    legend = plt.legend(fontsize=fsize)

#    legend.set_frame_on('True')
#    legend.get_frame().set_facecolor('white')
#    legend.get_frame().set_edgecolor('white')

    for i in ['left', 'right', 'top', 'bottom']:
        ax.spines[i].set_visible(False)
        ax0.spines[i].set_visible(False)

    if save_or_show == 'show':
        plt.show()
    if save_or_show == 'save':
        plt.savefig(filename)

def plot_excitation(sing_trip, error_dict, exci_labels, save_show, filename):

    fig_width, fig_height, fsize = figure_size('A4')
    dot_size = 3
    line_width = 0.5
    line_style = '-'
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    nr_exci = len(error_dict[exci_labels[0]])
    x_values = np.linspace(1, nr_exci, nr_exci)
    if (nr_exci % 5 == 0):
        rounded_up = nr_exci
    else:
        rounded_up = (5 - nr_exci % 5) + nr_exci
    x2_values = np.arange(0,rounded_up,5)
    colors = sns.color_palette()
    for i in range(len(exci_labels)):
        ax1 = ax.plot(x_values,   error_dict[exci_labels[i]], label=exci_labels[i], marker='o', markersize=dot_size, linestyle='', color=colors[i])
        ax1 = ax.plot(x_values,   error_dict[exci_labels[i]], linestyle=line_style, linewidth=line_width, alpha=0.5, color=colors[i])
    ax.tick_params(labelsize=fsize)
    ax.set_xticks(x2_values)
    ax.set_xticks(x_values, minor=True)
    ax.grid(which='both')
    ax.set_ylabel('model - CC3 (eV)', fontsize=fsize)
    ax.set_xlim([0,nr_exci+1])
#    plt.tight_layout()
    plt.legend(fontsize=fsize)
    if sing_trip == 'singlet':
        ax.set_title('117 singlet excited states', size=fsize)
    if sing_trip == 'triplet':
        ax.set_title('55 triplet excited states', size=fsize)
    if save_show == 'show':
        plt.show()
    if save_show == 'save':
        plt.savefig(filename)

def plot_excitation_and_weights(sing_trip, error_dict, weight_dict, exci_labels, weight_labels, save_show, filename):

    fig_width, fig_height, fsize = figure_size('A4')
    dot_size = 3
    line_width = 0.5
    line_style = '-'
    fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))

    nr_exci = len(error_dict[exci_labels[0]])
    x_values = np.linspace(1, nr_exci, nr_exci)
    if (nr_exci % 5 == 0):
        rounded_up = nr_exci
    else:
        rounded_up = (5 - nr_exci % 5) + nr_exci
    x_minorvalues = np.arange(0,rounded_up,5)
    colors = sns.color_palette()
    for i in range(len(exci_labels)):
        line1, = ax1.plot(x_values, error_dict[exci_labels[i]], label='errors', marker='o', markersize=dot_size, linewidth=line_width, linestyle=' ', color=colors[i])
        line12, = ax1.plot(x_values,   error_dict[exci_labels[i]], linewidth=line_width, linestyle=line_style, color=colors[i], alpha=0.5)
    ax1.set_ylabel('Error relative to CC3 (eV)', fontsize=fsize)
    ax1.set_xticks(x_minorvalues)
    ax1.set_xticks(x_values, minor=True)
    ax1.tick_params(axis='x', labelsize=fsize)
    ax1.tick_params(axis='y', labelsize=fsize, colors=colors[0])
    ax1.grid(which='both')
    ax1.set_xlim([0,nr_exci+1])

    ax2 = ax1.twinx()
    for i in range(len(weight_labels)):
        line2,  = ax2.plot(x_values,   weight_dict[weight_labels[i]], label='weights', marker='o', markersize=dot_size, linewidth=line_width, linestyle=' ', color=colors[i+2])
        line22, = ax2.plot(x_values,   weight_dict[weight_labels[i]], linewidth=line_width, linestyle=line_style, color=colors[i+2], alpha=0.5)
    ax2.set_ylabel('Single excitation weights (%)', fontsize=fsize)
    ax2.tick_params(axis='y', labelsize=fsize, colors=colors[2])
    ax2.set_xlim([0,nr_exci+1])
    ax2.grid(False)

##    legend1 = plt.legend(handles=[line1], fontsize=fsize, loc=9)
#    legend1 = plt.legend(handles=[line1], fontsize=fsize, bbox_to_anchor=(1,1.05))
#    ax1 = plt.gca().add_artist(legend1)
##    legend2 = plt.legend(handles=[line2], fontsize=fsize, loc=1)
#    legend2 = plt.legend(handles=[line2], fontsize=fsize, bbox_to_anchor=(1,0.95))

    if sing_trip == 'singlet':
        plt.title('117 singlet excited states', fontsize=fsize)
    if sing_trip == 'triplet':
#        plt.title('55 singlet excited states', fontsize=fsize)
        plt.title('71 singlet excited states', fontsize=fsize)
    if save_show == 'show':
        plt.show()
    if save_show == 'save':
        plt.savefig(filename)

def plot_reg2():
    colors = cm.Set1
    ax = sns.regplot(cc3_exci, exci_dict_3['RPA'],     'o', color=colors(0./8.), ci=None, truncate=True, label='RPA')
    ax = sns.regplot(cc3_exci, exci_dict_3['RPA(D)'],  'o', color=colors(1./8.), ci=None, truncate=True, label='RPA(D)')
    ax = sns.regplot(cc3_exci, exci_dict_3['SOPPA'],   'o', color=colors(4./8.), ci=None, truncate=True, label='SOPPA')
    ax = sns.regplot(cc3_exci, exci_dict_3['HRPA(D)'], 'o', color=colors(2./8.), ci=None, truncate=True, label='HRPA(D)')
#    ax = sns.regplot(cc3_exci, exci_dict_3['HRPA'],    'o', color=colors(3./8.), ci=None, truncate=True, label='HRPA')
#    ax.set_ylim(-3, 5)
#    ax.set_xlim(1, 13)
    ax.tick_params(labelsize=14)
    
    ax.set_xlabel('Excitation energy (eV)', size=14)
    ax.set_ylabel('Model - CC3 (eV)', size=14)
    
    fig_width = 427. / 72.27
    fig_height = 0.8 * fig_width

    ax.figure.set_size_inches(fig_width, fig_height)
    plt.tight_layout()

    plt.legend(loc=2,fontsize=14, markerscale=2)
    plt.show()
#    plt.savefig("tzvp_error_vs_value.pdf")

def plot_hist(stat, exci_labels):
    #stats2 = np.delete(stats, 0, 0)
    stat_measures = ['Mean', 'Abs mean', 'Std. dev.', 'Abs. max.']
    new_stat = np.zeros((len(stat_measures), len(exci_labels)))
    k = 0
    stat = np.delete(stat, 0, 0)
    for i in range(len(stat)):
        for j in stat_measures:
            if j == stat[i,0]:
                new_stat[k,:] = stat[i,1:]
                k = k+1
    stat = np.around(new_stat, decimals=2, out=None)
    
    df2 = pd.DataFrame(new_stat.T, index=exci_labels, columns=['Mean', 'Abs. mean', 'Std. dev.', 'Max.'])
    ax = df2.plot(kind='bar')
    plt.legend(fontsize=20)
    ax.set_xticklabels(exci_labels, fontsize=20, rotation='horizontal')
    plt.yticks(fontsize=20)
    plt.ylabel('[eV]', fontsize=20)
    plt.title('117 singlet excited states', fontsize=20)
    plt.axhline(0, color='gray')
    plt.tight_layout()
    plt.show()
#    fig_width = 427. / 72.27
#    fig_height = 0.5 * fig_width
#    ax.figure.set_size_inches(fig_width, fig_height)
#    plt.savefig("tzvp_3_hist.pdf")
    plt.show()

# Exclude imaginary RPA excitations
def exclude_imag_rpa(exci_dict, error_dict, abs_error_dict, dc_true_false, weight_dict, exci_labels, type_55_or_71):

    if dc_true_false == 'dc_false':
        list_of_dicts = ['exci_dict', 'error_dict', 'abs_error_dict', 'weight_dict']
        list_of_dicts_2 = ['exci_dict_2', 'error_dict_2', 'abs_error_dict_2', 'weight_dict_2']
        list_of_dicts_4 = ['exci_dict_4', 'error_dict_4', 'abs_error_dict_4', 'weight_dict_4']

    if dc_true_false == 'dc_true':
        list_of_dicts = ['exci_dict', 'error_dict', 'abs_error_dict']
        list_of_dicts_2 = ['exci_dict_2', 'error_dict_2', 'abs_error_dict_2']
        list_of_dicts_4 = ['exci_dict_4', 'error_dict_4', 'abs_error_dict_4']

    dict_of_71 = dict()
    for j,i in enumerate(list_of_dicts):
        dict_copy = copy.deepcopy(eval(i))
        dict_of_71[list_of_dicts_2[j]] = dict_copy
    
    dict_of_55 = dict()
    for j in list_of_dicts_4:
        dict_of_55[j] = dict()
        for i in exci_labels:
            dict_of_55[j][i] = np.zeros(55)

    # Initiate counter
    k = 0

    if dc_true_false == 'dc_false':
        rpa = 'RPA'
    if dc_true_false == 'dc_true':
        rpa = 'ph(0)'

    for i in range(71):
        if (exci_dict[rpa][i] < 0.1):
#            print '\\rowcolor{lightgray} '
            for j in exci_labels:
                for l in dict_of_71:
                    dict_of_71[l][j][i] = None
        else:
#            print '\\rowcolor{white}     '
            for j in exci_labels:
                for m,n in enumerate(list_of_dicts):
                    dict_of_55[list_of_dicts_4[m]][j][k] = eval(n)[j][i]
            k = k + 1

    if type_55_or_71 == 71:
        return (dict_of_71[i] for i in list_of_dicts_2)
    if type_55_or_71 == 55:
        return (dict_of_55[i] for i in list_of_dicts_4)

# Exclude imaginary RPA excitations
def exclude_imag_rpa_old(exci_dict, error_dict, abs_error_dict, exci_labels, dc_labels, dc_123_labels, dc_error_dict, dc_dict, dc_123_error_dict, dc_123_dict):
    exci_dict_2 = copy.deepcopy(exci_dict)
    error_dict_2 = copy.deepcopy(error_dict)
    abs_error_dict_2 = copy.deepcopy(abs_error_dict)
    
#    # Exclude RPA excitations with error 'larger' than -3
#    exci_dict_3 = copy.deepcopy(exci_dict)
#    error_dict_3 = copy.deepcopy(error_dict)
#    abs_error_dict_3 = copy.deepcopy(abs_error_dict)
    
    error_dict_4 = dict()
    abs_error_dict_4 = dict()
    dc_error_dict_2 = dict()
    dc_dict_2 = dict()
    dc_123_error_dict_2 = dict()
    dc_123_dict_2 = dict()
    for i in exci_labels:
        error_dict_4[i] = np.zeros(55)
        abs_error_dict_4[i] = np.zeros(55)
    for l in dc_labels:
        dc_error_dict_2[l] = np.zeros(55)
        dc_dict_2[l] = np.zeros(55)
    for j in dc_123_labels:
        dc_123_error_dict_2[j] = np.zeros(55)
        dc_123_dict_2[j] = np.zeros(55)
    k = 0
    
    for i in range(71):
        if (exci_dict['RPA'][i] < 0.1):
            for j in exci_labels:
                exci_dict_2[j][i] = None
                error_dict_2[j][i] = None
                abs_error_dict_2[j][i] = None
#                exci_dict_3[j][i] = None
#                error_dict_3[j][i] = None
#                abs_error_dict_3[j][i] = None
#        elif (error_dict['RPA'][i] < -3):
#            for j in exci_labels:
#                exci_dict_3[j][i] = None
#                error_dict_3[j][i] = None
#                abs_error_dict_3[j][i] = None
        else:
            for j in exci_labels:
                error_dict_4[j][k] = error_dict[j][i]
                abs_error_dict_4[j][k] = abs_error_dict[j][i]
            for l in dc_labels:
                dc_error_dict_2[l][k] = dc_error_dict[l][i]
                dc_dict_2[l][k] = dc_dict[l][i]
            for m in dc_123_labels:
                dc_123_error_dict_2[m][k] = dc_123_error_dict[m][i]
                dc_123_dict_2[m][k] = dc_123_dict[m][i]
            k = k + 1
    return error_dict_4, abs_error_dict_4, dc_error_dict_2, dc_dict_2, dc_123_error_dict_2, dc_123_dict_2

# Exclude imaginary RPA excitations
def exclude_imag_rpa_new(exci_dict, error_dict, abs_error_dict, dc_true_false, weight_dict, exci_labels, type_55_or_71):

    if dc_true_false == 'dc_false':
        list_of_dicts = ['exci_dict', 'error_dict', 'abs_error_dict', 'weight_dict']
        list_of_dicts_2 = ['exci_dict_2', 'error_dict_2', 'abs_error_dict_2', 'weight_dict_2']
        list_of_dicts_4 = ['exci_dict_4', 'error_dict_4', 'abs_error_dict_4', 'weight_dict_4']

    if dc_true_false == 'dc_true':
        list_of_dicts = ['exci_dict', 'error_dict', 'abs_error_dict']
        list_of_dicts_2 = ['exci_dict_2', 'error_dict_2', 'abs_error_dict_2']
        list_of_dicts_4 = ['exci_dict_4', 'error_dict_4', 'abs_error_dict_4']

    dict_of_71 = dict()
    for j,i in enumerate(list_of_dicts):
        dict_copy = copy.deepcopy(eval(i))
        dict_of_71[list_of_dicts_2[j]] = dict_copy

    dict_of_55 = dict()
    for j in list_of_dicts_4:
        dict_of_55[j] = dict()
        for i in exci_labels:
            dict_of_55[j][i] = np.zeros(55)

    dict_of_50 = dict()
    for j in list_of_dicts_4:
        dict_of_50[j] = dict()
        for i in exci_labels:
            dict_of_50[j][i] = np.zeros(50)

    # Initiate counter
    k = 0
    c = 0

    if dc_true_false == 'dc_false':
        rpa = 'RPA'
    if dc_true_false == 'dc_true':
        #rpa = 'ph(0)'
        rpa = exci_labels[0]

#    for i in range(71):
#        if (exci_dict[rpa][i] < 0.1):
#            for j in ['RPA', 'RPA(D)']:
#                for l in dict_of_71:
#                    dict_of_71[l][j][i] = None
#            #continue
#        else:
#            for j in exci_labels:
#                for m,n in enumerate(list_of_dicts):
#                    dict_of_55[list_of_dicts_4[m]][j][k] = eval(n)[j][i]
#            k = k + 1


#    print dict_of_71['exci_dict_2']
#    print dict_of_55['exci_dict_4']

    for i in range(71):
        if (exci_dict[rpa][i] < 0.1):
#            for j in ['RPA', 'RPA(D)']:
#                for l in dict_of_71:
#                    dict_of_71[l][j][i] = None
            continue
        elif (error_dict[rpa][i] < -3):
#            for j in ['RPA', 'RPA(D)']:
#                for l in dict_of_71:
#                    dict_of_71[l][j][i] = None
            continue
        else:
            for j in exci_labels:
                for m,n in enumerate(list_of_dicts):
                    dict_of_50[list_of_dicts_4[m]][j][c] = eval(n)[j][i]
            c = c + 1

    if type_55_or_71 == 71:
        return (dict_of_71[i] for i in list_of_dicts_2)
    if type_55_or_71 == 55:
        return (dict_of_55[i] for i in list_of_dicts_4)
    if type_55_or_71 == 50:
        return (dict_of_50[i] for i in list_of_dicts_4)

