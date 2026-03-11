import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    custom_colors = {
        'MA-F1': '#FF0000',  
        'F1': '#4477FF',  
        'PA-F1': '#FFA500',  
        'Event-based-F1': '#4CAF50', 
        'R-based-F1': '#00CED1', 
        'Affiliation-F1': '#BA55D3', 
        'AUC-PR': '#FF6B6B', 
        'AUC-ROC': '#FF69B4',  
        'VUS-PR': '#7f7f7f',  
        'VUS-ROC': '#bcbd22'  
    }
    plt.rcParams['font.family'] = 'Times New Roman'

    data_20 = {
        'Metric': ['MA-F1', 'F1', 'PA-F1', 'Event-based-F1', 'R-based-F1',
                   'Affiliation-F1', 'AUC-PR', 'AUC-ROC', 'VUS-PR', 'VUS-ROC'],
        'Rank1': [0.9980, 0.9231, 0.9231, 0.9231, 0.8889, 0.9359, 0.8571, 0.9808, 0.9695, 0.9959],
        'Rank2': [0.9305, 0.8571, 0.8571, 0.8571, 0.8571, 0.9302, 0.7500, 0.9615, 0.9883, 0.9955],
        'Rank3': [0.9091, 0.9091, 1.0000, 1.0000, 0.9655, 0.9953, 0.8646, 0.9167, 0.9899, 0.9917],
        'Rank4': [0.8000, 0.8000, 0.9091, 0.8571, 0.8119, 0.8511, 0.7292, 0.8333, 0.9750, 0.9803],
        'Rank5': [0.5738, 0.5455, 0.7692, 0.6667, 0.6879, 0.7863, 0.3938, 0.7115, 0.9650, 0.9779],
        'Rank6': [0.2258, 0.2000, 0.2000, 0.2500, 0.2857, 0.7100, 0.1979, 0.5256, 0.9526, 0.9665]
    }

    data_40 = {
        'Metric': ['MA-F1', 'F1', 'PA-F1', 'Event-based-F1', 'R-based-F1',
                   'Affiliation-F1', 'AUC-PR', 'AUC-ROC', 'VUS-PR', 'VUS-ROC'],
        'Rank1': [0.9888, 0.9600, 0.9600, 0.9600, 0.9412, 0.9851, 0.9231, 0.9750, 0.9925, 0.9975],
        'Rank2': [0.9496, 0.8889, 0.8889, 0.8889, 0.8571, 0.9335, 0.8000, 0.9250, 0.9883, 0.9955],
        'Rank3': [0.9091, 0.9091, 1.0000, 1.0000, 0.9655, 0.9838, 0.8958, 0.9167, 0.9964, 0.9963],
        'Rank4': [0.7368, 0.7368, 0.9091, 0.8571, 0.7879, 0.8324, 0.7396, 0.7917, 0.9882, 0.9884],
        'Rank5': [0.6906, 0.6364, 0.8000, 0.7241, 0.6682, 0.7674, 0.5646, 0.7167, 0.9793, 0.9848],
        'Rank6': [0.3371, 0.3000, 0.4545, 0.3750, 0.3384, 0.5803, 0.3750, 0.5000, 0.9749, 0.9776]
    }

    df_20 = pd.DataFrame(data_20).melt(id_vars='Metric', var_name='Rank', value_name='Value_20')
    df_40 = pd.DataFrame(data_40).melt(id_vars='Metric', var_name='Rank', value_name='Value_40')
    df = pd.merge(df_20, df_40, on=['Metric', 'Rank'])
    df['Mean'] = (df['Value_20'] + df['Value_40']) / 2 
    df['Error'] = df[['Value_20', 'Value_40']].max(axis=1) - df[['Value_20', 'Value_40']].min(axis=1) 

    plt.figure(figsize=(15, 8))
    models = ['Rank1', 'Rank2', 'Rank3', 'Rank4', 'Rank5', 'Rank6']
    x = np.arange(len(models))

    for i, metric in enumerate(df['Metric'].unique()):
        subset = df[df['Metric'] == metric].sort_values('Rank')
        y = subset['Mean'].values
        error = subset['Error'].values
        x_offset = x + (i % 3 - 1) * 0.08  
        plt.errorbar(x_offset, y,
                     yerr=error / 2,
                     fmt='none',
                     capsize=8,  
                     elinewidth=2.5, 
                     alpha=0.8,  
                     color=custom_colors[metric],
                     zorder=1)  
        plt.plot(x_offset, y,
                 marker=['o', 's', 'D', '^', 'v', '>', '<', 'p', '*', 'X'][i],  
                 linestyle=['-', '--', '-.', ':'][i % 4], 
                 color=custom_colors[metric],
                 markersize=8,
                 label=metric)
        # plt.errorbar(x, y, yerr=error/2, fmt='none', color=custom_colors[metric], capsize=8, alpha=1)  

    plt.xticks(x, labels=[f'Rank{i+1}' for i in range(len(models))], fontsize=37)
    plt.yticks(np.arange(0, 1.1, 0.2), fontsize=37)
    plt.xlabel('Models', fontsize=37)
    plt.ylabel('Value (Mean ± Range)', fontsize=37)
    plt.legend(
        loc='lower left',  
        bbox_to_anchor=(0.02, 0.02),  
        ncol=2,  
        fontsize=30,
        framealpha=0.8,
        borderaxespad=0.5,  
        columnspacing=1.2  
    )
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0.2, 1.05)  

    plt.tight_layout()

    plt.savefig(
        'performance_metrics.pdf',
        format='pdf',
        bbox_inches='tight',
        dpi=300,
        facecolor='white'
    )

    plt.show()
