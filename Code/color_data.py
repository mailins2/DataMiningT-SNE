
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Data/reduced_data.csv')
colorby = pd.read_csv('Data\pre-normalized_data.csv')

for column in range(15):
    plt.figure()
    sns.scatterplot(x='x',y='y',data=data, 
                hue=colorby.iloc[:,column+1])
    plt.legend(title=colorby.iloc[:,column+1].name)  
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("t-SNE Visualization")
    plt.title("t-SNE Visualization - color by "+colorby.iloc[:,column+1].name) 
    plt.savefig('Result/figure'+str(column+1)+'.png',dpi=300, 
                bbox_inches='tight')
    plt.show()
    plt.close()
    
print ('All results were saved!')


