from sklearn.manifold import TSNE
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv('Data\post-normalization_data.csv')

m= TSNE(n_components=2,       
        perplexity=50,        
        learning_rate=50,    
        n_iter=500,          
        random_state=42)      
tsne_features = m.fit_transform(data)
data['x']= tsne_features[:,0]
data['y']=tsne_features[:,1]
df = pd.DataFrame(tsne_features, columns=['x', 'y'])
df.to_csv('Data/reduced_data.csv', index=False)
print('Reduced data!')

plt.figure()
sns.scatterplot(x='x',y='y',data=data)  
plt.xlabel("x")
plt.ylabel("y")
plt.title("t-SNE Visualization")
plt.show()


