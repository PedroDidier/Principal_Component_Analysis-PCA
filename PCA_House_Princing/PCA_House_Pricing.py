import numpy as np
import pandas as pd

#changes our set to a z-score based dataframe
def standardize_df(df):
    
    for col in df.columns:
        datatype = df[col].dtypes

        if (datatype == 'float64') or (datatype == 'int64'):
            std = df[col].std()
            mean = df[col].mean()
            df[col] = (df[col] - mean) / std

    return df

#returns the coeficients from the projection of a vector 'v' in a vector 'u' of same dimensions
def vector_projection_coef(v, u):
    return np.dot(v, u) / np.dot(u, u)

#adapts a new vector on the space created by the most relevant eigenvectors of the covariance matrix transformation
def vector_transformation(v, obj):
    result = []
    
    for d_vec in obj:
        result.append(vector_projection_coef(v, d_vec))
        
    return result

#returns eigenvalues with their respective eigenvectors in ascending order
def component_analysis(df):
    df = df.drop(columns = 'Output')
    cov = df.cov()  #calculates the covariance matrix
    eigval, eigvec = np.linalg.eig(cov)   #gets the set of eigenvalues and eigenvectors
    comp_relevance = pd.DataFrame([(eigval[0], eigvec[0])], columns=['Eigenvalue', 'Eigenvector'])   #turn everything into a df
    
    for i in range(len(eigval)-1):
        comp_relevance = comp_relevance.append(pd.DataFrame([(eigval[i+1], eigvec[i+1])], 
                                                            columns=['Eigenvalue', 'Eigenvector']), ignore_index=True)
        
    comp_relevance = comp_relevance.sort_values('Eigenvalue', ascending=False).reset_index(drop=True)
    return comp_relevance

#returns the principal components based on a certain "amount of information preserved"
def principal_component_evaluation(aip, c_rel):
    total = c_rel['Eigenvalue'].sum() 
    cnt = 0
    
    for i, row in c_rel.iterrows():
        cnt = cnt + c_rel['Eigenvalue'][i]
        if(cnt*100/total > aip):
            principal_components = c_rel.iloc[:i+1]
            break
            
    return principal_components

#that part is where the real dimensionality reduction is applied
def make_reduced_df(df, p_comp):
    outcome = df['Output']
    df = df.drop(columns = 'Output')
    
    #creating the newdf and naming the new variables
    new_df = pd.DataFrame(columns=['x' + str(i) for i in range(len(p_comp))])
        
    for i in range(len(df.index)):
        new_df.loc[i] = vector_transformation(df.iloc[i].values, p_comp)
      
    new_df['Output'] = outcome
    return new_df

#reading the csv from a download link
df = pd.read_csv("http://abre.ai/house_p")

#no values on this df should be 0, therefore, they should be treated as NaN
df = df.replace(0, np.nan)

#making so that Parking_type and City_type become numerical variables
df['Parking_type'] = df['Parking_type'].replace({'No Parking' : 0, 'Open': 1, 'Covered': 2, 'Not Provided' : np.nan})
df['City_type'] = df['City_type'].replace({'CAT A' : 0, 'CAT B': 1, 'CAT C': 2})

#cleaning missing data that may mess things up
df = df.dropna().reset_index(drop=True)

#standardizing data without the output
p_house = df['Price_house']
df = df.drop(columns=['Price_house'])
df = standardize_df(df)
df['Output'] = p_house

#applying PCA retaining 90% of the df's information
cr = component_analysis(df)
pc = principal_component_evaluation(90, cr)
clean_df = make_reduced_df(df, pc['Eigenvector'])

#we could eliminate 2 columns in the process and still preserve a lot of information from the set
print(clean_df)