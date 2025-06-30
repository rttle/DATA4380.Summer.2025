#Libraries
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#function to calculate age
today=pd.to_datetime('2020-1-1')
def age(birthday):
    return relativedelta(today,birthday).years

#function to categorize age
def age_group(a):
    if a<25:
        return 'Youth'
    elif a>=25 and a<35:
        return 'Young Adults'
    elif a>=35 and a<45:
        return 'Midlife Adults'
    elif a>=45 and a<55:
        return 'Established Adults'
    elif a>=55 and a<65:
        return 'Early Retirees/ Well-Established Adults'
    elif a>=65 and a>75:
        return 'Retirees'
    elif a>=75:
        return 'Elderly'
    
#function to categorize country
def country_class(c):
    if c=='United States':
        return 'Developed'
    elif c=='Bangladesh':
        return 'Developing'
    elif c=='Brazil':
        return 'Emerging'
    elif c=='China':
        return 'Emerging'
    elif c=='India':
        return 'Developing'
    elif c=='Indonesia':
        return 'Emerging'
    elif c=='Mexico':
        return 'Emerging'
    elif c=='Nigeria':
        return 'Developing'
    elif c=='Pakistan':
        return 'Developing'
    elif c=='Russia':
        return 'Emerging'

#Biographical Table Preprocessing (Pre-Merge)
def bio_preprocess(bio):
    bio['birthday']=pd.to_datetime(bio['birthday'])
    bio['zip']=bio['zip'].apply(lambda x: f'{int(x):05d}' if pd.notnull(x) else x)
    bio['Age']=bio['birthday'].apply(lambda x: age(x) if pd.notnull(x) else x)
    bio['Age']=pd.to_numeric(bio['Age'], errors='coerce').astype('Int64')
    bio['Birthday Month']=bio['birthday'].dt.month
    bio.rename(columns={'id':'ID'},inplace=True)

    return bio

#Giving Table Preprocessing (Pre-Merge)
def giving_preprocess(gifts):
    gifts['gift date']=pd.to_datetime(gifts['gift date'])
    gifts['Gift Month']=gifts['gift date'].dt.month

    #splitting giving history
    past=gifts[gifts['gift date']< pd.to_datetime('2020-01-01')]
    future=gifts[gifts['gift date']>= pd.to_datetime('2020-01-01')]

    #feature engineering
    past['Time Between']=past.groupby('ID')['gift date'].diff().dt.days
    past['Hard Credit']=past['credit Type']=='Hard-Credit'
    past['Soft Credit']=past['credit Type']=='Soft-Credit'

    #Past: aggregation
    past_agg=past.groupby('ID').agg(
        First_Gift_Date=('gift date','min'),
        Last_Gift_Date=('gift date','max'),
        Median_Time_Between=('Time Between','median'),
        Average_Time_Between=('Time Between','mean'),
        Average_Amount=('gift amt','mean'),
        Number_HardCr=('Hard Credit', 'sum'),
        Number_SoftCr=('Soft Credit','sum'),
        Number_of_Gifts=('gift amt','count')
    ).reset_index()

    #post-aggregation feature engineering
    past_agg['Time from First Gift']=(today-past_agg['First_Gift_Date']).dt.days

    #Future (target variables): engineering
    target_vars=future.groupby('ID').agg(Next_Gift_Month=('Gift Month','min')).reset_index()
    target_vars['Give in 2020?']=target_vars['Next_Gift_Month'].apply(lambda x: 1 if x>0 else 0)
    target_vars['Give in 2020?']=pd.to_numeric(target_vars['Give in 2020?'])

    #merge giving features with target variables
    giving=pd.merge(past_agg,target_vars, on='ID', how='left')

    #NULL values
    giving['Median_Time_Between']=giving['Median_Time_Between'].fillna(giving['Time from First Gift'])
    giving['Average_Time_Between']= giving['Average_Time_Between'].fillna(giving['Time from First Gift'])
    giving['Next_Gift_Month']=giving['Next_Gift_Month'].fillna(0)
    giving['Give in 2020?']=giving['Give in 2020?'].fillna(0)

    return giving

#Merge tables and preprocess Function
def merge_preprocess(bio,gifts):
    #merge
    df=pd.merge(bio,gifts, on='ID', how='left')

    #NULL values resulting from merge
    df['Next_Gift_Month']=df['Next_Gift_Month'].fillna(0)
    df['Give in 2020?']=df['Give in 2020?'].fillna(0)
    df['deceased']=df['deceased'].fillna('N')
    df['Average_Amount']=df['Average_Amount'].fillna(0)
    df['Number_HardCr']=df['Number_HardCr'].fillna(0)
    df['Number_SoftCr']=df['Number_SoftCr'].fillna(0)
    df['Number_of_Gifts']=df['Number_of_Gifts'].fillna(0)

    #More complex NULL values
    firstday=pd.to_datetime('2016-10-21')
    daterange=(today-firstday).days
    daterange

    df['Median_Time_Between']=df['Median_Time_Between'].fillna(daterange)
    df['Average_Time_Between']=df['Average_Time_Between'].fillna(daterange)
    df['Time from First Gift']=df['Time from First Gift'].fillna(daterange)

    df['capacity']=df['capacity'].fillna(df['capacity'].mode()[0])
    df['capacity_source']=df['capacity_source'].fillna(df['capacity_source'].mode()[0])
    df['Age']=df['Age'].fillna(df['Age'].median())
    df['Birthday Month']=df['Birthday Month'].fillna(df['Birthday Month'].mode()[0])

    #trimming columns
    df=df.drop(columns=['birthday','zip','state','lat','lon','First_Gift_Date','Last_Gift_Date'],axis=1)

    #numeric to categorical
    df['Age']=df['Age'].apply(age_group)
    df['country']=df['country'].apply(country_class)
    df.rename(columns={'country':'Country Class'},inplace=True)

    #dropping identification columns
    df=df.iloc[:,3:]

    return df


#Encompassing Preprocessing Function
def load_preprocess(bio,gifts,binary=True,cities=True):
    #applying preprocessing to pre-merge tables
    bio=bio_preprocess(bio)
    gifts=giving_preprocess(gifts)

    #merging tables
    dataset=merge_preprocess(bio,gifts)

    #encoding
    if cities==True:
        cat_varnames=['Country Class','city','deceased','capacity','capacity_source','race','Age','Birthday Month','Number_HardCr','Number_SoftCr','Number_of_Gifts']
        encoder=OneHotEncoder()
        encoder.fit(dataset[cat_varnames])
        encoded=pd.DataFrame(encoder.transform(dataset[cat_varnames]).toarray())
    else:
        cat_varnames=['Country Class','deceased','capacity','capacity_source','race','Age','Birthday Month','Number_HardCr','Number_SoftCr','Number_of_Gifts']
        encoder=OneHotEncoder()
        encoder.fit(dataset[cat_varnames])
        encoded=pd.DataFrame(encoder.transform(dataset[cat_varnames]).toarray())

    #normalizing numerical columns
    nums=dataset.drop(columns=cat_varnames)
    nums=nums.drop(columns=['Next_Gift_Month','Give in 2020?']) #also dropping target variables
    scaler=MinMaxScaler()
    nums=pd.DataFrame(scaler.fit_transform(nums),
                  columns=['Median_Time_Between','Average_Time_Between','Average_Amount','Time from First Gift'])
    
    #recombining encoded, normalized columns, and target variables
    if binary==True:
        final=pd.concat([nums,encoded,dataset['Give in 2020?']],axis=1)
        return final
    else:
        final=pd.concat([nums,encoded,dataset['Next_Gift_Month']],axis=1)
        return final