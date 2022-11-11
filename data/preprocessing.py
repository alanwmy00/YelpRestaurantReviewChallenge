import json
import numpy as np
import pandas as pd
User_id = []
Bus_id = []
Star = []
Useful = []
Funny = []
Cool = []
Text = []
I=1
with open('C:\\D_Disk\\UCLA_101C\\Final_Project\\archive\\yelp_academic_dataset_review.json', 'r',encoding='UTF-8') as f:
    for line in f:
        I += 1
        #if I==3:
           # break
        Temp_data = json.loads(line)
        User_id.append(Temp_data.get('user_id'))
        Bus_id.append(Temp_data.get('business_id'))
        Star.append(Temp_data.get('stars'))
        Useful.append(Temp_data.get('useful'))
        Cool.append(Temp_data.get('cool'))
        Funny.append(Temp_data.get('funny'))
        Text.append(Temp_data.get('text'))
        print(I)
Review_data = pd.DataFrame({
    'User_id':User_id,
    'Bus_id':Bus_id,
    'Star':Star,
    'Useful':Useful,
    'Cool':Cool,
    'Funny':Funny,
    'Review':Text
})
Review_data.to_csv('Review_Data')
#pd.read_csv('Review_Data')
#----------------------------------Loading the Users Data---------------------------------------------
User_id_list = []
User_review_count = []
Useful_count = []
Funny_count = []
Cool_count = []
elite = []
Fans_count = []
Average_Star = []
I = 1
with open('C:\\D_Disk\\UCLA_101C\\Final_Project\\archive\\yelp_academic_dataset_user.json', 'r',encoding='UTF-8') as f:
    for line in f:
        I += 1
        print(I)
        #if I==3:
            #break
        Temp_data = json.loads(line)
        User_id_list.append(Temp_data.get("user_id"))
        User_review_count.append(Temp_data.get("review_count"))
        Useful_count.append(Temp_data.get('useful'))
        Funny_count.append(Temp_data.get('funny'))
        Cool_count.append(Temp_data.get('cool'))
        elite.append(Temp_data.get('elite'))
        Fans_count.append(Temp_data.get('fans'))
        Average_Star.append(Temp_data.get('average_stars'))

       # if I == 10:
            #break
User_Data = pd.DataFrame({'User_id':User_id_list,
              'User_Review_count':User_review_count,
              'User_Useful_count':Useful_count,
              'User_Funny_count':Funny_count,
              'User_Cool_count':Cool_count,
              'Elite':elite,
              'User_Fans':Fans_count,
              'Users_Ave_Star':Average_Star})
User_Data.to_csv('User_data') # save data as csv

#--------------------------------Business dataset-------------
I = 1
Bus_id = []
State = []
City = []
Star = []
with open('C:\\D_Disk\\UCLA_101C\\Final_Project\\archive\\yelp_academic_dataset_business.json', 'r',encoding='UTF-8') as f:
    for line in f:
        Temp_data = json.loads(line)
        Bus_id.append(Temp_data.get("business_id"))
        State.append(Temp_data.get("state"))
        City.append(Temp_data.get('city'))
        #ACC.append(Temp_data.get('attributes').get('BusinessAcceptsCreditCards'))
        Star.append(Temp_data.get('stars'))

Business_Data = pd.DataFrame({'Bus_id':Bus_id,
              'State':State,
              'City':City,
              'Bus_Ave_Star':Star})
Business_Data.to_csv('Busi_data') # save data as csv


#---------------------------------------------------------------
import pandas as pd
Review = pd.read_csv('Review_data')
User = pd.read_csv('User_data')
Business = pd.read_csv('Busi_data')
User_subset = User[User['User_Review_count']>150].iloc[:,1::]
Business_list_CA = Business[Business['State'] == 'CA' ].iloc[:,1::]
Review_Usersub_CA = Review[(Review['User_id'].isin(User_subset['User_id'])) & Review['Bus_id'].isin(Business_list_CA['Bus_id'])]
Review_Final = Review_Usersub_CA.iloc[:,1::]
DF_RB = Review_Final.merge(Business_list_CA, how='left', on='Bus_id')
DF_Final = DF_RB.merge(User_subset, how='left', on='User_id')
DF_Final.to_csv('Data_Final')


#------------------------------------------Use TF-IDF to choose dictionary-----------------------
documents = []
import re
import nltk
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

X = DF_Final.Review
stemmer = WordNetLemmatizer()
for i in range(0, len(X)):
    # Remove all the special characters, like parathesis
    document = re.sub(r'\W', ' ', str(X[i]))
    # remove all single characters: like a, b, c, d
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    # Converting to Lowercase
    document = document.lower()
    # Lemmatization
    document = document.split()
    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    documents.append(document)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=1500, min_df=0.1, max_df=0.7, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents).toarray()


#------------------------------------Sentiment Lexicon------------------------------------------\
f_n = open("C:\\D_Disk\\UCLA_101C\\Final_Project\\negative-words.txt", "r")
Negative_words = f_n.readlines()
f_p = open("C:\\D_Disk\\UCLA_101C\\Final_Project\\positive-words.txt", "r")
Positive_words = f_p.readlines()
Voca = [i.rstrip('\n') for i in Positive_words] + [i.rstrip('\n') for i in Negative_words]
vectorizer = CountVectorizer(vocabulary=np.unique(Voca))
X = vectorizer.fit_transform(documents).toarray()

