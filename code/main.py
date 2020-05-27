import os
import pandas
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords 
from nltk import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re



parent_floder_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path=parent_floder_path+r"\cachedir\mbti_1.csv"

data=pandas.read_csv(csv_path)
#print(data)

def word_processing(series):
    # Lemmatizer | Stemmatizer
    stemmer = PorterStemmer()
    lemmatiser = WordNetLemmatizer()

    # Cache the stop words for speed 
    cachedStopWords = stopwords.words("english")
    for i in range(len(series)):
        OnePost=series[i]
        # One post
        # List all urls
        urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', OnePost)

        # Remove urls
        temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'link', OnePost)

        # Keep only words
        temp = re.sub("[^a-zA-Z]", " ", temp)

        # Remove spaces > 1
        temp = re.sub(' +', ' ', temp).lower()
        #print(len(OnePost))
        #print(len(temp))
        #print("stop")
        series[i]=temp
    return series
def word_processing_1(series):
    # Lemmatizer | Stemmatizer
    stemmer = PorterStemmer()
    lemmatiser = WordNetLemmatizer()

    # Cache the stop words for speed 
    cachedStopWords = stopwords.words("english")
    # One post
    post=series
    # List all urls
    print(len(post))
    post = post.str.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    print(len(post))
    # Remove urls
    post = post.astype(str).str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'link')
    print(len(post))
    # Keep only words
    post = post.astype(str).str.replace("[^a-zA-Z]", " ")
    print(len(post))
    # Remove spaces > 1
    post = post.str.match(r'^[a-z]+$')
    print(len(post))
    #print(type(OnePost))
    #print(type(temp))
    #series.str.replace(OnePost,temp)
    return post
def MBTI_to_binary(str):
    """
    introversion=1  : extraversion=0
    Intuition=1(N)  : Sensing=0 
    Thinking=1      :  Feeling=0
    Judging =1      :  Perceiving=0
    input : string, exp:MBTI
    output: list
    """
    MBTI_binary=[]
    for index in range(len(str)):
        char=str[index]
        if index==0:
            if char=='I':
                MBTI_binary.append(1)
            elif char=='E':
                MBTI_binary.append(0)
        elif index==1:
            if char=='N':
                MBTI_binary.append(1)
            elif char=='S':
                MBTI_binary.append(0)
        elif index==2:
            if char=='T':
                MBTI_binary.append(1)
            elif char=='F':
                MBTI_binary.append(0)
        else:
            if char=='J':
                MBTI_binary.append(1)
            elif char=='P':
                MBTI_binary.append(0)
    return MBTI_binary
def split_data_sentences(str):
    input=str.split("|||")
    while len(input)!=50:
        input.append("")
    return (input)
def split_raw_data_for_ml(dataframe):
    total_data=len(data.index)
    train, test = train_test_split(dataframe, test_size=0.4)
    cv, test = train_test_split(test, test_size=0.5)
    #print(train)
    #print(test)
    #print(cv)
    return train,cv,test








from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

vectorizer = TfidfVectorizer()
#data.iloc[6,1].str.replace("2","2")
print(data.columns[1])
df = data[data.columns[1]]
print((df))
new_df=word_processing(df)
print(new_df)

X = vectorizer.fit_transform(new_df)
print(len(vectorizer.get_feature_names()))
#print(MBTI_to_binary("ESFP"))
#print(len(data.index))
#split_raw_data_for_ml(data)
#print(word_processing((data.iloc[6,1])))
#iloc [row_index,0 is column]

