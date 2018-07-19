from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def Process_Msg(msg, lower_case = True, stem = True, stop_words = True, gram = 1):
    if lower_case:
        msg = msg.lower()
    
    #Tokenize the Message
    words = word_tokenize(msg)
    words = [word for word in words if len(word) > 2]

    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [' '.join(words[i:i + gram])]
        return w
    
    #Remove stop words
    if stop_words:
        sw = stopwords.words('english')
        words = [word for word in words if word not in sw]
        
    #Apply Stemming on the words
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    
    return words