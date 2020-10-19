import re

def clean_doc(doc):
    doc = doc.lower()
    doc = re.sub(r"´,`", "\'", doc)
    doc = re.sub(r"\'s", " is", doc)
    doc = re.sub(r"\'ve", " have", doc)
    doc = re.sub(r"n\'t", " not", doc)
    doc = re.sub(r"\'re", " are", doc)
    doc = re.sub(r"\'d", " \'d", doc)
    doc = re.sub(r"\'ll", " will", doc)
    doc = re.sub(r",", " ", doc)
    doc = re.sub(r"!", " ! ", doc)
    doc = re.sub(r"\(", "", doc)
    doc = re.sub(r"\)", "", doc)
    doc = re.sub(r"\?", r" \? ", doc)
    doc = re.sub(r"\s{2,}", " ", doc)
    doc = re.sub(r"\?", " ", doc)
    doc = re.sub(r"[^A-Za-z0-9(),!?\\`]", " ", doc)
    
    doc = re.sub(r" br ", "", doc) # remove <br> tags
    
    tokens = doc.split()
    tokens = [word for word in tokens if len(word) > 1]
    return ' '.join(tokens)