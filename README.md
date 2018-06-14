

```python
from gensim import models
from nltk.corpus import stopwords
from database import Database
import re
import pickle
import numpy
from collections import OrderedDict
import string
import io
from gensim.models.wrappers import FastText
from nltk.stem import PorterStemmer
import numpy
```


```python
class Helper:
    """Helper class for text pre-processing tasks."""
    p = re.compile(r'[^\w\s]')
    
    @classmethod
    def sanitise_corpus(cls, corpus):
        sanitised_corpus = []   
        for raw_sentence in corpus:
            filtered_words = [word for word in raw_sentence.split() if word not in stopwords.words('english')]
            sanitised_words = [cls.p.sub('', word) for word in filtered_words]
            sanitised_corpus.append(list(set(filter(None, sanitised_words))))
        return sanitised_corpus

# Loading event data for model evaluation
event_data = Database.get_instance().list_companies_by_event('ijl_18')
event_data = [str(c['summary']).lower() for c in event_data]
event_data = Helper.sanitise_corpus(event_data)


# Here, I am using Word2vec and FastText model wrappers provided by Gensim. 
# Gensim ported the original C++ Word2vec or FastText library into python.

# Initialising Word2Vec model
word2vec_model = models.Word2Vec(size=300, window=10, min_count=1, workers=4, hs=1, sg=1)
# Building model vocabulary using words from event data
word2vec_model.build_vocab(event_data)
# Training Word2Vec model and presisting it in memory
word2vec_model.train(event_data, total_examples=word2vec_model.corpus_count, epochs=word2vec_model.iter)

# Initialising Fasttext model
fasttext_model = models.FastText(size=300, window=10, min_count=1, workers=4, hs=1, sg=1)
# Building model vocabulary using words from event data
fasttext_model.build_vocab(event_data)
# Training Word2Vec model and presisting it in memory
fasttext_model.train(event_data, total_examples=fasttext_model.corpus_count, epochs=fasttext_model.iter)

"""
The word vectors are stored in a KeyedVectors instance in *_model.wv. 
This separates the read-only word vector lookup operations in KeyedVectors from the training code in Word2Vec and Fasttext
To save memory we will grab KeyVector instance and clear original models presisted in memory.
""""
word2vec_model_wv = word2vec_model.wv
del word2vec_model
fasttext_model_wv = fasttext_model.wv
del fasttext_model
```


```python
# Getting vector of a word using word2vec model
word2vec_model_wv['diamond'] # return numpy vector of a word
```




    array([ 2.48286307e-01, -5.64355180e-02,  4.46890742e-02, -4.88192327e-02,
           -1.98240116e-01,  3.49513665e-02, -2.98750512e-02,  4.62635532e-02,
           -1.49095133e-01, -5.30804619e-02,  7.88651481e-02,  1.57252297e-01,
           -2.35441312e-01,  1.71836503e-02,  2.62964442e-02, -1.01933852e-01,
            1.71097308e-01,  1.13690436e-01,  9.75958407e-02, -2.38527442e-04,
           -2.36815140e-02,  8.89937114e-03,  1.60161391e-01, -1.84518039e-01,
           -4.57111560e-02, -1.30511925e-01,  1.15983278e-01,  2.81220134e-02,
           -1.54188335e-01,  2.03590333e-01, -1.31132044e-02, -3.71631682e-02,
            1.10348044e-02,  2.56354511e-02,  1.30372182e-01, -6.34903163e-02,
           -2.36494374e-03, -1.51552960e-01,  1.04012601e-02, -2.24542208e-02,
            9.31991190e-02, -1.48600459e-01,  1.18603028e-01,  1.06553361e-01,
           -4.59983461e-02,  4.40131612e-02, -1.02719478e-01, -3.17772180e-02,
            1.07664071e-01,  9.26099569e-02,  1.06270544e-01, -1.25118792e-01,
           -9.16944966e-02,  4.00068751e-03, -1.69821873e-01,  9.74044949e-02,
           -2.36400604e-01,  6.05269261e-02,  3.11887637e-02, -3.29254717e-02,
            1.11161452e-02, -5.31867146e-02, -2.47554500e-02,  4.00341824e-02,
            1.13739751e-01,  5.19985557e-02, -1.25105575e-01, -6.95842952e-02,
           -6.40230030e-02, -6.74609020e-02, -1.33724837e-02,  3.34777422e-02,
            5.66267073e-02,  1.04122147e-01,  1.06273666e-02, -3.44551280e-02,
           -4.35582623e-02,  8.50144997e-02,  1.12361692e-01, -2.08143413e-01,
            2.42738314e-02,  1.64467096e-01, -4.03382592e-02, -9.51004028e-03,
           -1.32795244e-01, -2.11431365e-02, -1.58981487e-01, -4.55943905e-02,
            2.46419664e-02,  3.10758203e-02, -5.35269566e-02,  6.87358482e-03,
           -3.37928608e-02, -1.49377987e-01, -1.69749297e-02, -4.72207293e-02,
           -5.75273894e-02, -1.12657301e-01,  2.66044915e-01, -8.35137293e-02,
           -1.69214442e-01,  9.44790542e-02, -9.75461453e-02,  1.70362502e-01,
            2.24321216e-01,  1.10926945e-02, -9.01855081e-02,  5.79390340e-02,
           -1.39444336e-01, -1.62510514e-01,  7.68100619e-02,  5.50280921e-02,
            1.57300696e-01, -5.85156493e-03,  3.69411297e-02, -4.16330770e-02,
            1.60195809e-02, -6.67677894e-02, -1.15271084e-01, -5.48959896e-02,
           -1.89911313e-02,  3.07648368e-02, -1.37091428e-01,  1.74718767e-01,
            7.12380782e-02,  6.22691810e-02, -6.53725490e-02, -2.59193555e-02,
           -2.85226759e-02,  7.52446130e-02,  4.11144570e-02,  1.15477704e-01,
            1.89663157e-01, -7.55747929e-02,  2.55587921e-02,  1.22958831e-01,
           -7.56668225e-02,  4.75344509e-02, -2.96834439e-01, -1.52653828e-01,
           -1.03319652e-01,  1.97237153e-02, -1.07492127e-01, -7.85266310e-02,
            1.24300554e-01, -1.60684913e-01,  1.36051446e-01, -7.00214971e-03,
            1.52362779e-01,  5.15005141e-02,  8.61639902e-02,  1.65568721e-02,
            5.40966168e-02,  4.33424823e-02,  9.11639109e-02, -1.01997845e-01,
            2.97924668e-01,  2.70745829e-02,  1.99464392e-02, -1.21212691e-01,
           -2.22050287e-02,  1.47077948e-01, -2.94153113e-02, -2.12010562e-01,
           -2.36241654e-01, -9.00867358e-02, -1.10599771e-02, -2.79824883e-02,
            8.71971250e-02,  1.90532714e-01,  1.27036422e-01, -1.46486402e-01,
            1.56313106e-01,  7.95527734e-03, -1.74012035e-01,  1.65330410e-01,
           -1.64467812e-01,  8.30970705e-02,  1.71484705e-02, -7.58139268e-02,
            2.31705248e-01, -2.02921003e-01, -3.34899463e-02, -7.29067847e-02,
           -5.96000366e-02,  4.49101701e-02, -1.17906835e-02,  2.27808356e-02,
           -8.90353788e-03,  2.98282541e-02, -1.70169689e-03,  1.55700762e-02,
            6.58463836e-02,  3.71731498e-04,  1.83914050e-01,  8.05284530e-02,
           -1.55595422e-01, -7.18572065e-02, -7.55379796e-02,  1.24580853e-01,
            1.23525091e-01, -2.64782719e-02,  4.81999442e-02, -1.97736844e-02,
           -8.87587741e-02, -1.30917102e-01,  2.32194904e-02, -1.27246365e-01,
            5.47367521e-02,  2.98439451e-02,  9.30211470e-02, -1.17375053e-01,
           -4.73043174e-02,  3.17977630e-02, -8.02433565e-02,  6.29881322e-02,
           -2.08134979e-01,  1.74856931e-02,  1.64460354e-02, -9.77146998e-02,
            1.22473594e-02, -1.03812711e-02,  2.92491596e-02,  6.86078966e-02,
           -9.05889198e-02,  1.99674312e-02, -1.23756882e-02,  4.48829792e-02,
            2.19724044e-01,  6.22809259e-03, -1.47569757e-02,  6.11104108e-02,
            6.70625195e-02,  8.79660696e-02, -9.84674171e-02, -5.73451295e-02,
            2.23615915e-02, -1.39997154e-01, -3.81745771e-02,  5.40614203e-02,
           -1.81763038e-01,  2.47065108e-02, -2.46889330e-02, -1.40272796e-01,
           -1.10365795e-02, -1.31158158e-01,  5.07341363e-02,  8.43064412e-02,
            8.79175290e-02, -7.36558996e-03, -2.92150546e-02, -1.53277099e-01,
           -8.46672431e-02,  3.85862663e-02,  4.93975542e-03,  1.07479371e-01,
           -5.76553643e-02, -1.17774643e-01, -1.90000385e-02,  1.10722393e-01,
            1.16200484e-02,  3.63477133e-02,  1.07744582e-01, -4.34846655e-02,
            9.00923312e-02, -5.49719706e-02, -6.07758909e-02, -8.77083689e-02,
            2.20272318e-01,  2.85059214e-03, -1.99164567e-03,  7.82013759e-02,
            2.96078846e-02, -2.99311262e-02, -2.14068424e-02, -1.16724232e-02,
           -1.92309290e-01,  1.15535289e-01, -9.83336046e-02,  3.63648683e-02,
           -7.13005811e-02,  5.08792177e-02,  5.55089228e-02,  1.10541299e-01,
            7.67843723e-02, -2.03329325e-02, -1.52626574e-01,  3.28207090e-02,
           -1.67642150e-03,  8.52342844e-02, -2.14864500e-02,  2.03230660e-02,
            1.13749184e-01,  4.00907397e-02,  2.53424257e-01,  1.47033721e-01,
           -1.59439489e-01,  1.06924810e-01, -5.18351346e-02,  4.23241034e-02],
          dtype=float32)




```python
# Getting similar words using word2vec
word2vec_model_wv.most_similar(['diamond']) # this could also be used to generate relevant tags
```




    [('finest', 0.9987034797668457),
     ('since', 0.9986259341239929),
     ('old', 0.9983115196228027),
     ('manufacturer', 0.9981980919837952),
     ('etc', 0.9978218078613281),
     ('vintage', 0.9977965354919434),
     ('gemstone', 0.9977301359176636),
     ('visit', 0.9976707100868225),
     ('specialize', 0.9976336359977722),
     ('sourced', 0.997613787651062)]




```python
# Getting the similary score between 2 words
word2vec_model_wv.similarity('ring', 'necklace')
```




    0.9968213082781366




```python
"""
However, word2vec model is not best for unseen words
for example if we try to find the similary score for the word that was not in training set,
it will break.
"""
word2vec_model_wv.most_similar('microsoft')
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-24-b23f001792da> in <module>()
          2 # for example if we try to find the similary score for the word that was not in training set,
          3 # it will break.
    ----> 4 word2vec_model_wv.most_similar('microsoft')
    

    /anaconda3/lib/python3.6/site-packages/gensim/models/keyedvectors.py in most_similar(self, positive, negative, topn, restrict_vocab, indexer)
        363                 mean.append(weight * word)
        364             else:
    --> 365                 mean.append(weight * self.word_vec(word, use_norm=True))
        366                 if word in self.vocab:
        367                     all_words.add(self.vocab[word].index)


    /anaconda3/lib/python3.6/site-packages/gensim/models/keyedvectors.py in word_vec(self, word, use_norm)
        272             return result
        273         else:
    --> 274             raise KeyError("word '%s' not in vocabulary" % word)
        275 
        276     def get_vector(self, word):


    KeyError: "word 'microsoft' not in vocabulary"



```python
"""
fasttext comes handy in this case
fasttext is provides an optimised implementatioln of skip-gram and cbow algorithms.
for the words not present in vocabulary it trys to break it down to revelant n-grams.
"""
fasttext_model_wv.most_similar('microsoft') 
```




    [('studded', 0.9993187785148621),
     ('maintained', 0.9993143081665039),
     ('german', 0.9993093609809875),
     ('pursuit', 0.9993017911911011),
     ('japanese', 0.999301552772522),
     ('male', 0.9992967247962952),
     ('folds', 0.9992942214012146),
     ('oneoff', 0.9992929697036743),
     ('cuban', 0.9992919564247131),
     ('ullmann', 0.9992835521697998)]




```python
"""
we know that word 'microsoft' is not within the training set, therefore the most similar words returned 
are at the far end of the vector space (with no context).
On the experimentation side, fasttext could also be used to find sectence similarity (like Ollie)
"""
event_data = Database.get_instance().list_companies_by_event('ijl_18')
event_data = [str(c['summary']).lower() for c in event_data]
tag_similarity = []
for i in range(len(event_data[:5])):
    new_match = {}
    new_match['summary'] = corpus_raw[i]
    new_match['match'] = fasttext_model_wv.n_similarity('traditional handmade diamonds'.lower().split(),
                                               corpus_raw[i].lower().split())
    tag_similarity.append(new_match)
    tag_similarity = sorted(tag_similarity, key=lambda k: k['match'])
    tag_similarity.reverse()
print(tag_similarity[:2])
```

    [{'summary': 'amber hall jewellery are an amber wholesalers who are truly passionate abour bringing you contemporary silver and gold, and baltic amber set jewellery. we also stock semi precious stone set and plain silver pieces in both traditional and more contemporary designs.', 'match': 0.9992067313422733}, {'summary': 'manufacturers of natural fancy color diamond and bridal jewelry. our natural fancy color diamond collection boasts an incredible array of blue, pink, yellow, green, and multicolor diamond pieces, documented with g.i.a. certificates. we also carry an a line of bridal jewelry for any sizes, shapes of colored and white diamonds.', 'match': 0.9989219551041687}]



```python
"""
According to Tomas Mikolov Word2Vec embedding has many advantages 
compared to earlier algorithms such as latent semantic analysis.
Sources for reading:
 - https://fasttext.cc/docs/en/support.html
 - https://www.tensorflow.org/tutorials/word2vec
 - http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
"""
```


```python
# On experimental side, I download pretrained vectors by Google and explore more about word2vec wrapper in gensim.
google_vectors = 'GoogleNews-vectors-negative300.bin'

pretrained_model = models.KeyedVectors.load_word2vec_format(google_vectors, binary=True)
print(pretrained_model)
```

    <gensim.models.keyedvectors.Word2VecKeyedVectors object at 0x112143780>



```python
# this is just to clear some space in momery
model = pretrained_model.wv
del pretrained_model
model
```

    /anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:2: DeprecationWarning: Call to deprecated `wv` (Attribute will be removed in 4.0.0, use self instead).
      





    <gensim.models.keyedvectors.Word2VecKeyedVectors at 0x112143780>




```python
model.vocab # a preview of model vocabulary
```




    {'</s>': <gensim.models.keyedvectors.Vocab at 0x112143cc0>,
     'in': <gensim.models.keyedvectors.Vocab at 0x1121435c0>,
     'for': <gensim.models.keyedvectors.Vocab at 0x112143828>,
     'that': <gensim.models.keyedvectors.Vocab at 0x1121435f8>,
     'is': <gensim.models.keyedvectors.Vocab at 0x112143a58>,
     'on': <gensim.models.keyedvectors.Vocab at 0x112143160>,
     '##': <gensim.models.keyedvectors.Vocab at 0x1121432e8>,
     'The': <gensim.models.keyedvectors.Vocab at 0x1121434a8>,
     'with': <gensim.models.keyedvectors.Vocab at 0x1121433c8>,
     'said': <gensim.models.keyedvectors.Vocab at 0x112143f60>,
     'was': <gensim.models.keyedvectors.Vocab at 0x112143ef0>,
     'the': <gensim.models.keyedvectors.Vocab at 0x112143c50>,
     'at': <gensim.models.keyedvectors.Vocab at 0x112143c88>,
     'not': <gensim.models.keyedvectors.Vocab at 0x1121431d0>,
     'as': <gensim.models.keyedvectors.Vocab at 0x112147828>,
     'it': <gensim.models.keyedvectors.Vocab at 0x112147b70>,
     'be': <gensim.models.keyedvectors.Vocab at 0x1121474a8>,
     'from': <gensim.models.keyedvectors.Vocab at 0x112147d68>,
     'by': <gensim.models.keyedvectors.Vocab at 0x112147978>,
     'are': <gensim.models.keyedvectors.Vocab at 0x112147da0>,
     'I': <gensim.models.keyedvectors.Vocab at 0x112147588>,
     'have': <gensim.models.keyedvectors.Vocab at 0x112147dd8>,
     'he': <gensim.models.keyedvectors.Vocab at 0x112147eb8>,
     'will': <gensim.models.keyedvectors.Vocab at 0x112147c50>,
     'has': <gensim.models.keyedvectors.Vocab at 0x112147400>,
     '####': <gensim.models.keyedvectors.Vocab at 0x112147780>,
     'his': <gensim.models.keyedvectors.Vocab at 0x1121477f0>,
     'an': <gensim.models.keyedvectors.Vocab at 0x1121475f8>,
     'this': <gensim.models.keyedvectors.Vocab at 0x1121479b0>,
     'or': <gensim.models.keyedvectors.Vocab at 0x112147860>,
     'their': <gensim.models.keyedvectors.Vocab at 0x112147f28>,
     'who': <gensim.models.keyedvectors.Vocab at 0x112147128>,
     'they': <gensim.models.keyedvectors.Vocab at 0x112147898>,
     'but': <gensim.models.keyedvectors.Vocab at 0x1121476d8>,
     '$': <gensim.models.keyedvectors.Vocab at 0x112147cc0>,
     'had': <gensim.models.keyedvectors.Vocab at 0x112147358>,
     'year': <gensim.models.keyedvectors.Vocab at 0x112147320>,
     'were': <gensim.models.keyedvectors.Vocab at 0x112147438>,
     'we': <gensim.models.keyedvectors.Vocab at 0x112147668>,
     'more': <gensim.models.keyedvectors.Vocab at 0x1121474e0>,
     '###': <gensim.models.keyedvectors.Vocab at 0x112147390>,
     'up': <gensim.models.keyedvectors.Vocab at 0x112147240>,
     'been': <gensim.models.keyedvectors.Vocab at 0x112147550>,
     'you': <gensim.models.keyedvectors.Vocab at 0x1121476a0>,
     'its': <gensim.models.keyedvectors.Vocab at 0x1121471d0>,
     'one': <gensim.models.keyedvectors.Vocab at 0x112142be0>,
     'about': <gensim.models.keyedvectors.Vocab at 0x112142f28>,
     'would': <gensim.models.keyedvectors.Vocab at 0x112142b38>,
     'which': <gensim.models.keyedvectors.Vocab at 0x112142d30>,
     'out': <gensim.models.keyedvectors.Vocab at 0x112142b00>,
     'can': <gensim.models.keyedvectors.Vocab at 0x112142e48>,
     'It': <gensim.models.keyedvectors.Vocab at 0x112142c18>,
     'all': <gensim.models.keyedvectors.Vocab at 0x112142358>,
     'also': <gensim.models.keyedvectors.Vocab at 0x112142c88>,
     'two': <gensim.models.keyedvectors.Vocab at 0x112142cc0>,
     'after': <gensim.models.keyedvectors.Vocab at 0x1121425c0>,
     'first': <gensim.models.keyedvectors.Vocab at 0x112142828>,
     'He': <gensim.models.keyedvectors.Vocab at 0x112142cf8>,
     'do': <gensim.models.keyedvectors.Vocab at 0x1121426a0>,
     'time': <gensim.models.keyedvectors.Vocab at 0x112142d68>,
     'than': <gensim.models.keyedvectors.Vocab at 0x112142550>,
     'when': <gensim.models.keyedvectors.Vocab at 0x112142a90>,
     'We': <gensim.models.keyedvectors.Vocab at 0x112142940>,
     'over': <gensim.models.keyedvectors.Vocab at 0x112142470>,
     'last': <gensim.models.keyedvectors.Vocab at 0x112142400>,
     'new': <gensim.models.keyedvectors.Vocab at 0x1121421d0>,
     'other': <gensim.models.keyedvectors.Vocab at 0x112142630>,
     'her': <gensim.models.keyedvectors.Vocab at 0x112142198>,
     'people': <gensim.models.keyedvectors.Vocab at 0x1121425f8>,
     'into': <gensim.models.keyedvectors.Vocab at 0x1121420b8>,
     'In': <gensim.models.keyedvectors.Vocab at 0x1121422e8>,
     'our': <gensim.models.keyedvectors.Vocab at 0x112142240>,
     'there': <gensim.models.keyedvectors.Vocab at 0x112142390>,
     'A': <gensim.models.keyedvectors.Vocab at 0x111f68ef0>,
     'she': <gensim.models.keyedvectors.Vocab at 0x111f687b8>,
     'could': <gensim.models.keyedvectors.Vocab at 0x111f68b38>,
     'just': <gensim.models.keyedvectors.Vocab at 0x111f68c50>,
     'years': <gensim.models.keyedvectors.Vocab at 0x111f68c88>,
     'some': <gensim.models.keyedvectors.Vocab at 0x111f689b0>,
     'U.S.': <gensim.models.keyedvectors.Vocab at 0x111f689e8>,
     'three': <gensim.models.keyedvectors.Vocab at 0x111f688d0>,
     'million': <gensim.models.keyedvectors.Vocab at 0x111f68da0>,
     'them': <gensim.models.keyedvectors.Vocab at 0x111f68470>,
     'what': <gensim.models.keyedvectors.Vocab at 0x111f68a58>,
     'But': <gensim.models.keyedvectors.Vocab at 0x111f68550>,
     'so': <gensim.models.keyedvectors.Vocab at 0x111f68668>,
     'no': <gensim.models.keyedvectors.Vocab at 0x111f680f0>,
     'like': <gensim.models.keyedvectors.Vocab at 0x111f68a20>,
     'if': <gensim.models.keyedvectors.Vocab at 0x111f684e0>,
     'only': <gensim.models.keyedvectors.Vocab at 0x111f68278>,
     'percent': <gensim.models.keyedvectors.Vocab at 0x111f68e10>,
     'get': <gensim.models.keyedvectors.Vocab at 0x111f68390>,
     'did': <gensim.models.keyedvectors.Vocab at 0x111f68710>,
     'him': <gensim.models.keyedvectors.Vocab at 0x111f68f28>,
     'game': <gensim.models.keyedvectors.Vocab at 0x111f68588>,
     'back': <gensim.models.keyedvectors.Vocab at 0x111f68080>,
     'because': <gensim.models.keyedvectors.Vocab at 0x111f68208>,
     'now': <gensim.models.keyedvectors.Vocab at 0x111f68048>,
     '#.#': <gensim.models.keyedvectors.Vocab at 0x111f469b0>,
     'before': <gensim.models.keyedvectors.Vocab at 0x111f46d68>,
     'company': <gensim.models.keyedvectors.Vocab at 0x111f46470>,
     'any': <gensim.models.keyedvectors.Vocab at 0x111f460b8>,
     'team': <gensim.models.keyedvectors.Vocab at 0x111f46160>,
     'against': <gensim.models.keyedvectors.Vocab at 0x111f46978>,
     'off': <gensim.models.keyedvectors.Vocab at 0x111f46358>,
     'This': <gensim.models.keyedvectors.Vocab at 0x111f46dd8>,
     'most': <gensim.models.keyedvectors.Vocab at 0x111f46fd0>,
     'made': <gensim.models.keyedvectors.Vocab at 0x111f46780>,
     'through': <gensim.models.keyedvectors.Vocab at 0x111f46a90>,
     'make': <gensim.models.keyedvectors.Vocab at 0x111f46f60>,
     'second': <gensim.models.keyedvectors.Vocab at 0x111f46940>,
     'state': <gensim.models.keyedvectors.Vocab at 0x111f46390>,
     'well': <gensim.models.keyedvectors.Vocab at 0x111f465c0>,
     'day': <gensim.models.keyedvectors.Vocab at 0x111f46b38>,
     'season': <gensim.models.keyedvectors.Vocab at 0x111f46e80>,
     'says': <gensim.models.keyedvectors.Vocab at 0x111f46550>,
     'week': <gensim.models.keyedvectors.Vocab at 0x111f46518>,
     'where': <gensim.models.keyedvectors.Vocab at 0x112145908>,
     'while': <gensim.models.keyedvectors.Vocab at 0x1121455c0>,
     'down': <gensim.models.keyedvectors.Vocab at 0x1121455f8>,
     'being': <gensim.models.keyedvectors.Vocab at 0x112145d68>,
     'government': <gensim.models.keyedvectors.Vocab at 0x112145748>,
     'your': <gensim.models.keyedvectors.Vocab at 0x112145a90>,
     '#-#': <gensim.models.keyedvectors.Vocab at 0x112145d30>,
     'home': <gensim.models.keyedvectors.Vocab at 0x112145da0>,
     'going': <gensim.models.keyedvectors.Vocab at 0x112145518>,
     'my': <gensim.models.keyedvectors.Vocab at 0x112145630>,
     'good': <gensim.models.keyedvectors.Vocab at 0x112145cc0>,
     'They': <gensim.models.keyedvectors.Vocab at 0x112145b00>,
     "'re": <gensim.models.keyedvectors.Vocab at 0x112145b70>,
     'should': <gensim.models.keyedvectors.Vocab at 0x112145e10>,
     'many': <gensim.models.keyedvectors.Vocab at 0x112145400>,
     'way': <gensim.models.keyedvectors.Vocab at 0x1121453c8>,
     'those': <gensim.models.keyedvectors.Vocab at 0x1121456a0>,
     'four': <gensim.models.keyedvectors.Vocab at 0x112145ba8>,
     'during': <gensim.models.keyedvectors.Vocab at 0x1121454a8>,
     'such': <gensim.models.keyedvectors.Vocab at 0x112145be0>,
     'may': <gensim.models.keyedvectors.Vocab at 0x112145a58>,
     'very': <gensim.models.keyedvectors.Vocab at 0x112145860>,
     'how': <gensim.models.keyedvectors.Vocab at 0x112145828>,
     'since': <gensim.models.keyedvectors.Vocab at 0x112145208>,
     'work': <gensim.models.keyedvectors.Vocab at 0x112145358>,
     'take': <gensim.models.keyedvectors.Vocab at 0x111f4a550>,
     'including': <gensim.models.keyedvectors.Vocab at 0x111f4ad68>,
     'high': <gensim.models.keyedvectors.Vocab at 0x111f4ae10>,
     'then': <gensim.models.keyedvectors.Vocab at 0x111f4abe0>,
     '%': <gensim.models.keyedvectors.Vocab at 0x111f4a0f0>,
     'next': <gensim.models.keyedvectors.Vocab at 0x111f4af98>,
     '#,###': <gensim.models.keyedvectors.Vocab at 0x111f4ab38>,
     'By': <gensim.models.keyedvectors.Vocab at 0x111f4a4e0>,
     'much': <gensim.models.keyedvectors.Vocab at 0x111f4af28>,
     'still': <gensim.models.keyedvectors.Vocab at 0x111f4a400>,
     'go': <gensim.models.keyedvectors.Vocab at 0x111f4a2b0>,
     'think': <gensim.models.keyedvectors.Vocab at 0x111f4a588>,
     'old': <gensim.models.keyedvectors.Vocab at 0x111f4aac8>,
     'even': <gensim.models.keyedvectors.Vocab at 0x111f4a518>,
     '#.##': <gensim.models.keyedvectors.Vocab at 0x111f4a0b8>,
     'world': <gensim.models.keyedvectors.Vocab at 0x111f4a668>,
     'see': <gensim.models.keyedvectors.Vocab at 0x111f4aa20>,
     'say': <gensim.models.keyedvectors.Vocab at 0x111f4a940>,
     'business': <gensim.models.keyedvectors.Vocab at 0x111f4dc50>,
     'five': <gensim.models.keyedvectors.Vocab at 0x111f4d128>,
     'told': <gensim.models.keyedvectors.Vocab at 0x111f4db70>,
     'under': <gensim.models.keyedvectors.Vocab at 0x111f4d550>,
     'us': <gensim.models.keyedvectors.Vocab at 0x111f4dcf8>,
     '1': <gensim.models.keyedvectors.Vocab at 0x111f4d4a8>,
     'these': <gensim.models.keyedvectors.Vocab at 0x111f4d9b0>,
     'If': <gensim.models.keyedvectors.Vocab at 0x111f4d438>,
     'right': <gensim.models.keyedvectors.Vocab at 0x111f4de80>,
     'And': <gensim.models.keyedvectors.Vocab at 0x111f4dba8>,
     'me': <gensim.models.keyedvectors.Vocab at 0x111f4d898>,
     'between': <gensim.models.keyedvectors.Vocab at 0x111f4df28>,
     'play': <gensim.models.keyedvectors.Vocab at 0x111f4d860>,
     'help': <gensim.models.keyedvectors.Vocab at 0x111f4d6d8>,
     '##,###': <gensim.models.keyedvectors.Vocab at 0x111f4dd68>,
     'market': <gensim.models.keyedvectors.Vocab at 0x111f4dac8>,
     'That': <gensim.models.keyedvectors.Vocab at 0x111f4d828>,
     'know': <gensim.models.keyedvectors.Vocab at 0x111f38e48>,
     'end': <gensim.models.keyedvectors.Vocab at 0x111f38358>,
     'AP': <gensim.models.keyedvectors.Vocab at 0x111f38208>,
     'long': <gensim.models.keyedvectors.Vocab at 0x111f38780>,
     'information': <gensim.models.keyedvectors.Vocab at 0x111f38da0>,
     'points': <gensim.models.keyedvectors.Vocab at 0x111f38470>,
     'does': <gensim.models.keyedvectors.Vocab at 0x111f38588>,
     'both': <gensim.models.keyedvectors.Vocab at 0x111f385c0>,
     'There': <gensim.models.keyedvectors.Vocab at 0x111f38fd0>,
     'part': <gensim.models.keyedvectors.Vocab at 0x111f38048>,
     'around': <gensim.models.keyedvectors.Vocab at 0x111f384a8>,
     'police': <gensim.models.keyedvectors.Vocab at 0x111f38d68>,
     'want': <gensim.models.keyedvectors.Vocab at 0x111f38dd8>,
     "'ve": <gensim.models.keyedvectors.Vocab at 0x111f38940>,
     'based': <gensim.models.keyedvectors.Vocab at 0x111f38160>,
     'For': <gensim.models.keyedvectors.Vocab at 0x111f38a90>,
     'got': <gensim.models.keyedvectors.Vocab at 0x111f381d0>,
     'third': <gensim.models.keyedvectors.Vocab at 0x111f38cf8>,
     'school': <gensim.models.keyedvectors.Vocab at 0x111f38828>,
     'left': <gensim.models.keyedvectors.Vocab at 0x111f387f0>,
     'another': <gensim.models.keyedvectors.Vocab at 0x111f350b8>,
     'country': <gensim.models.keyedvectors.Vocab at 0x111f35780>,
     'need': <gensim.models.keyedvectors.Vocab at 0x111f359b0>,
     '2': <gensim.models.keyedvectors.Vocab at 0x111f35518>,
     'best': <gensim.models.keyedvectors.Vocab at 0x111f35630>,
     'win': <gensim.models.keyedvectors.Vocab at 0x111f35080>,
     'quarter': <gensim.models.keyedvectors.Vocab at 0x111f35d30>,
     'use': <gensim.models.keyedvectors.Vocab at 0x111f35b70>,
     'today': <gensim.models.keyedvectors.Vocab at 0x111f353c8>,
     '##.#': <gensim.models.keyedvectors.Vocab at 0x111f35ba8>,
     'same': <gensim.models.keyedvectors.Vocab at 0x111f35b38>,
     'public': <gensim.models.keyedvectors.Vocab at 0x111f35978>,
     'run': <gensim.models.keyedvectors.Vocab at 0x111f358d0>,
     'Friday': <gensim.models.keyedvectors.Vocab at 0x112155ba8>,
     'set': <gensim.models.keyedvectors.Vocab at 0x1121559e8>,
     'month': <gensim.models.keyedvectors.Vocab at 0x112144208>,
     'top': <gensim.models.keyedvectors.Vocab at 0x1121441d0>,
     'billion': <gensim.models.keyedvectors.Vocab at 0x1121447b8>,
     'Tuesday': <gensim.models.keyedvectors.Vocab at 0x112144ac8>,
     'come': <gensim.models.keyedvectors.Vocab at 0x112144be0>,
     'Monday': <gensim.models.keyedvectors.Vocab at 0x112144898>,
     'She': <gensim.models.keyedvectors.Vocab at 0x112144048>,
     'city': <gensim.models.keyedvectors.Vocab at 0x112144128>,
     'place': <gensim.models.keyedvectors.Vocab at 0x1121440b8>,
     'night': <gensim.models.keyedvectors.Vocab at 0x1121444e0>,
     'six': <gensim.models.keyedvectors.Vocab at 0x112144710>,
     'each': <gensim.models.keyedvectors.Vocab at 0x112144390>,
     'Thursday': <gensim.models.keyedvectors.Vocab at 0x112144198>,
     '###,###': <gensim.models.keyedvectors.Vocab at 0x112144518>,
     'Wednesday': <gensim.models.keyedvectors.Vocab at 0x112144d68>,
     'here': <gensim.models.keyedvectors.Vocab at 0x231c91c18>,
     'You': <gensim.models.keyedvectors.Vocab at 0x231c914a8>,
     'group': <gensim.models.keyedvectors.Vocab at 0x231c91e10>,
     'really': <gensim.models.keyedvectors.Vocab at 0x231c91240>,
     'found': <gensim.models.keyedvectors.Vocab at 0x231c91978>,
     'As': <gensim.models.keyedvectors.Vocab at 0x231c91828>,
     'used': <gensim.models.keyedvectors.Vocab at 0x231c91278>,
     '3': <gensim.models.keyedvectors.Vocab at 0x231c91eb8>,
     'lot': <gensim.models.keyedvectors.Vocab at 0x231c91860>,
     "'m": <gensim.models.keyedvectors.Vocab at 0x231c91a90>,
     'money': <gensim.models.keyedvectors.Vocab at 0x231c91f60>,
     'put': <gensim.models.keyedvectors.Vocab at 0x231c917f0>,
     'games': <gensim.models.keyedvectors.Vocab at 0x231c91898>,
     'support': <gensim.models.keyedvectors.Vocab at 0x111f35f60>,
     'program': <gensim.models.keyedvectors.Vocab at 0x111f35470>,
     'half': <gensim.models.keyedvectors.Vocab at 0x111f35dd8>,
     'report': <gensim.models.keyedvectors.Vocab at 0x111f35240>,
     'family': <gensim.models.keyedvectors.Vocab at 0x111f35e48>,
     'months': <gensim.models.keyedvectors.Vocab at 0x111f35cf8>,
     'number': <gensim.models.keyedvectors.Vocab at 0x111f35e10>,
     'officials': <gensim.models.keyedvectors.Vocab at 0x111f35400>,
     'am': <gensim.models.keyedvectors.Vocab at 0x111f35a58>,
     'former': <gensim.models.keyedvectors.Vocab at 0x111f35128>,
     'own': <gensim.models.keyedvectors.Vocab at 0x111f334a8>,
     'man': <gensim.models.keyedvectors.Vocab at 0x111f332e8>,
     'Saturday': <gensim.models.keyedvectors.Vocab at 0x111f336d8>,
     'too': <gensim.models.keyedvectors.Vocab at 0x111f33c88>,
     'better': <gensim.models.keyedvectors.Vocab at 0x111f33668>,
     'days': <gensim.models.keyedvectors.Vocab at 0x111f337f0>,
     'came': <gensim.models.keyedvectors.Vocab at 0x111f33358>,
     'lead': <gensim.models.keyedvectors.Vocab at 0x111f33cc0>,
     'life': <gensim.models.keyedvectors.Vocab at 0x111f33940>,
     'American': <gensim.models.keyedvectors.Vocab at 0x111f33860>,
     '##-##': <gensim.models.keyedvectors.Vocab at 0x111f33e48>,
     'show': <gensim.models.keyedvectors.Vocab at 0x111f33f60>,
     'past': <gensim.models.keyedvectors.Vocab at 0x111f33ac8>,
     'took': <gensim.models.keyedvectors.Vocab at 0x111f33198>,
     'added': <gensim.models.keyedvectors.Vocab at 0x111f33e80>,
     'expected': <gensim.models.keyedvectors.Vocab at 0x111f33fd0>,
     'called': <gensim.models.keyedvectors.Vocab at 0x111f334e0>,
     'great': <gensim.models.keyedvectors.Vocab at 0x111f33cf8>,
     'State': <gensim.models.keyedvectors.Vocab at 0x111f33d68>,
     'services': <gensim.models.keyedvectors.Vocab at 0x111f330f0>,
     'children': <gensim.models.keyedvectors.Vocab at 0x111f33da0>,
     'hit': <gensim.models.keyedvectors.Vocab at 0x111f33f98>,
     'area': <gensim.models.keyedvectors.Vocab at 0x111f33048>,
     'system': <gensim.models.keyedvectors.Vocab at 0x111f332b0>,
     'every': <gensim.models.keyedvectors.Vocab at 0x111f4f0f0>,
     'pm': <gensim.models.keyedvectors.Vocab at 0x111f4f048>,
     'big': <gensim.models.keyedvectors.Vocab at 0x111f3ef28>,
     'service': <gensim.models.keyedvectors.Vocab at 0x111f3e208>,
     'few': <gensim.models.keyedvectors.Vocab at 0x111f3e6d8>,
     'per': <gensim.models.keyedvectors.Vocab at 0x111f3e048>,
     'members': <gensim.models.keyedvectors.Vocab at 0x111f3e128>,
     'Sunday': <gensim.models.keyedvectors.Vocab at 0x111f3eb70>,
     'early': <gensim.models.keyedvectors.Vocab at 0x111f3e978>,
     'point': <gensim.models.keyedvectors.Vocab at 0x111f3e940>,
     'start': <gensim.models.keyedvectors.Vocab at 0x111f3e198>,
     'companies': <gensim.models.keyedvectors.Vocab at 0x111f3e630>,
     'little': <gensim.models.keyedvectors.Vocab at 0x111f3e1d0>,
     '&': <gensim.models.keyedvectors.Vocab at 0x111f3eb38>,
     'case': <gensim.models.keyedvectors.Vocab at 0x111f3eb00>,
     'ago': <gensim.models.keyedvectors.Vocab at 0x111f3e320>,
     'local': <gensim.models.keyedvectors.Vocab at 0x111f3e668>,
     'according': <gensim.models.keyedvectors.Vocab at 0x111f3e9e8>,
     'never': <gensim.models.keyedvectors.Vocab at 0x111f3e5c0>,
     '5': <gensim.models.keyedvectors.Vocab at 0x111f3ebe0>,
     'without': <gensim.models.keyedvectors.Vocab at 0x111f3efd0>,
     'sales': <gensim.models.keyedvectors.Vocab at 0x111f3ef98>,
     'until': <gensim.models.keyedvectors.Vocab at 0x111f3e518>,
     'went': <gensim.models.keyedvectors.Vocab at 0x111eb6a58>,
     'players': <gensim.models.keyedvectors.Vocab at 0x111eb6b00>,
     '##th': <gensim.models.keyedvectors.Vocab at 0x111eb6d68>,
     'New_York': <gensim.models.keyedvectors.Vocab at 0x111eb6fd0>,
     'won': <gensim.models.keyedvectors.Vocab at 0x111eb6ba8>,
     'financial': <gensim.models.keyedvectors.Vocab at 0x111eb6e10>,
     'news': <gensim.models.keyedvectors.Vocab at 0x111eb6668>,
     '4': <gensim.models.keyedvectors.Vocab at 0x111eb6630>,
     'When': <gensim.models.keyedvectors.Vocab at 0x111eb65f8>,
     'share': <gensim.models.keyedvectors.Vocab at 0x111eb6550>,
     'several': <gensim.models.keyedvectors.Vocab at 0x111eb64e0>,
     'free': <gensim.models.keyedvectors.Vocab at 0x111eb64a8>,
     'away': <gensim.models.keyedvectors.Vocab at 0x111eb6400>,
     '##.##': <gensim.models.keyedvectors.Vocab at 0x111eb6208>,
     'already': <gensim.models.keyedvectors.Vocab at 0x111eb63c8>,
     'On': <gensim.models.keyedvectors.Vocab at 0x111eb6358>,
     'industry': <gensim.models.keyedvectors.Vocab at 0x111eb6320>,
     "'ll": <gensim.models.keyedvectors.Vocab at 0x111eb62b0>,
     'call': <gensim.models.keyedvectors.Vocab at 0x111eb6198>,
     'With': <gensim.models.keyedvectors.Vocab at 0x111eb60f0>,
     'students': <gensim.models.keyedvectors.Vocab at 0x111eb6160>,
     'line': <gensim.models.keyedvectors.Vocab at 0x111eb6080>,
     'available': <gensim.models.keyedvectors.Vocab at 0x111eb6048>,
     'County': <gensim.models.keyedvectors.Vocab at 0x111eb6710>,
     'making': <gensim.models.keyedvectors.Vocab at 0x111eb6b70>,
     'held': <gensim.models.keyedvectors.Vocab at 0x111f2d390>,
     'final': <gensim.models.keyedvectors.Vocab at 0x111f2d630>,
     '#:##': <gensim.models.keyedvectors.Vocab at 0x111f2d2e8>,
     'power': <gensim.models.keyedvectors.Vocab at 0x111f2df28>,
     'plan': <gensim.models.keyedvectors.Vocab at 0x111f2dcc0>,
     'might': <gensim.models.keyedvectors.Vocab at 0x111f2d940>,
     'least': <gensim.models.keyedvectors.Vocab at 0x111f2d6a0>,
     'look': <gensim.models.keyedvectors.Vocab at 0x111f2d048>,
     'forward': <gensim.models.keyedvectors.Vocab at 0x111f2d668>,
     'give': <gensim.models.keyedvectors.Vocab at 0x111f2d4a8>,
     'At': <gensim.models.keyedvectors.Vocab at 0x111f2d240>,
     'again': <gensim.models.keyedvectors.Vocab at 0x111f2deb8>,
     'later': <gensim.models.keyedvectors.Vocab at 0x111f2d358>,
     'full': <gensim.models.keyedvectors.Vocab at 0x111f2d7f0>,
     'must': <gensim.models.keyedvectors.Vocab at 0x111f2d320>,
     'things': <gensim.models.keyedvectors.Vocab at 0x111f2def0>,
     'major': <gensim.models.keyedvectors.Vocab at 0x111f2de80>,
     'community': <gensim.models.keyedvectors.Vocab at 0x111f2db38>,
     'announced': <gensim.models.keyedvectors.Vocab at 0x111f2d5c0>,
     'open': <gensim.models.keyedvectors.Vocab at 0x111f2d470>,
     'record': <gensim.models.keyedvectors.Vocab at 0x111f2d7b8>,
     'reported': <gensim.models.keyedvectors.Vocab at 0x111f2d9b0>,
     'court': <gensim.models.keyedvectors.Vocab at 0x111f2d780>,
     'working': <gensim.models.keyedvectors.Vocab at 0x111f2dfd0>,
     'able': <gensim.models.keyedvectors.Vocab at 0x111f37f98>,
     'something': <gensim.models.keyedvectors.Vocab at 0x111f37be0>,
     'president': <gensim.models.keyedvectors.Vocab at 0x111f37dd8>,
     'meeting': <gensim.models.keyedvectors.Vocab at 0x111f37550>,
     'keep': <gensim.models.keyedvectors.Vocab at 0x111f372b0>,
     'March': <gensim.models.keyedvectors.Vocab at 0x111f37048>,
     'future': <gensim.models.keyedvectors.Vocab at 0x111f37cc0>,
     'far': <gensim.models.keyedvectors.Vocab at 0x111f37b38>,
     'deal': <gensim.models.keyedvectors.Vocab at 0x111f37668>,
     'City': <gensim.models.keyedvectors.Vocab at 0x111f37b70>,
     'May': <gensim.models.keyedvectors.Vocab at 0x111f37128>,
     'development': <gensim.models.keyedvectors.Vocab at 0x111f370f0>,
     'University': <gensim.models.keyedvectors.Vocab at 0x111f37470>,
     'find': <gensim.models.keyedvectors.Vocab at 0x111f37e80>,
     'times': <gensim.models.keyedvectors.Vocab at 0x111f376a0>,
     'After': <gensim.models.keyedvectors.Vocab at 0x111f37c18>,
     'office': <gensim.models.keyedvectors.Vocab at 0x111f37780>,
     'led': <gensim.models.keyedvectors.Vocab at 0x111f37080>,
     'among': <gensim.models.keyedvectors.Vocab at 0x111f37208>,
     'June': <gensim.models.keyedvectors.Vocab at 0x111f37358>,
     'increase': <gensim.models.keyedvectors.Vocab at 0x111f32be0>,
     'China': <gensim.models.keyedvectors.Vocab at 0x111f32c18>,
     'John': <gensim.models.keyedvectors.Vocab at 0x111f32dd8>,
     'whether': <gensim.models.keyedvectors.Vocab at 0x111f32c88>,
     'cost': <gensim.models.keyedvectors.Vocab at 0x111f32630>,
     'security': <gensim.models.keyedvectors.Vocab at 0x111f32128>,
     'job': <gensim.models.keyedvectors.Vocab at 0x111f32a58>,
     'less': <gensim.models.keyedvectors.Vocab at 0x111f32898>,
     'head': <gensim.models.keyedvectors.Vocab at 0x111f320f0>,
     'seven': <gensim.models.keyedvectors.Vocab at 0x111f328d0>,
     'growth': <gensim.models.keyedvectors.Vocab at 0x111f32240>,
     'lost': <gensim.models.keyedvectors.Vocab at 0x111f32e48>,
     'pay': <gensim.models.keyedvectors.Vocab at 0x111f32ac8>,
     'looking': <gensim.models.keyedvectors.Vocab at 0x111f32b38>,
     'provide': <gensim.models.keyedvectors.Vocab at 0x111f329b0>,
     '6': <gensim.models.keyedvectors.Vocab at 0x111f325f8>,
     'To': <gensim.models.keyedvectors.Vocab at 0x111f32da0>,
     'plans': <gensim.models.keyedvectors.Vocab at 0x111f32b70>,
     'products': <gensim.models.keyedvectors.Vocab at 0x111f32ef0>,
     'car': <gensim.models.keyedvectors.Vocab at 0x111f30358>,
     'recent': <gensim.models.keyedvectors.Vocab at 0x111f30438>,
     'hard': <gensim.models.keyedvectors.Vocab at 0x111f30240>,
     'always': <gensim.models.keyedvectors.Vocab at 0x111f30e10>,
     'include': <gensim.models.keyedvectors.Vocab at 0x111f302b0>,
     'women': <gensim.models.keyedvectors.Vocab at 0x111f305f8>,
     'across': <gensim.models.keyedvectors.Vocab at 0x111f308d0>,
     'tax': <gensim.models.keyedvectors.Vocab at 0x111f30e48>,
     'water': <gensim.models.keyedvectors.Vocab at 0x111f30dd8>,
     'April': <gensim.models.keyedvectors.Vocab at 0x111f30f28>,
     'continue': <gensim.models.keyedvectors.Vocab at 0x111f30c88>,
     'important': <gensim.models.keyedvectors.Vocab at 0x111f30518>,
     'different': <gensim.models.keyedvectors.Vocab at 0x111f30d68>,
     'close': <gensim.models.keyedvectors.Vocab at 0x111f30780>,
     '7': <gensim.models.keyedvectors.Vocab at 0x111f30588>,
     'One': <gensim.models.keyedvectors.Vocab at 0x111f30fd0>,
     'late': <gensim.models.keyedvectors.Vocab at 0x111f309b0>,
     'decision': <gensim.models.keyedvectors.Vocab at 0x111f30978>,
     'current': <gensim.models.keyedvectors.Vocab at 0x111f30080>,
     'law': <gensim.models.keyedvectors.Vocab at 0x111f30668>,
     'within': <gensim.models.keyedvectors.Vocab at 0x111f2a860>,
     'along': <gensim.models.keyedvectors.Vocab at 0x111f2a3c8>,
     'played': <gensim.models.keyedvectors.Vocab at 0x111f2a320>,
     'move': <gensim.models.keyedvectors.Vocab at 0x111f2abe0>,
     'United_States': <gensim.models.keyedvectors.Vocab at 0x111f2a0f0>,
     'enough': <gensim.models.keyedvectors.Vocab at 0x111f2a1d0>,
     'become': <gensim.models.keyedvectors.Vocab at 0x111f2a828>,
     'side': <gensim.models.keyedvectors.Vocab at 0x111f2a4a8>,
     'national': <gensim.models.keyedvectors.Vocab at 0x111f2a630>,
     'Inc.': <gensim.models.keyedvectors.Vocab at 0x111f2a710>,
     'results': <gensim.models.keyedvectors.Vocab at 0x111f2a2e8>,
     'level': <gensim.models.keyedvectors.Vocab at 0x111f2aac8>,
     'loss': <gensim.models.keyedvectors.Vocab at 0x111f2a550>,
     'economic': <gensim.models.keyedvectors.Vocab at 0x111f2a9b0>,
     'coach': <gensim.models.keyedvectors.Vocab at 0x111f2aa58>,
     'near': <gensim.models.keyedvectors.Vocab at 0x111f2ab00>,
     'getting': <gensim.models.keyedvectors.Vocab at 0x111f2aa20>,
     'price': <gensim.models.keyedvectors.Vocab at 0x111f2a748>,
     'Department': <gensim.models.keyedvectors.Vocab at 0x111f2a2b0>,
     'event': <gensim.models.keyedvectors.Vocab at 0x111f2ac88>,
     'fourth': <gensim.models.keyedvectors.Vocab at 0x111f26cf8>,
     'change': <gensim.models.keyedvectors.Vocab at 0x111f267f0>,
     'All': <gensim.models.keyedvectors.Vocab at 0x111f26278>,
     'small': <gensim.models.keyedvectors.Vocab at 0x111f266a0>,
     'board': <gensim.models.keyedvectors.Vocab at 0x111f26f98>,
     'National': <gensim.models.keyedvectors.Vocab at 0x111f26e48>,
     'So': <gensim.models.keyedvectors.Vocab at 0x111f26940>,
     'goal': <gensim.models.keyedvectors.Vocab at 0x111f264a8>,
     'taken': <gensim.models.keyedvectors.Vocab at 0x111f26320>,
     'field': <gensim.models.keyedvectors.Vocab at 0x111f26668>,
     'prices': <gensim.models.keyedvectors.Vocab at 0x111f26f60>,
     'weeks': <gensim.models.keyedvectors.Vocab at 0x111f26390>,
     'men': <gensim.models.keyedvectors.Vocab at 0x111f26f28>,
     'asked': <gensim.models.keyedvectors.Vocab at 0x111f26048>,
     'eight': <gensim.models.keyedvectors.Vocab at 0x111f267b8>,
     'data': <gensim.models.keyedvectors.Vocab at 0x111f26e80>,
     'shot': <gensim.models.keyedvectors.Vocab at 0x111f260b8>,
     'New': <gensim.models.keyedvectors.Vocab at 0x111f26fd0>,
     'started': <gensim.models.keyedvectors.Vocab at 0x111f23a20>,
     'July': <gensim.models.keyedvectors.Vocab at 0x111f239e8>,
     'director': <gensim.models.keyedvectors.Vocab at 0x111f23128>,
     'President': <gensim.models.keyedvectors.Vocab at 0x111f23b00>,
     'party': <gensim.models.keyedvectors.Vocab at 0x111f23cc0>,
     'federal': <gensim.models.keyedvectors.Vocab at 0x111f23358>,
     'done': <gensim.models.keyedvectors.Vocab at 0x111f230b8>,
     'political': <gensim.models.keyedvectors.Vocab at 0x111f23320>,
     'minutes': <gensim.models.keyedvectors.Vocab at 0x111f23da0>,
     'taking': <gensim.models.keyedvectors.Vocab at 0x111f23748>,
     'Company': <gensim.models.keyedvectors.Vocab at 0x111f234a8>,
     'technology': <gensim.models.keyedvectors.Vocab at 0x111f23908>,
     'project': <gensim.models.keyedvectors.Vocab at 0x111f23438>,
     'center': <gensim.models.keyedvectors.Vocab at 0x111f237b8>,
     'leading': <gensim.models.keyedvectors.Vocab at 0x111f23ba8>,
     'issue': <gensim.models.keyedvectors.Vocab at 0x111f23550>,
     'though': <gensim.models.keyedvectors.Vocab at 0x111f23518>,
     'having': <gensim.models.keyedvectors.Vocab at 0x111f23400>,
     'period': <gensim.models.keyedvectors.Vocab at 0x111f235f8>,
     'likely': <gensim.models.keyedvectors.Vocab at 0x111f25358>,
     'scored': <gensim.models.keyedvectors.Vocab at 0x111f25710>,
     '8': <gensim.models.keyedvectors.Vocab at 0x111f25eb8>,
     'strong': <gensim.models.keyedvectors.Vocab at 0x111f25c88>,
     'series': <gensim.models.keyedvectors.Vocab at 0x111f25208>,
     'military': <gensim.models.keyedvectors.Vocab at 0x111f258d0>,
     'seen': <gensim.models.keyedvectors.Vocab at 0x111f257b8>,
     'trying': <gensim.models.keyedvectors.Vocab at 0x111f252e8>,
     'What': <gensim.models.keyedvectors.Vocab at 0x111f25f98>,
     'coming': <gensim.models.keyedvectors.Vocab at 0x111f25240>,
     'process': <gensim.models.keyedvectors.Vocab at 0x111f250b8>,
     'building': <gensim.models.keyedvectors.Vocab at 0x111f25cf8>,
     'behind': <gensim.models.keyedvectors.Vocab at 0x111f252b0>,
     'performance': <gensim.models.keyedvectors.Vocab at 0x111f25fd0>,
     'management': <gensim.models.keyedvectors.Vocab at 0x111f25080>,
     'Iraq': <gensim.models.keyedvectors.Vocab at 0x111f25898>,
     'saying': <gensim.models.keyedvectors.Vocab at 0x111f25d68>,
     'earlier': <gensim.models.keyedvectors.Vocab at 0x111f25c18>,
     'believe': <gensim.models.keyedvectors.Vocab at 0x111f25390>,
     'oil': <gensim.models.keyedvectors.Vocab at 0x111f25438>,
     'given': <gensim.models.keyedvectors.Vocab at 0x111f25908>,
     'Police': <gensim.models.keyedvectors.Vocab at 0x111f44e80>,
     'customers': <gensim.models.keyedvectors.Vocab at 0x111f44f98>,
     'due': <gensim.models.keyedvectors.Vocab at 0x111f443c8>,
     'following': <gensim.models.keyedvectors.Vocab at 0x111f44358>,
     'term': <gensim.models.keyedvectors.Vocab at 0x111f44e48>,
     'others': <gensim.models.keyedvectors.Vocab at 0x111f44b38>,
     'statement': <gensim.models.keyedvectors.Vocab at 0x111f44898>,
     'international': <gensim.models.keyedvectors.Vocab at 0x111f44fd0>,
     'economy': <gensim.models.keyedvectors.Vocab at 0x111f44a90>,
     'health': <gensim.models.keyedvectors.Vocab at 0x111f44d30>,
     'thing': <gensim.models.keyedvectors.Vocab at 0x111f449e8>,
     'Obama': <gensim.models.keyedvectors.Vocab at 0x111f446d8>,
     'return': <gensim.models.keyedvectors.Vocab at 0x111f447b8>,
     'killed': <gensim.models.keyedvectors.Vocab at 0x111f444e0>,
     'Washington': <gensim.models.keyedvectors.Vocab at 0x111f44c88>,
     'further': <gensim.models.keyedvectors.Vocab at 0x111f448d0>,
     'However': <gensim.models.keyedvectors.Vocab at 0x111f44978>,
     'doing': <gensim.models.keyedvectors.Vocab at 0x111f44588>,
     'face': <gensim.models.keyedvectors.Vocab at 0x111f44c18>,
     'low': <gensim.models.keyedvectors.Vocab at 0x111f44cc0>,
     'higher': <gensim.models.keyedvectors.Vocab at 0x111f44128>,
     'site': <gensim.models.keyedvectors.Vocab at 0x111f9f1d0>,
     'once': <gensim.models.keyedvectors.Vocab at 0x111f9f320>,
     'yet': <gensim.models.keyedvectors.Vocab at 0x111f9f2e8>,
     'hours': <gensim.models.keyedvectors.Vocab at 0x111f9f390>,
     'America': <gensim.models.keyedvectors.Vocab at 0x111f9f860>,
     'control': <gensim.models.keyedvectors.Vocab at 0x111f9f780>,
     'received': <gensim.models.keyedvectors.Vocab at 0x111f9ff28>,
     'rate': <gensim.models.keyedvectors.Vocab at 0x111f9fc18>,
     'career': <gensim.models.keyedvectors.Vocab at 0x111f9f5c0>,
     'Bush': <gensim.models.keyedvectors.Vocab at 0x111f9f630>,
     'teams': <gensim.models.keyedvectors.Vocab at 0x111f9f080>,
     'known': <gensim.models.keyedvectors.Vocab at 0x111f9fef0>,
     'offer': <gensim.models.keyedvectors.Vocab at 0x111f9f4a8>,
     'race': <gensim.models.keyedvectors.Vocab at 0x111f9ff60>,
     'ever': <gensim.models.keyedvectors.Vocab at 0x111f9fc88>,
     'experience': <gensim.models.keyedvectors.Vocab at 0x111f9fcf8>,
     'playing': <gensim.models.keyedvectors.Vocab at 0x111f9f400>,
     'name': <gensim.models.keyedvectors.Vocab at 0x111f9f748>,
     'possible': <gensim.models.keyedvectors.Vocab at 0x111f9f828>,
     'countries': <gensim.models.keyedvectors.Vocab at 0x111f9f240>,
     'Mr.': <gensim.models.keyedvectors.Vocab at 0x111f9f2b0>,
     'average': <gensim.models.keyedvectors.Vocab at 0x111f9f940>,
     'together': <gensim.models.keyedvectors.Vocab at 0x111f22cf8>,
     'using': <gensim.models.keyedvectors.Vocab at 0x111f227f0>,
     '9': <gensim.models.keyedvectors.Vocab at 0x111f222e8>,
     'cut': <gensim.models.keyedvectors.Vocab at 0x111f22390>,
     'While': <gensim.models.keyedvectors.Vocab at 0x111f22ba8>,
     'total': <gensim.models.keyedvectors.Vocab at 0x111f229b0>,
     'round': <gensim.models.keyedvectors.Vocab at 0x111f22588>,
     'young': <gensim.models.keyedvectors.Vocab at 0x111f22da0>,
     'nearly': <gensim.models.keyedvectors.Vocab at 0x111f22198>,
     'shares': <gensim.models.keyedvectors.Vocab at 0x111f22e10>,
     'member': <gensim.models.keyedvectors.Vocab at 0x111f224e0>,
     'campaign': <gensim.models.keyedvectors.Vocab at 0x111f220f0>,
     'media': <gensim.models.keyedvectors.Vocab at 0x111f22550>,
     'needs': <gensim.models.keyedvectors.Vocab at 0x111f22710>,
     'why': <gensim.models.keyedvectors.Vocab at 0x111f223c8>,
     'house': <gensim.models.keyedvectors.Vocab at 0x111f22c50>,
     'issues': <gensim.models.keyedvectors.Vocab at 0x111f22400>,
     'costs': <gensim.models.keyedvectors.Vocab at 0x111f22f60>,
     'fire': <gensim.models.keyedvectors.Vocab at 0x111f22f28>,
     '##-#': <gensim.models.keyedvectors.Vocab at 0x111f22358>,
     'victory': <gensim.models.keyedvectors.Vocab at 0x111f49358>,
     'player': <gensim.models.keyedvectors.Vocab at 0x111f49a20>,
     'began': <gensim.models.keyedvectors.Vocab at 0x111f496d8>,
     'sure': <gensim.models.keyedvectors.Vocab at 0x111f49e80>,
     'story': <gensim.models.keyedvectors.Vocab at 0x111f494a8>,
     'per_cent': <gensim.models.keyedvectors.Vocab at 0x111f499b0>,
     'North': <gensim.models.keyedvectors.Vocab at 0x111f49320>,
     'His': <gensim.models.keyedvectors.Vocab at 0x111f49198>,
     'staff': <gensim.models.keyedvectors.Vocab at 0x111f49b70>,
     'order': <gensim.models.keyedvectors.Vocab at 0x111f49c88>,
     'war': <gensim.models.keyedvectors.Vocab at 0x111f49c18>,
     'large': <gensim.models.keyedvectors.Vocab at 0x111f49668>,
     'interest': <gensim.models.keyedvectors.Vocab at 0x111f499e8>,
     'stock': <gensim.models.keyedvectors.Vocab at 0x111f491d0>,
     'food': <gensim.models.keyedvectors.Vocab at 0x111f49a90>,
     'research': <gensim.models.keyedvectors.Vocab at 0x111f496a0>,
     'key': <gensim.models.keyedvectors.Vocab at 0x111f492e8>,
     'India': <gensim.models.keyedvectors.Vocab at 0x111f494e0>,
     'South': <gensim.models.keyedvectors.Vocab at 0x111f9d0b8>,
     'morning': <gensim.models.keyedvectors.Vocab at 0x111f9db38>,
     'conference': <gensim.models.keyedvectors.Vocab at 0x111f9ddd8>,
     'senior': <gensim.models.keyedvectors.Vocab at 0x111f9d438>,
     'global': <gensim.models.keyedvectors.Vocab at 0x111f9d898>,
     'Center': <gensim.models.keyedvectors.Vocab at 0x111f9d828>,
     'death': <gensim.models.keyedvectors.Vocab at 0x111f9d630>,
     'person': <gensim.models.keyedvectors.Vocab at 0x111f9df28>,
     'thought': <gensim.models.keyedvectors.Vocab at 0x111f9d7f0>,
     'gave': <gensim.models.keyedvectors.Vocab at 0x111f9df98>,
     'feel': <gensim.models.keyedvectors.Vocab at 0x111f9d9b0>,
     'energy': <gensim.models.keyedvectors.Vocab at 0x111f9d390>,
     'history': <gensim.models.keyedvectors.Vocab at 0x111f9d128>,
     'recently': <gensim.models.keyedvectors.Vocab at 0x111f9d6a0>,
     'largest': <gensim.models.keyedvectors.Vocab at 0x111f9d8d0>,
     'No.': <gensim.models.keyedvectors.Vocab at 0x111f9d6d8>,
     'general': <gensim.models.keyedvectors.Vocab at 0x111f9d978>,
     'official': <gensim.models.keyedvectors.Vocab at 0x111f9deb8>,
     'released': <gensim.models.keyedvectors.Vocab at 0x111f9db70>,
     'wanted': <gensim.models.keyedvectors.Vocab at 0x111f9df60>,
     'meet': <gensim.models.keyedvectors.Vocab at 0x111f9da90>,
     'short': <gensim.models.keyedvectors.Vocab at 0x111f99128>,
     'outside': <gensim.models.keyedvectors.Vocab at 0x111f99320>,
     'running': <gensim.models.keyedvectors.Vocab at 0x111f99f28>,
     'live': <gensim.models.keyedvectors.Vocab at 0x111f99cc0>,
     'ball': <gensim.models.keyedvectors.Vocab at 0x111f994e0>,
     'online': <gensim.models.keyedvectors.Vocab at 0x111f99208>,
     'real': <gensim.models.keyedvectors.Vocab at 0x111f993c8>,
     'position': <gensim.models.keyedvectors.Vocab at 0x111f99a20>,
     'fact': <gensim.models.keyedvectors.Vocab at 0x111f992e8>,
     'fell': <gensim.models.keyedvectors.Vocab at 0x111f99cf8>,
     'nine': <gensim.models.keyedvectors.Vocab at 0x111f99978>,
     'December': <gensim.models.keyedvectors.Vocab at 0x111f99898>,
     'front': <gensim.models.keyedvectors.Vocab at 0x111f99198>,
     'action': <gensim.models.keyedvectors.Vocab at 0x111f996a0>,
     'defense': <gensim.models.keyedvectors.Vocab at 0x111f997f0>,
     'problem': <gensim.models.keyedvectors.Vocab at 0x111f99278>,
     'problems': <gensim.models.keyedvectors.Vocab at 0x111f99240>,
     'Mr': <gensim.models.keyedvectors.Vocab at 0x111f990b8>,
     'nation': <gensim.models.keyedvectors.Vocab at 0x111f99400>,
     'needed': <gensim.models.keyedvectors.Vocab at 0x111f98c18>,
     'special': <gensim.models.keyedvectors.Vocab at 0x111f98c50>,
     'January': <gensim.models.keyedvectors.Vocab at 0x111f98eb8>,
     'almost': <gensim.models.keyedvectors.Vocab at 0x111f982e8>,
     'chance': <gensim.models.keyedvectors.Vocab at 0x111f98240>,
     "'d": <gensim.models.keyedvectors.Vocab at 0x111f98fd0>,
     'result': <gensim.models.keyedvectors.Vocab at 0x111f98d68>,
     'West': <gensim.models.keyedvectors.Vocab at 0x111f98e10>,
     'September': <gensim.models.keyedvectors.Vocab at 0x111f98160>,
     'reports': <gensim.models.keyedvectors.Vocab at 0x111f98f98>,
     'leader': <gensim.models.keyedvectors.Vocab at 0x111f98978>,
     'investment': <gensim.models.keyedvectors.Vocab at 0x111f98550>,
     'yesterday': <gensim.models.keyedvectors.Vocab at 0x111f98358>,
     'Some': <gensim.models.keyedvectors.Vocab at 0x111f98f60>,
     'leaders': <gensim.models.keyedvectors.Vocab at 0x111f98668>,
     'ahead': <gensim.models.keyedvectors.Vocab at 0x111f98828>,
     'production': <gensim.models.keyedvectors.Vocab at 0x111f98748>,
     'comes': <gensim.models.keyedvectors.Vocab at 0x111f980b8>,
     'No': <gensim.models.keyedvectors.Vocab at 0x111f98898>,
     'runs': <gensim.models.keyedvectors.Vocab at 0x111f96f98>,
     'match': <gensim.models.keyedvectors.Vocab at 0x111f96f28>,
     'role': <gensim.models.keyedvectors.Vocab at 0x111f96ef0>,
     'kind': <gensim.models.keyedvectors.Vocab at 0x111f96b38>,
     'try': <gensim.models.keyedvectors.Vocab at 0x111f96e48>,
     'ended': <gensim.models.keyedvectors.Vocab at 0x111f96630>,
     'risk': <gensim.models.keyedvectors.Vocab at 0x111f96ac8>,
     'areas': <gensim.models.keyedvectors.Vocab at 0x111f96898>,
     'election': <gensim.models.keyedvectors.Vocab at 0x111f967f0>,
     'workers': <gensim.models.keyedvectors.Vocab at 0x111f96cc0>,
     'visit': <gensim.models.keyedvectors.Vocab at 0x111f96780>,
     'bring': <gensim.models.keyedvectors.Vocab at 0x111f96208>,
     'road': <gensim.models.keyedvectors.Vocab at 0x111f96b00>,
     'music': <gensim.models.keyedvectors.Vocab at 0x111f96860>,
     'study': <gensim.models.keyedvectors.Vocab at 0x111f96978>,
     'makes': <gensim.models.keyedvectors.Vocab at 0x111f960b8>,
     'often': <gensim.models.keyedvectors.Vocab at 0x111f96400>,
     'release': <gensim.models.keyedvectors.Vocab at 0x111f75be0>,
     'woman': <gensim.models.keyedvectors.Vocab at 0x111f75fd0>,
     'vote': <gensim.models.keyedvectors.Vocab at 0x111f75780>,
     'care': <gensim.models.keyedvectors.Vocab at 0x111f758d0>,
     'town': <gensim.models.keyedvectors.Vocab at 0x111f750f0>,
     'clear': <gensim.models.keyedvectors.Vocab at 0x111f75e48>,
     'comment': <gensim.models.keyedvectors.Vocab at 0x111f75400>,
     'budget': <gensim.models.keyedvectors.Vocab at 0x111f75b38>,
     'potential': <gensim.models.keyedvectors.Vocab at 0x111f75588>,
     'single': <gensim.models.keyedvectors.Vocab at 0x111f75080>,
     'markets': <gensim.models.keyedvectors.Vocab at 0x111f75f98>,
     'policy': <gensim.models.keyedvectors.Vocab at 0x111f75860>,
     'capital': <gensim.models.keyedvectors.Vocab at 0x111f75c50>,
     'saw': <gensim.models.keyedvectors.Vocab at 0x111f75278>,
     'access': <gensim.models.keyedvectors.Vocab at 0x111f759e8>,
     'weekend': <gensim.models.keyedvectors.Vocab at 0x111f75940>,
     'operations': <gensim.models.keyedvectors.Vocab at 0x111f75e80>,
     'whose': <gensim.models.keyedvectors.Vocab at 0x111f75e10>,
     'net': <gensim.models.keyedvectors.Vocab at 0x111f92b70>,
     'House': <gensim.models.keyedvectors.Vocab at 0x111f92d30>,
     'hand': <gensim.models.keyedvectors.Vocab at 0x111f92ba8>,
     'increased': <gensim.models.keyedvectors.Vocab at 0x111f92e48>,
     'charges': <gensim.models.keyedvectors.Vocab at 0x111f924e0>,
     'winning': <gensim.models.keyedvectors.Vocab at 0x111f92f60>,
     'trade': <gensim.models.keyedvectors.Vocab at 0x111f926a0>,
     'These': <gensim.models.keyedvectors.Vocab at 0x111f92a20>,
     'income': <gensim.models.keyedvectors.Vocab at 0x111f92390>,
     'value': <gensim.models.keyedvectors.Vocab at 0x111f92470>,
     'involved': <gensim.models.keyedvectors.Vocab at 0x111f92c50>,
     'Bank': <gensim.models.keyedvectors.Vocab at 0x111f92160>,
     'November': <gensim.models.keyedvectors.Vocab at 0x111f92e10>,
     'bill': <gensim.models.keyedvectors.Vocab at 0x111f92748>,
     'compared': <gensim.models.keyedvectors.Vocab at 0x111f92908>,
     'anything': <gensim.models.keyedvectors.Vocab at 0x111f92b00>,
     'manager': <gensim.models.keyedvectors.Vocab at 0x111f92978>,
     'Texas': <gensim.models.keyedvectors.Vocab at 0x111f92780>,
     'property': <gensim.models.keyedvectors.Vocab at 0x111f925c0>,
     'stop': <gensim.models.keyedvectors.Vocab at 0x111f64630>,
     'annual': <gensim.models.keyedvectors.Vocab at 0x111f644a8>,
     'private': <gensim.models.keyedvectors.Vocab at 0x111f64cc0>,
     'contract': <gensim.models.keyedvectors.Vocab at 0x111f647b8>,
     'died': <gensim.models.keyedvectors.Vocab at 0x111f649e8>,
     'Now': <gensim.models.keyedvectors.Vocab at 0x111f64048>,
     'hope': <gensim.models.keyedvectors.Vocab at 0x111f64668>,
     'product': <gensim.models.keyedvectors.Vocab at 0x111f64ef0>,
     'fans': <gensim.models.keyedvectors.Vocab at 0x111f648d0>,
     'lower': <gensim.models.keyedvectors.Vocab at 0x111f64198>,
     'demand': <gensim.models.keyedvectors.Vocab at 0x111f64a20>,
     'News': <gensim.models.keyedvectors.Vocab at 0x111f644e0>,
     'David': <gensim.models.keyedvectors.Vocab at 0x111f67588>,
     'club': <gensim.models.keyedvectors.Vocab at 0x111f67a58>,
     'comments': <gensim.models.keyedvectors.Vocab at 0x111f67390>,
     'film': <gensim.models.keyedvectors.Vocab at 0x111f67cc0>,
     'yards': <gensim.models.keyedvectors.Vocab at 0x111f679e8>,
     'quality': <gensim.models.keyedvectors.Vocab at 0x111f67550>,
     'currently': <gensim.models.keyedvectors.Vocab at 0x111f675f8>,
     'events': <gensim.models.keyedvectors.Vocab at 0x111f67c88>,
     'addition': <gensim.models.keyedvectors.Vocab at 0x111f67978>,
     'couple': <gensim.models.keyedvectors.Vocab at 0x111f67e48>,
     'schools': <gensim.models.keyedvectors.Vocab at 0x111f670f0>,
     'attack': <gensim.models.keyedvectors.Vocab at 0x111f67dd8>,
     'region': <gensim.models.keyedvectors.Vocab at 0x111f67320>,
     'latest': <gensim.models.keyedvectors.Vocab at 0x111f67ef0>,
     'opportunity': <gensim.models.keyedvectors.Vocab at 0x111f67208>,
     'worked': <gensim.models.keyedvectors.Vocab at 0x111f67be0>,
     'course': <gensim.models.keyedvectors.Vocab at 0x111f709b0>,
     'bad': <gensim.models.keyedvectors.Vocab at 0x111f707b8>,
     'fall': <gensim.models.keyedvectors.Vocab at 0x111f70a20>,
     'Group': <gensim.models.keyedvectors.Vocab at 0x111f70eb8>,
     'October': <gensim.models.keyedvectors.Vocab at 0x111f70978>,
     'jobs': <gensim.models.keyedvectors.Vocab at 0x111f70160>,
     'list': <gensim.models.keyedvectors.Vocab at 0x111f708d0>,
     'let': <gensim.models.keyedvectors.Vocab at 0x111f70c18>,
     'however': <gensim.models.keyedvectors.Vocab at 0x111f70b38>,
     'chief': <gensim.models.keyedvectors.Vocab at 0x111f70128>,
     'summer': <gensim.models.keyedvectors.Vocab at 0x111f70898>,
     'programs': <gensim.models.keyedvectors.Vocab at 0x111f74048>,
     'According': <gensim.models.keyedvectors.Vocab at 0x111f74a58>,
     'revenue': <gensim.models.keyedvectors.Vocab at 0x111f747f0>,
     'Our': <gensim.models.keyedvectors.Vocab at 0x111f74780>,
     'rose': <gensim.models.keyedvectors.Vocab at 0x111f74c18>,
     'previous': <gensim.models.keyedvectors.Vocab at 0x111f74b00>,
     'TV': <gensim.models.keyedvectors.Vocab at 0x111f74fd0>,
     'football': <gensim.models.keyedvectors.Vocab at 0x111f74320>,
     'biggest': <gensim.models.keyedvectors.Vocab at 0x111f74f60>,
     'employees': <gensim.models.keyedvectors.Vocab at 0x111f748d0>,
     'changes': <gensim.models.keyedvectors.Vocab at 0x111f74e10>,
     'residents': <gensim.models.keyedvectors.Vocab at 0x111f742b0>,
     'means': <gensim.models.keyedvectors.Vocab at 0x111f743c8>,
     'agreement': <gensim.models.keyedvectors.Vocab at 0x111f74898>,
     'includes': <gensim.models.keyedvectors.Vocab at 0x111f74f98>,
     'post': <gensim.models.keyedvectors.Vocab at 0x111f744e0>,
     'Canada': <gensim.models.keyedvectors.Vocab at 0x111f744a8>,
     'probably': <gensim.models.keyedvectors.Vocab at 0x111f74438>,
     'related': <gensim.models.keyedvectors.Vocab at 0x111f74978>,
     'training': <gensim.models.keyedvectors.Vocab at 0x111f6ab38>,
     'allowed': <gensim.models.keyedvectors.Vocab at 0x111f6a470>,
     'class': <gensim.models.keyedvectors.Vocab at 0x111f6a2b0>,
     'bit': <gensim.models.keyedvectors.Vocab at 0x111f6a7f0>,
     'video': <gensim.models.keyedvectors.Vocab at 0x111f6a860>,
     'Michael': <gensim.models.keyedvectors.Vocab at 0x111f6a898>,
     'An': <gensim.models.keyedvectors.Vocab at 0x111f6a3c8>,
     'sent': <gensim.models.keyedvectors.Vocab at 0x111f6a908>,
     'education': <gensim.models.keyedvectors.Vocab at 0x111f6a1d0>,
     'states': <gensim.models.keyedvectors.Vocab at 0x111f6a390>,
     'straight': <gensim.models.keyedvectors.Vocab at 0x111f6a748>,
     'love': <gensim.models.keyedvectors.Vocab at 0x111f6a2e8>,
     'beat': <gensim.models.keyedvectors.Vocab at 0x111f6a0b8>,
     'hold': <gensim.models.keyedvectors.Vocab at 0x111f6a978>,
     'turn': <gensim.models.keyedvectors.Vocab at 0x111f6aa20>,
     'finished': <gensim.models.keyedvectors.Vocab at 0x111f6a5f8>,
     'network': <gensim.models.keyedvectors.Vocab at 0x111f6a400>,
     'Smith': <gensim.models.keyedvectors.Vocab at 0x111f6aba8>,
     'buy': <gensim.models.keyedvectors.Vocab at 0x111f6a4e0>,
     'foreign': <gensim.models.keyedvectors.Vocab at 0x111f6a630>,
     'especially': <gensim.models.keyedvectors.Vocab at 0x111f6a828>,
     'groups': <gensim.models.keyedvectors.Vocab at 0x111f6a7b8>,
     'wants': <gensim.models.keyedvectors.Vocab at 0x111f6a080>,
     'title': <gensim.models.keyedvectors.Vocab at 0x111f6ba20>,
     'included': <gensim.models.keyedvectors.Vocab at 0x111f6bc50>,
     'turned': <gensim.models.keyedvectors.Vocab at 0x111f6bd30>,
     'bank': <gensim.models.keyedvectors.Vocab at 0x111f6b2b0>,
     'Florida': <gensim.models.keyedvectors.Vocab at 0x111f6bac8>,
     'efforts': <gensim.models.keyedvectors.Vocab at 0x111f6bc18>,
     'personal': <gensim.models.keyedvectors.Vocab at 0x111f6b6a0>,
     'businesses': <gensim.models.keyedvectors.Vocab at 0x111f6b710>,
     'August': <gensim.models.keyedvectors.Vocab at 0x111f6b588>,
     'California': <gensim.models.keyedvectors.Vocab at 0x111f6bba8>,
     'situation': <gensim.models.keyedvectors.Vocab at 0x111f6b048>,
     'district': <gensim.models.keyedvectors.Vocab at 0x111f6b780>,
     'allow': <gensim.models.keyedvectors.Vocab at 0x111f6b400>,
     'helped': <gensim.models.keyedvectors.Vocab at 0x111f6bf60>,
     'body': <gensim.models.keyedvectors.Vocab at 0x111f6b6d8>,
     'nothing': <gensim.models.keyedvectors.Vocab at 0x111f6bf98>,
     'soon': <gensim.models.keyedvectors.Vocab at 0x111f6b898>,
     'safety': <gensim.models.keyedvectors.Vocab at 0x111f6b080>,
     'officer': <gensim.models.keyedvectors.Vocab at 0x111f6ba90>,
     'cents': <gensim.models.keyedvectors.Vocab at 0x111f6b630>,
     'Europe': <gensim.models.keyedvectors.Vocab at 0x111f5f048>,
     'St.': <gensim.models.keyedvectors.Vocab at 0x111f5f898>,
     'additional': <gensim.models.keyedvectors.Vocab at 0x111f5f828>,
     'spokesman': <gensim.models.keyedvectors.Vocab at 0x111f5fac8>,
     'February': <gensim.models.keyedvectors.Vocab at 0x111f5fc18>,
     'wife': <gensim.models.keyedvectors.Vocab at 0x111f5fc88>,
     'showed': <gensim.models.keyedvectors.Vocab at 0x111f5fda0>,
     'leave': <gensim.models.keyedvectors.Vocab at 0x111f5ff28>,
     'investors': <gensim.models.keyedvectors.Vocab at 0x111f5f0b8>,
     'parents': <gensim.models.keyedvectors.Vocab at 0x111f5f978>,
     'medical': <gensim.models.keyedvectors.Vocab at 0x111f5f8d0>,
     'spending': <gensim.models.keyedvectors.Vocab at 0x111f1d438>,
     'non': <gensim.models.keyedvectors.Vocab at 0x111f1d6d8>,
     'London': <gensim.models.keyedvectors.Vocab at 0x111f1dc88>,
     'Council': <gensim.models.keyedvectors.Vocab at 0x111f1db38>,
     'matter': <gensim.models.keyedvectors.Vocab at 0x111f1d9e8>,
     'spent': <gensim.models.keyedvectors.Vocab at 0x111f1ddd8>,
     'child': <gensim.models.keyedvectors.Vocab at 0x111f1dac8>,
     'World': <gensim.models.keyedvectors.Vocab at 0x111f1d2e8>,
     'effort': <gensim.models.keyedvectors.Vocab at 0x111f1d668>,
     'opening': <gensim.models.keyedvectors.Vocab at 0x111f1d0f0>,
     'either': <gensim.models.keyedvectors.Vocab at 0x111f732e8>,
     'range': <gensim.models.keyedvectors.Vocab at 0x111f73128>,
     'question': <gensim.models.keyedvectors.Vocab at 0x111f738d0>,
     'European': <gensim.models.keyedvectors.Vocab at 0x111f73470>,
     'goals': <gensim.models.keyedvectors.Vocab at 0x111f73da0>,
     'administration': <gensim.models.keyedvectors.Vocab at 0x111f736d8>,
     'friends': <gensim.models.keyedvectors.Vocab at 0x111f730f0>,
     'himself': <gensim.models.keyedvectors.Vocab at 0x111f73080>,
     'shows': <gensim.models.keyedvectors.Vocab at 0x111f73828>,
     'difficult': <gensim.models.keyedvectors.Vocab at 0x111f73f98>,
     'kids': <gensim.models.keyedvectors.Vocab at 0x111f73c18>,
     'paid': <gensim.models.keyedvectors.Vocab at 0x111f73c50>,
     'create': <gensim.models.keyedvectors.Vocab at 0x111f73710>,
     'cash': <gensim.models.keyedvectors.Vocab at 0x111f73278>,
     'age': <gensim.models.keyedvectors.Vocab at 0x111f9be80>,
     'league': <gensim.models.keyedvectors.Vocab at 0x111f9b780>,
     'form': <gensim.models.keyedvectors.Vocab at 0x111f9bac8>,
     'impact': <gensim.models.keyedvectors.Vocab at 0x111f9b518>,
     'drive': <gensim.models.keyedvectors.Vocab at 0x111f9b908>,
     'someone': <gensim.models.keyedvectors.Vocab at 0x111f9bd30>,
     'became': <gensim.models.keyedvectors.Vocab at 0x111f9b8d0>,
     'stay': <gensim.models.keyedvectors.Vocab at 0x111f9bc88>,
     'fight': <gensim.models.keyedvectors.Vocab at 0x111f9b160>,
     'significant': <gensim.models.keyedvectors.Vocab at 0x111f9b2e8>,
     'firm': <gensim.models.keyedvectors.Vocab at 0x111f9bcc0>,
     'Senate': <gensim.models.keyedvectors.Vocab at 0x111f9bcf8>,
     'hospital': <gensim.models.keyedvectors.Vocab at 0x111f9b128>,
     'charged': <gensim.models.keyedvectors.Vocab at 0x111f9bf98>,
     'operating': <gensim.models.keyedvectors.Vocab at 0x111f9b748>,
     'main': <gensim.models.keyedvectors.Vocab at 0x111f9b978>,
     'book': <gensim.models.keyedvectors.Vocab at 0x111f15080>,
     'success': <gensim.models.keyedvectors.Vocab at 0x111f154a8>,
     'son': <gensim.models.keyedvectors.Vocab at 0x111f15748>,
     'trading': <gensim.models.keyedvectors.Vocab at 0x111f15828>,
     '###-####': <gensim.models.keyedvectors.Vocab at 0x111f15470>,
     'focus': <gensim.models.keyedvectors.Vocab at 0x111f15a20>,
     'room': <gensim.models.keyedvectors.Vocab at 0x111f15d30>,
     'continued': <gensim.models.keyedvectors.Vocab at 0x111f15eb8>,
     'Congress': <gensim.models.keyedvectors.Vocab at 0x111f15f28>,
     'everything': <gensim.models.keyedvectors.Vocab at 0x111f15dd8>,
     'Park': <gensim.models.keyedvectors.Vocab at 0x111f15588>,
     'agency': <gensim.models.keyedvectors.Vocab at 0x111f159b0>,
     'brought': <gensim.models.keyedvectors.Vocab at 0x111f15c50>,
     'talk': <gensim.models.keyedvectors.Vocab at 0x111f15b70>,
     'break': <gensim.models.keyedvectors.Vocab at 0x111f15940>,
     'air': <gensim.models.keyedvectors.Vocab at 0x111f15be0>,
     'software': <gensim.models.keyedvectors.Vocab at 0x111f157f0>,
     'decided': <gensim.models.keyedvectors.Vocab at 0x111f8fb38>,
     'Do': <gensim.models.keyedvectors.Vocab at 0x111f8fba8>,
     'ready': <gensim.models.keyedvectors.Vocab at 0x111f8ff98>,
     'arrested': <gensim.models.keyedvectors.Vocab at 0x111f8f9e8>,
     'track': <gensim.models.keyedvectors.Vocab at 0x111f8f828>,
     'provides': <gensim.models.keyedvectors.Vocab at 0x111f8fc50>,
     'mother': <gensim.models.keyedvectors.Vocab at 0x111f8fb70>,
     'base': <gensim.models.keyedvectors.Vocab at 0x111f8f2b0>,
     'trial': <gensim.models.keyedvectors.Vocab at 0x111f8f278>,
     'phone': <gensim.models.keyedvectors.Vocab at 0x111f8fd30>,
     'My': <gensim.models.keyedvectors.Vocab at 0x111f8fe10>,
     'build': <gensim.models.keyedvectors.Vocab at 0x111f8f0b8>,
     'conditions': <gensim.models.keyedvectors.Vocab at 0x111f8f668>,
     'rest': <gensim.models.keyedvectors.Vocab at 0x111f8fbe0>,
     'Johnson': <gensim.models.keyedvectors.Vocab at 0x111f8ff60>,
     'terms': <gensim.models.keyedvectors.Vocab at 0x111f8feb8>,
     'expect': <gensim.models.keyedvectors.Vocab at 0x111f8f978>,
     'England': <gensim.models.keyedvectors.Vocab at 0x111f8f898>,
     'Israel': <gensim.models.keyedvectors.Vocab at 0x111f8fd68>,
     'despite': <gensim.models.keyedvectors.Vocab at 0x111f20240>,
     'closed': <gensim.models.keyedvectors.Vocab at 0x111f20ac8>,
     'starting': <gensim.models.keyedvectors.Vocab at 0x111f20438>,
     'provided': <gensim.models.keyedvectors.Vocab at 0x111f20748>,
     'pressure': <gensim.models.keyedvectors.Vocab at 0x111f20320>,
     'lives': <gensim.models.keyedvectors.Vocab at 0x111f20d30>,
     'step': <gensim.models.keyedvectors.Vocab at 0x111f20a58>,
     'remain': <gensim.models.keyedvectors.Vocab at 0x111f20cf8>,
     'similar': <gensim.models.keyedvectors.Vocab at 0x111f20080>,
     'charge': <gensim.models.keyedvectors.Vocab at 0x111f20da0>,
     'date': <gensim.models.keyedvectors.Vocab at 0x111f20160>,
     'whole': <gensim.models.keyedvectors.Vocab at 0x111f20d68>,
     'land': <gensim.models.keyedvectors.Vocab at 0x111f20c88>,
     'growing': <gensim.models.keyedvectors.Vocab at 0x111f20048>,
     'James': <gensim.models.keyedvectors.Vocab at 0x111f20668>,
     'Internet': <gensim.models.keyedvectors.Vocab at 0x111f20358>,
     'projects': <gensim.models.keyedvectors.Vocab at 0x111f207f0>,
     'British': <gensim.models.keyedvectors.Vocab at 0x111f20278>,
     'cases': <gensim.models.keyedvectors.Vocab at 0x111f0d160>,
     'ground': <gensim.models.keyedvectors.Vocab at 0x111f0d4e0>,
     'legal': <gensim.models.keyedvectors.Vocab at 0x111f0d4a8>,
     'International': <gensim.models.keyedvectors.Vocab at 0x111f0da20>,
     'agreed': <gensim.models.keyedvectors.Vocab at 0x111f0dc50>,
     'tell': <gensim.models.keyedvectors.Vocab at 0x111f0df98>,
     'test': <gensim.models.keyedvectors.Vocab at 0x111f0d240>,
     'everyone': <gensim.models.keyedvectors.Vocab at 0x111f0dfd0>,
     'pretty': <gensim.models.keyedvectors.Vocab at 0x111f0d5f8>,
     'authorities': <gensim.models.keyedvectors.Vocab at 0x111f0d128>,
     'Two': <gensim.models.keyedvectors.Vocab at 0x111f0d080>,
     'above': <gensim.models.keyedvectors.Vocab at 0x111f0d748>,
     'moved': <gensim.models.keyedvectors.Vocab at 0x111f0d2b0>,
     'profit': <gensim.models.keyedvectors.Vocab at 0x111f0d3c8>,
     'throughout': <gensim.models.keyedvectors.Vocab at 0x111f0deb8>,
     'inside': <gensim.models.keyedvectors.Vocab at 0x111f0d9e8>,
     'ability': <gensim.models.keyedvectors.Vocab at 0x111f0d898>,
     'overall': <gensim.models.keyedvectors.Vocab at 0x111f0d8d0>,
     'pass': <gensim.models.keyedvectors.Vocab at 0x111f12080>,
     'officers': <gensim.models.keyedvectors.Vocab at 0x111f12e48>,
     'rather': <gensim.models.keyedvectors.Vocab at 0x111f12dd8>,
     'Australia': <gensim.models.keyedvectors.Vocab at 0x111f125c0>,
     'actually': <gensim.models.keyedvectors.Vocab at 0x111f12c88>,
     'county': <gensim.models.keyedvectors.Vocab at 0x111f12a20>,
     'amount': <gensim.models.keyedvectors.Vocab at 0x111f128d0>,
     'scheduled': <gensim.models.keyedvectors.Vocab at 0x111f12da0>,
     'themselves': <gensim.models.keyedvectors.Vocab at 0x111f123c8>,
     'organization': <gensim.models.keyedvectors.Vocab at 0x111f12e10>,
     'giving': <gensim.models.keyedvectors.Vocab at 0x111f129b0>,
     'credit': <gensim.models.keyedvectors.Vocab at 0x111f12c50>,
     'father': <gensim.models.keyedvectors.Vocab at 0x111f12a90>,
     'drug': <gensim.models.keyedvectors.Vocab at 0x111f12550>,
     'investigation': <gensim.models.keyedvectors.Vocab at 0x111f12438>,
     'families': <gensim.models.keyedvectors.Vocab at 0x111f12f98>,
     'Republican': <gensim.models.keyedvectors.Vocab at 0x111f18240>,
     'funds': <gensim.models.keyedvectors.Vocab at 0x111f18a90>,
     'patients': <gensim.models.keyedvectors.Vocab at 0x111f18a20>,
     'takes': <gensim.models.keyedvectors.Vocab at 0x111f18198>,
     'systems': <gensim.models.keyedvectors.Vocab at 0x111f188d0>,
     'Japan': <gensim.models.keyedvectors.Vocab at 0x111f18cf8>,
     'complete': <gensim.models.keyedvectors.Vocab at 0x111f18fd0>,
     'sold': <gensim.models.keyedvectors.Vocab at 0x111f18e10>,
     'practice': <gensim.models.keyedvectors.Vocab at 0x111f18390>,
     'calls': <gensim.models.keyedvectors.Vocab at 0x111f18908>,
     '•': <gensim.models.keyedvectors.Vocab at 0x111f18080>,
     'UK': <gensim.models.keyedvectors.Vocab at 0x111f18dd8>,
     'force': <gensim.models.keyedvectors.Vocab at 0x111f18eb8>,
     'student': <gensim.models.keyedvectors.Vocab at 0x111f18358>,
     'idea': <gensim.models.keyedvectors.Vocab at 0x10c0045f8>,
     'reached': <gensim.models.keyedvectors.Vocab at 0x10c045748>,
     'reason': <gensim.models.keyedvectors.Vocab at 0x10c045860>,
     'levels': <gensim.models.keyedvectors.Vocab at 0x10c045f98>,
     'space': <gensim.models.keyedvectors.Vocab at 0x10c045828>,
     'competition': <gensim.models.keyedvectors.Vocab at 0x10aa13080>,
     'forces': <gensim.models.keyedvectors.Vocab at 0x111ce1080>,
     'sector': <gensim.models.keyedvectors.Vocab at 0x111ce10f0>,
     'Last': <gensim.models.keyedvectors.Vocab at 0x111ce1668>,
     'tried': <gensim.models.keyedvectors.Vocab at 0x111ce16a0>,
     'common': <gensim.models.keyedvectors.Vocab at 0x10c4f09e8>,
     'homes': <gensim.models.keyedvectors.Vocab at 0x10c4f0940>,
     'stage': <gensim.models.keyedvectors.Vocab at 0x111cd9c50>,
     'department': <gensim.models.keyedvectors.Vocab at 0x111cd9ef0>,
     'named': <gensim.models.keyedvectors.Vocab at 0x111cd9e10>,
     'earnings': <gensim.models.keyedvectors.Vocab at 0x111cd9c88>,
     'offers': <gensim.models.keyedvectors.Vocab at 0x111cd9eb8>,
     'star': <gensim.models.keyedvectors.Vocab at 0x111e73b70>,
     'certain': <gensim.models.keyedvectors.Vocab at 0x111e73b38>,
     'double': <gensim.models.keyedvectors.Vocab at 0x111e73c50>,
     'longer': <gensim.models.keyedvectors.Vocab at 0x111e73c18>,
     'followed': <gensim.models.keyedvectors.Vocab at 0x111e4ca58>,
     'cause': <gensim.models.keyedvectors.Vocab at 0x111e4cb70>,
     'Association': <gensim.models.keyedvectors.Vocab at 0x111e4ca90>,
     'signed': <gensim.models.keyedvectors.Vocab at 0x111e4cc18>,
     'committee': <gensim.models.keyedvectors.Vocab at 0x111e4c780>,
     'hour': <gensim.models.keyedvectors.Vocab at 0x111e4c8d0>,
     'college': <gensim.models.keyedvectors.Vocab at 0x111e4c390>,
     'Pakistan': <gensim.models.keyedvectors.Vocab at 0x111e4cb38>,
     'users': <gensim.models.keyedvectors.Vocab at 0x111e4c470>,
     'Iran': <gensim.models.keyedvectors.Vocab at 0x111e4c278>,
     'sign': <gensim.models.keyedvectors.Vocab at 0x111e4c358>,
     'living': <gensim.models.keyedvectors.Vocab at 0x111e4c2b0>,
     'failed': <gensim.models.keyedvectors.Vocab at 0x111e4c0f0>,
     'reach': <gensim.models.keyedvectors.Vocab at 0x111e4c3c8>,
     'quickly': <gensim.models.keyedvectors.Vocab at 0x111e4c588>,
     'receive': <gensim.models.keyedvectors.Vocab at 0x111e4c828>,
     'debt': <gensim.models.keyedvectors.Vocab at 0x111e4cda0>,
     'sale': <gensim.models.keyedvectors.Vocab at 0x111e4c5f8>,
     'Board': <gensim.models.keyedvectors.Vocab at 0x111e4c6d8>,
     'Americans': <gensim.models.keyedvectors.Vocab at 0x111e4c860>,
     'Road': <gensim.models.keyedvectors.Vocab at 0x111e4ccc0>,
     'Brown': <gensim.models.keyedvectors.Vocab at 0x111e4c198>,
     'insurance': <gensim.models.keyedvectors.Vocab at 0x111e4cdd8>,
     '##:##': <gensim.models.keyedvectors.Vocab at 0x111e4c710>,
     'anyone': <gensim.models.keyedvectors.Vocab at 0x111e4cef0>,
     'tournament': <gensim.models.keyedvectors.Vocab at 0x111e4c748>,
     'More': <gensim.models.keyedvectors.Vocab at 0x111e4c668>,
     'gas': <gensim.models.keyedvectors.Vocab at 0x111e4c9e8>,
     'talks': <gensim.models.keyedvectors.Vocab at 0x111e4c9b0>,
     'serious': <gensim.models.keyedvectors.Vocab at 0x111e4ce48>,
     'required': <gensim.models.keyedvectors.Vocab at 0x111e4ce10>,
     'sell': <gensim.models.keyedvectors.Vocab at 0x111e4c908>,
     'construction': <gensim.models.keyedvectors.Vocab at 0x111e4ce80>,
     'evidence': <gensim.models.keyedvectors.Vocab at 0x111e4cd30>,
     'remains': <gensim.models.keyedvectors.Vocab at 0x111e4cf98>,
     'black': <gensim.models.keyedvectors.Vocab at 0x111e4cb00>,
     'below': <gensim.models.keyedvectors.Vocab at 0x111e4c160>,
     'improve': <gensim.models.keyedvectors.Vocab at 0x111e4c400>,
     'crisis': <gensim.models.keyedvectors.Vocab at 0x10bf72b70>,
     'address': <gensim.models.keyedvectors.Vocab at 0x111e087b8>,
     'questions': <gensim.models.keyedvectors.Vocab at 0x111e08710>,
     'easy': <gensim.models.keyedvectors.Vocab at 0x111cfd198>,
     'begin': <gensim.models.keyedvectors.Vocab at 0x10bfa2908>,
     'view': <gensim.models.keyedvectors.Vocab at 0x111e32400>,
     'School': <gensim.models.keyedvectors.Vocab at 0x111f0ecc0>,
     'heard': <gensim.models.keyedvectors.Vocab at 0x111f0ee80>,
     'executive': <gensim.models.keyedvectors.Vocab at 0x111f0e630>,
     'raised': <gensim.models.keyedvectors.Vocab at 0x111f0e2b0>,
     ...}




```python
print(model.most_similar('microsoft'))
```

    [('adobe_photoshop', 0.8042364716529846), ('microsoft_office', 0.7978680729866028), ('windows_xp', 0.7926486134529114), ('buy_microsoft', 0.7902629375457764), ('cs4', 0.7494896650314331), ('autocad', 0.7432770729064941), ('photoshop', 0.7404437065124512), ('windows_vista', 0.7382057309150696), ('quickbooks', 0.7320874929428101), ('adobe_photoshop_cs4', 0.7269179821014404)]



```python
"""
Transform learning mechanism could be applied to extend the knowledge base of custom word2vec model
potential use cases using Google's pretrained model are:
 - auto correct spellings
 - predict next word in the scentence
The use case above can be used in pre-processing stage to transform the user input 
closer to the context of trained model.
"""
```


```python
# Prediciting output word using gensim word2vec wrapper
training_questions = [
    "i want_to buy diamond jewellery",
    "handcrafted jewellery is good",
    "expensive diamonds are often handcrafted",
    "some exhibitors sell hand_made rings",
    "people are minions and they love diamond"
]
training_questions = [word.split() for word in training_questions]
# Initialising Word2Vec model
model = models.Word2Vec(size=50, window=10, min_count=1, workers=4, hs=1, sg=1)
# Building model vocabulary using words from event data
model.build_vocab(training_questions)
# Training Word2Vec model and presisting it in memory
model.train(training_questions, total_examples=model.corpus_count, epochs=model.iter)

# Report the probability distribution of the center word given the context words as input to the trained model.
model.predict_output_word('i want to sell'.split(), topn=1)
```

    /anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:15: DeprecationWarning: Call to deprecated `iter` (Attribute will be removed in 4.0.0, use self.epochs instead).
      from ipykernel import kernelapp as app





    [('jewellery', 0.04545477)]




```python
# Auto correction using Google's pretrained model
model = models.KeyedVectors.load_word2vec_format(google_vectors, binary=True)

words = model.index2word

w_rank = {}
for i,word in enumerate(words):
    w_rank[word] = i

WORDS = w_rank
```


```python
import re
from collections import Counter

def words(text): return re.findall(r'\w+', text.lower())

def P(word): 
    "Probability of `word`."
    # use inverse of rank as proxy
    # returns 0 if the word isn't in the dictionary
    return - WORDS.get(word, 0)

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

print(correction('dimond'))
print(correction('androd'))
```

    diamond
    android



```python
"""
Conclusion:
Word2Vec is great for word embeddings and it also comes handy before taking the words into embedding stage.
By using public vector sets like GoogleKeyedVectors, we can improve the preprocessing stage. The input
sentence can be parsed as according top the context of training set. For example, if the training set is all about
Jewellery expo and the user passed request something like -> 'i am looking to auto cad software' we can try to
parse this at preprocessing stage which would mean that the user might be looking for auto cad software to 
design Jewellery. On the top of that the pretrained vectors can be used to build micro models which can be exposed as
APIs for spelling check, auto corrections, suggesting next word, etc.
"""
```
