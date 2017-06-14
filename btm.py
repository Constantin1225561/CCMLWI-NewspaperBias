# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:21:30 2017

Adapted from:
https://github.com/jcapde/Biterm/blob/master/Biterm_sampler.py
"""

import numpy as np
from collections import Counter

class BitermTopicModel:
    def __init__(self, tokens, vocab_size=5000):
        """
        Create a Biterm Topic Model with given tokens.
            tokens :: list
                A list containing at each index a list of tokens that make
                up the sentence.
            vocab_size :: int, optional
                The maximum number of unique words in the vocabulary.
        """
        self.tokens = tokens
        self.flat = np.array([word for title in tokens for word in title if len(word) > 0])
        self.vocab_size = vocab_size
        
        self.Nz = None
        self.Nwz = None
        self.Z = None
        self.num_topics = 20
        self.aux = None
        self.pzb = None
        self.btmp = None
        self.thetaz = None
        self.phiwz = None
        self.pdz = None
        
        # Get the top vocab_size words
        vocab, counts = np.unique(self.flat, return_counts=True)
        sort = np.argsort(counts)[-vocab_size:]
        self.vocab = vocab[sort]
        
    def to_biterms(self, tokens):
        """Convert a list of tokens to a list of bi-term tuples."""
        biterms = []
        vocab = self.vocab
        for iword1, word1 in enumerate(tokens):
            for iword2, word2 in enumerate(tokens):
                if (iword1 < iword2):
                    vi1 = np.where(vocab==word1)[0]
                    vi2 = np.where(vocab==word2)[0]
                    if (len(vi1) > 0 and len(vi2) > 0):
                        biterms.append((vi1[0],vi2[0]))
        return biterms
    
    def fit(self, num_iterations=20, num_topics=10, alpha=1, beta=0.1):
        print("BTM: Creating bi-terms")
        vocab_size = self.vocab_size
        tokens = self.tokens
        self.num_topics = num_topics
        
        if self.aux == None:
            btmp = []
            for i, headline in enumerate(tokens):
                if (i % 1000 == 0):
                    print("Doc {}".format(i))
                if (len(headline) > 0):
                    btmp.append(self.to_biterms(headline))
            self.btmp = btmp
            aux = []
            for bi in btmp:
                aux.extend(bi)
            self.aux = aux
        else:
            aux = self.aux
            btmp = self.btmp
        
        b = list(set(aux))
        B = len(b)
                
        print("BTM: Computing bi-term probabilities")
        pbd_cts = [self.pbd(doc, False) for doc in btmp]
        
        print("BTM: Gibbs sampling")
        Nz, Nwz, Z = self.gibbs_sampler_LDA(It=num_iterations, V=vocab_size, B=B, num_topics=num_topics, b=b, alpha=alpha, beta=beta)
        
        self.Nz = Nz
        self.Nwz = Nwz
        self.Z = Z
             
        thetaz = (Nz + alpha)/(B + num_topics*alpha)
        phiwz = (Nwz + beta)/np.tile((Nwz.sum(axis=0)+vocab_size*beta),(vocab_size,1))
        self.thetaz = thetaz
        self.phiwz = phiwz
        
        # P(z|b)
        pzb = [[list(thetaz*phiwz[term[0],:]*phiwz[term[1],:]/(thetaz*phiwz[term[0],:]*phiwz[term[1],:]).sum()) for term in set(doc)] for doc in btmp]
        self.pzb = pzb
    
        # P(d | z)
        pdz = []
        for idoc, doc in enumerate(pzb):
            aux = 0
            for iterm, term in enumerate(doc):
                aux += np.array(term) * pbd_cts[idoc][iterm]
            pdz.append(aux)
    
        pdz = np.array(pdz)
        self.pdz = pdz
        
    def get_topic_words(self, words_per_topic=10):
        Nwz = self.Nwz
        topics =  [[self.vocab[ident] for ident in np.argsort(-Nwz[:,k])[0:words_per_topic]] for k in range(self.num_topics)]
        return topics
    
    def get_top_docs_for_topics(self, headlines_per_topic=5):
        pdz = self.pdz
        return [[split[ident] for ident in np.argsort(-pdz[:,k])[0:headlines_per_topic]] for k in range(self.num_topics)]
    
    def get_topics_for_docs(self, docs):
        thetaz = self.thetaz
        phiwz = self.phiwz
        docs = [self.to_biterms(doc) for doc in docs]
        pzb = [[list(thetaz*phiwz[term[0],:]*phiwz[term[1],:]/(thetaz*phiwz[term[0],:]*phiwz[term[1],:]).sum()) for term in set(doc)] for doc in docs]
        
        return [((np.sum(topic_prods_b,axis=0))/len(topic_prods_b)).tolist() for topic_prods_b in pzb]
    
    def save_params(self, path):
        np.savez_compressed(path,(self.Nwz,self.Nz,self.Z,self.pdz,self.thetaz,self.phiwz))
        
    def load_params(self, path):
        with np.load(path) as params:
            self.Nwz,self.Nz,self.Z,self.pdz,self.thetaz,self.phiwz = params['arr_0']
        
    def gibbs_sampler_LDA(self, It, V, B, num_topics, b, alpha=1., beta=0.1):
        print("Biterm model ------ ")
        print("Corpus length: " + str(len(b)))
        print("Number of topics: " + str(num_topics))
        print("alpha: " + str(alpha) + " beta: " + str(beta))
    
        Z =  np.zeros(B,dtype=int)
        Nwz = np.zeros((V, num_topics))
        Nz = np.zeros(num_topics)
    
        theta = np.random.dirichlet([alpha]*num_topics, 1)
        for ibi, bi in enumerate(b):
            topics = np.random.choice(num_topics, 1, p=theta[0,:])[0]
            Nwz[bi[0], topics] += 1
            Nwz[bi[1], topics] += 1
            Nz[topics] += 1
            Z[ibi] = topics
    
        for it in range(It):
            print("Iteration: " + str(it))
            Nzold = np.copy(Nz)
            for ibi, bi in enumerate(b):
                Nwz[bi[0], Z[ibi]] -= 1
                Nwz[bi[1], Z[ibi]] -= 1
                Nz[Z[ibi]] -= 1
                pz = (Nz + alpha)*(Nwz[bi[0],:]+beta)*(Nwz[bi[1],:]+beta)/(Nwz.sum(axis=0)+beta*V)**2
                pz = pz/pz.sum()
                Z[ibi] = np.random.choice(num_topics, 1, p=pz)
                Nwz[bi[0], Z[ibi]] += 1
                Nwz[bi[1], Z[ibi]] += 1
                Nz[Z[ibi]] += 1
            print("Variation between iterations:  " + str(np.sqrt(np.sum((Nz-Nzold)**2))))
        return Nz, Nwz, Z
    
    def pbd(self, doc,names):
        # Compute probability of drawing a biterm from a doc
        # P(b | d)
        ret = []
        retnames = []
        for term1 in set(doc):
            cnts = 0.
            for term2 in doc:
                if term1 == term2:
                    cnts +=1.
            ret.append(cnts/len(doc))
            retnames.append(term1)
        if names:
            return retnames
        else:
            return ret

if __name__=="__main__":
    with open('headlines2017.txt') as f:
        doc=f.read()
        
    vocab_size = 5000
    alpha = 1
    beta = 0.1
    
    headlines = doc.split('\n')
    unique_headlines = [item for item, count in Counter(headlines).items() if count == 1]
    split = [headline.split(" ") for headline in unique_headlines]
        
    # Expected running time: 3h
    
    btm = BitermTopicModel(split, vocab_size)
    #btm.fit(num_iterations=250, num_topics=20,alpha=2.5,beta=0.01)
    btm.load_params('btm_params_2017.npz')
    for i, topic in enumerate(btm.get_topic_words(words_per_topic=20)):
        print("Topic {}: ".format(i), topic)
    topic_probs = btm.get_topics_for_docs(split[0:10])
    for i, probs in enumerate(topic_probs):
        print("\""+unique_headlines[i]+"\": topic {}".format(np.argmax(probs)))
    #btm.save_params('btm_params_2017')