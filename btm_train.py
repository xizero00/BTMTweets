import numpy as np
import pickle
import pyLDAvis
from biterm.btm import oBTM 
from sklearn.feature_extraction.text import CountVectorizer
from biterm.utility import vec_to_biterms, topic_summuary # helper functions

import time
import pandas as pd

def __num_dist_rows__(array, ndigits=2):
    return array.shape[0] - int((pd.DataFrame(array).sum(axis=1) < 0.999).sum())


if __name__ == "__main__":
    timestamp = "{}".format(str(time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())))
    iteration_num = 1
    postfix = '_' + timestamp + '_' + str(iteration_num) + 'iter'
    
    texts = open('./data/after_preprocess_dataset_clean_english_only_new.txt').read().splitlines() # path of data file

    # vectorize texts
    vec = CountVectorizer(stop_words='english')
    X = vec.fit_transform(texts).toarray()

    # get vocabulary
    vocab = np.array(vec.get_feature_names())

    # get biterms
    biterms = vec_to_biterms(X)
    
    # # create btm
    btm = oBTM(num_topics=20, V=vocab)
    
    print("\n\n Train Online BTM ..")
    for i in range(0, len(biterms), 100): # prozess chunk of 200 texts
        print(f"bitems: {i}/{len(biterms)}")
        biterms_chunk = biterms[i:i + 100]
        btm.fit(biterms_chunk, iterations=iteration_num) 
    topics = btm.transform(biterms)
    print(topics.shape)
    print(__num_dist_rows__(topics))

    save_btm_model_path = './models/btm_model{}.pkl'.format(postfix)
    save_btm_topics_path = './models/btm_topics{}.pkl'.format(postfix)
    with open(save_btm_model_path, 'wb') as fd:
        pickle.dump(btm, fd)
    with open(save_btm_topics_path, 'wb') as fd:
        pickle.dump(topics, fd)

    print("\n\n Visualize Topics ..")

    topics = topics / topics.sum(axis=1)[:, None]
    print(__num_dist_rows__(topics))
    
    save_html_path = './vis/online_btm{}.html'.format(postfix)
    vis = pyLDAvis.prepare(btm.phi_wz.T, topics, np.count_nonzero(X, axis=1), vocab, np.sum(X, axis=0), mds='mmds')
    pyLDAvis.save_html(vis, save_html_path)  # path to output

    print("\n\n Topic coherence ..")
    save_topic_coherence_result_path = "./output/topic_coherence_result{}.txt".format(postfix)
    topic_summuary(btm.phi_wz.T, X, vocab, 10, save_topic_coherence_result_path)

    save_topic_result_path = "./output/topic_result{}.txt".format(postfix)
    result_str = ""
    print("\n\n Texts & Topics ..")
    for i in range(len(texts)):
        result_str += "{} (topic: {})\n".format(texts[i], topics[i].argmax())
        print("{} (topic: {})".format(texts[i], topics[i].argmax()))
    
    wf = open(save_topic_result_path, 'w')
    wf.write(result_str)