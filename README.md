# <BitermTopicModel-ChatGPTAnalysis>

## Description

The appearance of ChatGPT has caused a global craze, and people are discussing the impact of ChatGPT on human life and production. Our project adopts the BitermTopicModel to carry out topic analysis on the content about ChatGPT on Twitter, and analyzes the topic of each tweet and the specific topic content. Finally find out what people are talking about ChatGPT.

The reason why BitermTopicModel is used is that Twitter data is usually some short text with weak context, and this model can better complete the task of short text analysis.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

**Step 0.** Install BTM
```
pip install biterm
```
or install the biterm use the modifed version from my own repo.
 ```
pip install git+https://github.com/xizero00/biterm.git
 ```

**Step 1.** Modify the code in the package (if you use my own repo, you do not need to perform this action)

You should replace the `btm.py` and `utility.py` in the BTM package with the two files we provided in `replace_file`

Then you have completed the installation steps

## Folder Structure
```
BitermTopicModel-ChatGPT/
│
├── data/ - dataset in here
│   ├── after_preprocess_dataset_clean_english_only_new.csv - data after preprocessing
│   ├── after_preprocess_dataset_clean_english_only_new.txt - Another file format
│   ├── dataset_clean_english_only_new.csv - original data
│   ├── dataset_clean_english_only_new.txt - Another file format
│   └── dataset_clean_english_only_new.xlsx - Another file format
│
├── excel_result/ - a table of analysis results
│   ├── BTM.xlsx
│   └── tweets_by_topic.csv
│
├── models/ - default directory for storing input data
│   ├── btm_model_2023-04-22-13-40-46_1iter.pkl - Model weight file for 1 iteration (gibbs sampling 1 iteration)
│   ├── btm_model_2023-04-22-13-58-46_5iter.pkl - Model weight file for 5 iteration (gibbs sampling 5 iteration)
│   ├── btm_topics_2023-04-22-13-40-46_1iter.pkl - Topic probability distribution file for 1 iteration
│   └── btm_topics_2023-04-22-13-58-46_5iter.pkl - Topic probability distribution file for 5 iteration
│
├── output/ - btm output
│   ├── topic_coherence_result_2023-04-22-13-40-46_1iter.txt - topic-word probability distribution for 1 iteration 
│   ├── topic_coherence_result_2023-04-22-13-58-46_5iter.txt - topic-word probability distribution for 5 iteration 
│   ├── topic_result_2023-04-22-13-40-46_1iter.txt - The results of the topic assigned by each document for 1 iteration
│   ├── topic_result_2023-04-22-13-58-46_5iter.txt - The results of the topic assigned by each document for 5 iteration
│   └── tweets_by_topic.csv - topic-keywords-tweet table
│
├── replace_file/
│   ├── btm.py - replace the same file in btm package
│   └── utility.py - replace the same file in btm package
│
├── vis/ - visual results in html format 
│   ├── online_btm_2023-04-22-13-40-46_1iter.html - html result for 1 iteration
│   └── online_btm_2023-04-22-13-58-46_5iter.html - html result for 5 iteration
│
├── dataprocessing.py - data preprocessing code
├── btm_train.py - trian code
│
├── btm_analysis_result.ipynb - visualization of btm generation results
├── sentiment_analysis.ipynb - an emotional analysis of Twitter texts
└── table_generate_result.ipynb - analysis of btm generation results are stored in excel_result/BTM.xlsx
```

The visualization of html
![Intertopic Distance Map](./vis/Intertopic%20Distance%20Map.png)


## Usage

If you want to train the BTM model against your data set and analyze and visualize the results of the BTM model, the following changes are required

You should modify the code `btm = oBTM(num_topics=20, V=vocab)` in `btm_train.py`, the total number of topics can be determined by yourself. and in `btm.fit(biterms_chunk, iterations=iteration_num)`, you can specify the number of iterations of model training. Each iteration updates the entire corpus topic.

After the results are obtained, you can use `btm_analysis_result.ipynb` to generate following visual results.
![Documents and Topics](./vis/Documents%20and%20Topics.png)
![Topic Word Scores](./vis/Topic%20Word%20Scores.png)

You can use `sentiment_analysis.ipynb` to acquire
 the results of sentiment analysis and `table_generate_result.ipynb` to generate the table content in `BTM.xlsx` and `tweets_by_topic.csv`.

