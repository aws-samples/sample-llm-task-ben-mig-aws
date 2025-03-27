# Prepare your dataset for model evaluation/benchmarking and model migration 

* For the task specific model evaluation and benchmarking:   
  * upload the datafile to the `<Your_S3_Bucket>/<task_folder>/input-data` folder
* For the model migration optimizer:   
  * testing dataset: upload the data file to the `<Your_S3_Bucket>/<task_folder>/input-data` folder
  * training dataset: put the data file in the local /data folder  


## Data format 

### Summarization task

The input dataset for Summarization use case: a csv file containing the following data fields

* Section_id: a unique ID
* Section_title (optional): the title of the text for summarization
* Section_text: the text for summarization

### Question-Answer with RAG task 

The input dataset for Question-Answer (RAG) use case: a csv file containing the following data fields

* instruction: the question 
* input (optional): the context for generating the answer 
* response: the answer (ground truth)

### Classification task

The input dataset for Classification use case: a csv file containing the following data fields

* contact_id: a unique ID
* transcript: the text for classification
* ref_category: the class (ground truth)
* all_category: all the classes for this task

### Entity Extraction task 

The input dataset for Entity Extraction use case: a csv file containing the following data fields

* name: the name of the entity used for prompt
* reference: the entity (ground truth)

### Sentiment Analysis task

The input dataset for Sentiment use case: a csv file containing the following data fields

* contact_id: a unique ID
* transcript: the text to generate sentiment score
* ref_sentiment: the sentiment score (ground truth)











