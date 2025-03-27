# Developer's Guide 

## Task specific quality metrics 

We provide task specific evaluation metrics allows users to evaluate LLMs based on the specific criteria that are important for their application. This ensures that the model is not only performing well in general but also delivering the desired results for the intended use cases in accuracy and reliability. 

The quality metrics we provide are for the following LLM tasks:  

### Basic NLP evaluation metrics 

**BLEU score** (Bilingual Evaluation Understudy) : a metric used to evaluate the quality of machine-translated text by comparing it to a set of human-translated reference texts, measuring the similarity between the two using n-grams and a brevity penalty.  

**ROUGE score** (Recall-Oriented Understanding for Gisting Evaluation): a set of metrics used to evaluate the quality of automatically generated text, particularly in tasks like text summarization and machine translation, by comparing it to human-generated reference texts. 

**METEOR score** (Metric for Evaluation of Translation with Explicit Ordering): a metric used in natural language processing to evaluate the quality of machine translation by comparing it to human translations, considering precision, recall, and word order. 

**Perplexity score**: measures how well a model predicts a sequence of words, with lower scores indicating better predictive ability and a more confident model. It is calculated by exponentiating the negative average log-likelihood of the predicted word probabilities. 

### Text classification, sentiment analysis, text entity extraction: 

**Accuracy**: a metric that assesses the correctness of predictions made by an NLP system. It measures the proportion of correctly predicted instances across all classes. Accurate predictions are crucial in ensuring the reliability and usefulness of NLP systems.

**BERT-precision score**: quantifies how much of the candidate's content is semantically meaningful relative to the reference. It is calculated as the average of the maximum cosine similarities between each candidate token's embedding and the embeddings of all reference tokens.

**BERT-recall score**: measures how much of the reference text's meaning is captured by the candidate text, using cosine similarity between contextual embeddings of the tokens. 

**BERT-F1 score**: a metric that evaluates text generation tasks by measuring the semantic similarity between a candidate text and a reference text using BERT's contextual embeddings, ultimately providing a single score that balances precision and recall. 

### Question-answer (RAG) 

**Context-precision**: a metric that measures the proportion of relevant chunks in the retrieved_contexts. It is calculated as the mean of the precision@k for each chunk in the context. Precision@k is the ratio of the number of relevant chunks at rank k to the total number of chunks at rank k.

**Context-recall**: measures how many of the relevant documents (or pieces of information) were successfully retrieved. It focuses on not missing important results. Higher recall means fewer relevant documents were left out. In short, recall is about not missing anything important. Since it is about not missing anything, calculating context recall always requires a reference to compare against.

**Answer-semantic similarity**: assesses the semantic resemblance between the generated answer and the ground truth. This evaluation is based on the ground truth and the answer, with values falling within the range of 0 to 1. A higher score signifies a better alignment between the generated answer and the ground truth.

**Answer-factual-correctness**: compares and evaluates the factual accuracy of the generated response with the reference. It is used to determine the extent to which the generated response aligns with the reference. The factual correctness score ranges from 0 to 1, with higher values indicating better performance. 

**Answer-relevancy**: measures how relevant a response is to the user input. Higher scores indicate better alignment with the user input, while lower scores are given if the response is incomplete or includes redundant information.

**Answer-faithfulness**: measures how factually consistent a response is with the retrieved context. It ranges from 0 to 1, with higher scores indicating better consistency.

### Text summarization 

**BERT-F1 score**: measures how similar the text summary is to the original text. It performs similarity calculations using contextualized token embeddings shown to be effective for entailment detection.

**LLM-judge**: LLMs are used to evaluate other LLM outputs, providing a flexible and scalable alternative to human evaluation. It involves using an LLM to assess the quality of generated text based on criteria defined in a prompt. For a summarization task, it measures
   - Consistency: characterizes the summary’s factual and logical correctness. It should stay true to the original text, not introduce additional information, and use the same terminology.
   - Relevance: captures whether the summary is limited to the most pertinent information in the original text. A relevant summary focuses on the essential facts and key messages, omitting unnecessary details or trivial information.
   - Fluency: describes the readability of the summary. A fluent summary is well-written and uses proper syntax, vocabulary, and grammar.
   - Coherence: measures the logical flow and connectivity of ideas. A coherent summary presents the information in a structured, logical, and easily understandable manner.

## Evaluation functions 

We provide a number of evaluation functions in the package, for you to easily generate the above mentioned quality metrics when you build evaluation for your own LLM task. You can use these functions in your own code to generate the corresponding metrics needed. 

- **Accuracy score**: `peccyben.calculate_accuracy(pred_list,ref_list)`    
- **ROUGE score**: `peccyben.alculate_rouge(pred_list,ref_list)`  
- **Semantic similarity score** using embedding model in SentenceTransformer: `peccyben.calculate_semantic_sim(pred_list,ref_list)`     
- **Semantic similarity score** using the Titan-text-embedding model on Amazon Bedrock: `peccyben.calculate_semantic_sim_titan(pred_list,ref_list)`    
- **BERT-precision, recall and F1 score**: `peccyben.calculate_bertscore(pred_list,ref_list)`    
- **LLM-as-a-judge for summarization task**: `peccyben.llm_judge_summ(jm1,jm2,doc_text,summ_text)`     

## Inference LLM cross platform 

If you need to infer foundation models or fine-tuned models on AWS or other open-source frameworks, we provide you the following functions that you can use to infer the model you select. 

- **Models on Amazon Bedrock** by Conv API: `peccyben.LLM_Text_Bedrock_Conv_Infer(region,model_id,model_kwargs,prompt,cacheconf="default",latencyOpt="optimized")`
- **Models on Amazon Bedrock** by Inference API: `peccyben.LLM_Text_Bedrock_Infer(region,model_id,model_kwargs,prompt)`
- **Models on Huggingface**: `peccyben.LLM_Text_HF_Infer(model_id,model_kwargs,prompt)`
- **Models on Ollama**: `peccyben.LLM_Text_OLM_Infer(model_id,model_kwargs,prompt)` 

## Task based LLM evaluation functions

You can call the following functions to evaluate your LLM for specific tasks. Refer to the example notebooks [here](README.md#example-notebooks)

- **Text sumarization**: `peccyben.Summ_Ben(method,region,model_id,jm1,jm2,model_kwargs,prompt_template,s3_bucket,file_name,BENCH_KEY,task_folder,cost_key,save_id,SLEEP_SEC,SAMPLE_LEN=-1,PP_TIME=1,cacheconf="default",latencyOpt="optimized")`
- **Text classification**: `peccyben.Class_Ben(method,region,model_id,model_kwargs,prompt_template,s3_bucket,file_name,BENCH_KEY,task_folder,cost_key,save_id,SLEEP_SEC,SAMPLE_LEN=-1,PP_TIME=1,cacheconf="default",latencyOpt="optimized")`
- **Text entity extraction**: `peccyben.Extract_Ben(method,region,model_id,model_kwargs,prompt_template,document_text,s3_bucket,file_name,BENCH_KEY,task_folder,cost_key,save_id,SLEEP_SEC,SAMPLE_LEN=-1,PP_TIME=1,cacheconf="default",latencyOpt="optimized")`
- **Question-answer by RAG**: `peccyben.QA_Ben(method,region,model_id,judge_model_id,model_kwargs,prompt_template,vdb_name,s3_bucket,file_name,BENCH_KEY,task_folder,cost_key,save_id,SLEEP_SEC,SAMPLE_LEN=-1,PP_TIME=1,cacheconf="default",latencyOpt="optimized")`
- **Sentiment analysis**: `peccyben.Sent_Ben(method,region,model_id,model_kwargs,prompt_template,s3_bucket,file_name,BENCH_KEY,task_folder,cost_key,save_id,SLEEP_SEC,SAMPLE_LEN=-1,PP_TIME=1,cacheconf="default",latencyOpt="optimized")`
- **Open-ended question-answer**: `peccyben.PromptEval_Ben(method,region,model_id,model_kwargs,prompt_template,s3_bucket,file_name,BENCH_KEY,task_folder,cost_key,save_id,SLEEP_SEC,SAMPLE_LEN=-1,PP_TIME=1,cacheconf="default",latencyOpt="optimized")`

## Cost-performance analysis functions

We provide cost-performance analysis based on the benchmarking results from the task based evaluation. Call the following functions to generate cost-performance analysis graph

- **Graph for all models**: `peccyben.plot_ben_costperf(df_results,perf_metric,title,task_folder)`
- **Graph for top x models**: `peccyben.plot_topx_cp(df,perf_metric,top_num,task_folder)`

## Optimizing prompt for model migration 

You can call the following functions to optimize your prompt to be adaptive to the model you want to migrate to. 

- **Prompt translation by [Amazon Bedrock Prompt Optimization](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-management-optimize.html)**: `peccymig.Prompt_Opt_Template_Gen(APO_REGION_NAME, model_id, source_prompt_template)`
- **Data-aware prompt optimization**: `peccymig.data_aware_optimization(model,train_set,eval_metric,num_candidates=5,num_trials=7,minibatch_size=20,minibatch_full_eval_steps=7)`

## Bring your own inference results for evaluation

If you have a complex use case that you need to post-process your inference results for evaluation, you can do it in three steps. 
- Step 1: use `peccyben.PromptEval_Ben` to run inference on the models you select, to get the model responses and the non-quality metrics such as latency, throughput, and costs in a dataframe. 
- Step 2: post-process the model responses with your ground-truth data, and define your custom quality metrics, in a dataframe.
- Step 3: merge the non-quality dataframe (from step 1) and the quality dataframe (from step 2), and generate the final results by `peccyben.Ben_Save` function 






