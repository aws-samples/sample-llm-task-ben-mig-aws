#**********************************
#
# Evaluation Functions
#
#**********************************

import json
import boto3


from sentence_transformers import SentenceTransformer, util
from peccyben.utils import LLM_Text_Bedrock_Infer, LLM_Text_Bedrock_Conv_Infer, Init_Bedrock
from peccyben.utils import extract_strings_recursive


#--------- Accuracy Score ---------
def calculate_accuracy(pred_list,ref_list):
    
    accuracy_score_list = []
    accuracy_score_list = sum(1 for x,y in zip(pred_list,ref_list) if y in x) / float(len(ref_list))
    
    average_accuracy_score = np.mean(accuracy_score_list)
    
    return average_accuracy_score
    
    
#--------- ROUGE Score ---------
from rouge_score import rouge_scorer

import evaluate
rouge = evaluate.load('rouge')

import numpy as np

def calculate_rouge(pred_list,ref_list):
    
    rouge_results = rouge.compute(predictions=pred_list,
                         references=ref_list,
                         use_aggregator=True)
    avg_rougeLsum = np.mean(rouge_results["rougeLsum"])
    avg_rougeL = np.mean(rouge_results["rougeL"])
    avg_rouge2 = np.mean(rouge_results["rouge2"])
    avg_rouge1 = np.mean(rouge_results["rouge1"])
    
    #print("Rouge score = ", avg_rougeLsum, avg_rougeL, avg_rouge2, avg_rouge1)
    
    return avg_rougeLsum, avg_rougeL, avg_rouge2, avg_rouge1
    

#--------- Semantic Similarity Score ---------
# SS using 3P embedding model
def calculate_semantic_sim(pred_list,ref_list):
    
    #model = SentenceTransformer('all-MiniLM-L6-v2')
    model = SentenceTransformer('thenlper/gte-large')    # use this one in production
    #model = SentenceTransformer('llmrails/ember-v1')
    
    sem_score = []
    average_sem_sim = 0
    
    for i in range(len(ref_list)):
        #print(i," ",end = ':')
        ref_embedding = model.encode(ref_list[i])
        pred_embedding = model.encode(pred_list[i])
        cos_sim = util.cos_sim(ref_embedding, pred_embedding)
        #print(cos_sim[0][0].item())
        
        sem_score.append(cos_sim[0][0].item())
    
    average_sem_sim = np.mean(sem_score)   
    
    #print("Average similarity: ", average_sem_sim)
    
    return average_sem_sim

# SS using Titan embedding model
def get_titan_embedding(text):
    
    body = json.dumps({"inputText": text,"dimensions": 512, "normalize": True})
    modelId = 'amazon.titan-embed-text-v2:0'    # support 8K token 
    accept = '*/*'
    contentType = 'application/json'    

    boto3_bedrock_runtime = Init_Bedrock('us-east-1')
    response = boto3_bedrock_runtime.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get('embedding')
    
    return embedding
    
def calculate_semantic_sim_titan(pred_list,ref_list):
   
    sem_score = []
    average_sem_sim = 0
    
    for i in range(len(ref_list)):
        #print(i," ",end = ':')
        ref_embedding = get_titan_embedding(ref_list[i])
        pred_embedding = get_titan_embedding(pred_list[i])
        cos_sim = util.cos_sim(ref_embedding, pred_embedding)
        #print(cos_sim[0][0].item())
        
        sem_score.append(cos_sim[0][0].item())
    
    average_sem_sim_titan = np.mean(sem_score)   
    
    #print("Average similarity: ", average_sem_sim)
    
    return average_sem_sim_titan
    

#--------- Toxicity ---------
# method 1: 
from detoxify import Detoxify

def calculate_toxicity_1(text_list):

    tox1 = Detoxify('original').predict(text_list)
    
    average_tox1 = np.mean(tox1['toxicity'])
    
    return average_tox1
                                                
"""    
    tox_score = []
    average_tox_sim = 0

    for i in range(len(text_list)):
        tox_score.append(Detoxify('original').predict([text_list[i]))
        
    average_tox = np.mean(tox_score) 
                   
    return average_tox, tox_score
"""

# method 2:
import evaluate

def calculate_toxicity_2(text=[], aggregation_method=None):
    """
    Evaluate toxicity of a list of text strings using a pre-trained model.

    Args:
        text (list): List of text strings to evaluate for toxicity.
        aggregation_method (str): Method for aggregating toxicity scores.

    Returns:
        float: Toxicity score based on the specified aggregation method.
    """
    # specify model name
    toxicity_model_name = "facebook/roberta-hate-speech-dynabench-r4-target"

    toxicity_evaluator = evaluate.load(
        "toxicity",
        toxicity_model_name,
        module_type="measurement",
        toxic_label="hate",
    )
    toxicity_score = toxicity_evaluator.compute(
        predictions=text, aggregation=aggregation_method
    )

    if aggregation_method == None:
        toxicity_measure = "toxicity"
    elif aggregation_method == "maximum":
        toxicity_measure = "max_toxicity"
    elif aggregation_method == "ratio":
        toxicity_measure = "toxicity_ratio"
    else:
        toxicity_measure = "toxicity"

    return toxicity_score[toxicity_measure]
    

#--------- Bert Score ---------
from bert_score import score

def calculate_bertscore(pred_list,ref_list):
    
    P, R, F1 = score(pred_list,ref_list, lang='en', verbose=False)
    
    P_score = []
    average_P_score = 0
    R_score = []
    average_R_score = 0
    F1_score = []
    average_F1_score = 0
    
    for i in range(len(pred_list)):
        P_score.append(P[i].item())
        R_score.append(R[i].item())
        F1_score.append(F1[i].item())
        
    average_P_score = np.mean(P_score)
    average_R_score = np.mean(R_score)
    average_F1_score = np.mean(F1_score)
    
    return average_P_score, average_R_score, average_F1_score



# ------- LLM judge -------

from langchain.prompts import PromptTemplate

def llm_judge_summ(jm1,jm2,doc_text,summ_text):

    region = 'us-west-2'
    
    prompt_template_aif = """
    You are an AI assistant, your task is to compare the following LLM-generated summary with the original document, rate how well it captures the key points and conveys the most critical information, on a scale of 1-5.
    
    The score should be based on the following performance criteria:
    - Consistency: characterizes the summary’s factual and logical correctness. It should stay true to the original text, not introduce additional information, and use the same terminology.
    - Relevance: captures whether the summary is limited to the most pertinent information in the original text. A relevant summary focuses on the essential facts and key messages, omitting unnecessary details or trivial information.
    - Fluency: describes the readability of the summary. A fluent summary is well-written and uses proper syntax, vocabulary, and grammar.
    - Coherence: measures the logical flow and connectivity of ideas. A coherent summary presents the information in a structured, logical, and easily understandable manner.
    
    Score 5 means the LLM-generated summary is the best summary fully aligned with the original document, 
    Score 1 means the LLM-generated summary is the worst summary completely irrelevant to the original document.  

    Please also provide an explanation on why you provide the score. Keep the explanation as concise as possible.

    The LLM-generated summary is provided within the <summary> XML tag,
    The original document is provided within the <document> XML tag, 

    In your response, present the score within the <score> XML tag, and the explanation within the <thinking> XML tag.

    DO NOT nest <score> and <thinking> element. 
    DO NOT put any extra attribute in the <score> and <thinking> tag. 
    
    <document>
    {document}
    </document>

    LLM generated summary:
    <summary>
    {summary}
    </summary>
    """

    PROMPT_aif = PromptTemplate(template=prompt_template_aif, input_variables=["document", "summary"])

    # Judge 1: Claude
    judge_model_1_id = jm1
    prompt_1 = """Human:\n"""+PROMPT_aif.format(document = doc_text, summary = summ_text)+"""\nAssistant:""" 
    model_1_kwargs = {
        'maxTokens': 1024, 
        'temperature': 0
    }   
    aif_response_1 = LLM_Text_Bedrock_Conv_Infer(region,judge_model_1_id,model_1_kwargs,prompt_1,cacheconf="None",latencyOpt="None")
    lj_score_1 = float(extract_strings_recursive(aif_response_1[0], "score")[0])
    lj_thinking_1 = extract_strings_recursive(aif_response_1[0], "thinking")[0]
    
    # Judge 2: deepseek   
    judge_model_2_id = jm2
    #prompt_2 = """<s><INST> \n"""+PROMPT_aif.format(document = doc_text, summary = summ_text)+"""\n</INST>"""    # mistral
    prompt_2 = PROMPT_aif.format(document = doc_text, summary = summ_text) 
    model_2_kwargs = {
        'maxTokens': 1024, 
        'temperature': 0
    } 
    aif_response_2 = LLM_Text_Bedrock_Conv_Infer(region,judge_model_2_id,model_2_kwargs,prompt_2,cacheconf="None",latencyOpt="None")
    lj_score_2 = float(extract_strings_recursive(aif_response_2[0], "score")[0])
    lj_thinking_2 = extract_strings_recursive(aif_response_2[0], "thinking")[0]    
    
    lj_score = (lj_score_1+lj_score_2)/2
    lj_thinking = "Judge_1:("+str(lj_score_1)+")"+lj_thinking_1+"\nJudge_2:("+str(lj_score_2)+")"+lj_thinking_2

    return lj_score, lj_thinking
