#*******************************************************
# Prompt optimization by Bedrock APO and DSPy Optimizer
#*******************************************************

from enum import Enum
import time
import json, copy
import boto3
import pandas as pd
from sentence_transformers import SentenceTransformer, util

from dspy.evaluate import Evaluate, answer_exact_match, answer_passage_match
from bert_score import score
from sklearn.metrics import accuracy_score

from rouge_score import rouge_scorer
import evaluate
rouge = evaluate.load('rouge')

import dspy
from dspy.teleprompt import MIPROv2
from dspy.evaluate import Evaluate


# -------------- Bedrock optimizer ------------
# Bedrock APO
def get_input(prompt):
    """
    Return a prompt in json
    """
    return {
        "textPrompt": {
            "text": prompt
        } 
    }


def handle_response_stream(response):
    """
    
    Args:
    Return: a prompt in json
    """
    try:
        event_stream = response['optimizedPrompt']
        
        for event in event_stream:
            if 'optimizedPromptEvent' in event:
                #print("OPTIMIZED PROMPT ...")
                optimized_prompt = event['optimizedPromptEvent']
                #print(optimized_prompt)
            else:
                #print("ANALYZE PROMPT ...")
                analyze_prompt = event['analyzePromptEvent']
                #print(analyze_prompt)
    except Exception as e:
        raise e
        
    return optimized_prompt['optimizedPrompt']['textPrompt']['text']


def Prompt_Opt_Template_Gen(APO_REGION_NAME, model_id, source_prompt_template):
    """
    Invoke Bedrock APO's API to generate optimized prompt
    Args: 
        model_id: the model for the prompt to be optimized
        source_prompt_template: input prompt template before optimization
    Return: the optimized template
    """

    apo_client = boto3.client('bedrock-agent-runtime', region_name=APO_REGION_NAME)
    
    # prompt optimization 
    response = apo_client.optimize_prompt(
        input=get_input(source_prompt_template),
        targetModelId=model_id,
        #taskType=task_type
    )    
    optimized_prompt = handle_response_stream(response)    
    p_data_opt = json.loads(optimized_prompt) + '\n'  

    opt_prompt_template =  p_data_opt + '\n'
    #print("optimized... = \n", opt_prompt_template, '\n')       

    return opt_prompt_template


# --------------- DSPy eval --------------
# Evaluation functions 
def eval_metric_rouge(example, pred, trace=None):
    """
    Calculate rouge score
    Args: 
        example: groud-truth
        pred: prediction by LLM
    Return: rougeL score
    """    
    # ROUGE score
    rouge_results = rouge.compute(predictions=[pred.answer],references=[example.answer],use_aggregator=True)
    metric_value = rouge_results["rougeL"]
    
    return metric_value


def eval_metric_bert(example, pred, trace=None):
    """
    Calculate NLP-BERT score
    Args: 
        example: groud-truth
        pred: prediction by LLM
    Return: BERT score
    """    
    
    P, R, F1 = score([pred.answer],[example.answer], lang='en', verbose=False)
    metric_value = F1.item()
    
    return metric_value


def eval_metric_accuracy(example, pred, trace=None):
    """
    Calculate accuracy score
    Args: 
        example: groud-truth
        pred: prediction by LLM
    Return: accuracy score
    """    

    metric_value = accuracy_score([str(example.answer)],[str(pred.answer)])
    
    #if (pred.answer in example.answer):
    #    metric_value = 1.0
    #else:
    #    metric_value = 0.0
    
    return metric_value


def get_titan_embedding(text):
    """
    generate embedding vector from a text using Amazon Titan-embedding-V2 model
    Args: 
        text: input text to be embedded
    Return: embedding vector
    """   

    boto3_bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
    
    body = json.dumps({"inputText": text})
    modelId = 'amazon.titan-embed-text-v2:0'     
    accept = 'application/json'
    contentType = 'application/json'    
    
    response = boto3_bedrock_runtime.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get('embedding')
    
    return embedding


def eval_metric_ss(example, pred, trace=None):
    """
    Calculate semantic similarity score
    Args: 
        example: groud-truth
        pred: prediction by LLM
    Return: semantic similarity score
    """    
        
    ref_embedding = get_titan_embedding(example.answer)
    pred_embedding = get_titan_embedding(pred.answer)
    cos_sim = util.cos_sim(ref_embedding, pred_embedding)
        
    metric_value = cos_sim[0][0].item()
        
    return metric_value



class LLMJudge_Summ(dspy.Signature):
    """You are an AI assistant, your task is to compare the following predicted answer semantically with the groundtruth answer, \ 
    rate how well it captures the key points and conveys the most critical information from the groundtruth answer, \
    Provide a score between 0 and 1. \
    Score 1 means the predicted answer is fully aligned with the groundtruth answer,  \
    Score 0 means the predicted answer is completely irrelevant to the groundtruth answer. \
    The score should be based on the following performance criteria: \
    - Consistency: characterizes the predicted answer’s factual and logical correctness. It should stay true to the original text, not introduce additional information, and use the same terminology. \
    - Relevance: captures whether the predicted answer is limited to the most pertinent information in the groundthuth answer. A relevant predicted answer focuses on the essential facts and key messages, omitting unnecessary details or trivial information.  \
    - Fluency: describes the readability of the predicted answer. A fluent predicted answer is well-written and uses proper syntax, vocabulary, and grammar.   \
    - Coherence: measures the logical flow and connectivity of ideas. A coherent predicted answer presents the information in a structured, logical, and easily understandable manner.   \    
    In the response, only present the score in a decimal format. DO NOT add any text and any premables."""

    groundtruth_answer = dspy.InputField(desc="groundtruth answer for the question")
    predicted_answer = dspy.InputField(desc="predicted answer for the question")
    lj_score = dspy.OutputField(desc="Is the predicted answer factually correct to the groundtruth answer and same as groundtruth answer ?")


    
def lj_metric_summ(example, pred, trace=None):
    """
    generate a evaluation score by an LLM
    Args: 
        example: groud-truth
        pred: prediction by LLM
    Return: the LLM-judge score
    """    
    
    JUDGE_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    
    bedrock_judge = dspy.LM(JUDGE_ID)
    
    with dspy.settings.context(lm=bedrock_judge, temperature=0.01):      
        judge_summ = dspy.ChainOfThought(LLMJudge_Summ) 
        
        if (len(pred.answer)==0):
             llm_judge_score
        else:
            judge_metric = judge_summ(groundtruth_answer=example.answer, predicted_answer=pred.answer) 
            #print(judge_metric)
    
            score = judge_metric.lj_score
            print(score, end=' | ')
    
            llm_judge_score = float(score)
            
        print("llm_judge_ans = ",llm_judge_score)

    return llm_judge_score


# --------------- DSPy functions -----------------
def dspy_evaluation(devset, metric):
    """
    Apply a dataset on a evaluation metric by DSPy Evaluate
    Args: 
        devset: input dataset
        metric: metric function pre-defined
    Return: a dspy evaluate object
    """    
    evaluate = Evaluate(devset=devset, 
                    metric=metric,
                    num_threads=4, 
                    display_progress=True, 
                    display_table=True,
                    return_outputs=True)
    
    return evaluate


def dspy_program(model_id):
    """
    Initiate a DSPy program with a given model
    Args: 
        model_id: the model on bedrock
        metric: metric function pre-defined
    Return: a dspy evaluate object
    """    
    
    class dspy_CoT(dspy.Module):    
        def __init__(self,model_id):
            super().__init__()
            bedrock_model = dspy.LM(model_id)
            dspy.settings.configure(lm=bedrock_model, temperature=0.1, max_tokens=20480)
            self.prog = dspy.ChainOfThought("question -> answer")

        def forward(self, question):
            return self.prog(question=question)
    
    program = dspy_CoT(model_id)    
    
    return program


def retry(exceptions, tries=3, delay=1, backoff=1):
    def retry_decorator(func):
        def wrapper(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    print(f"Retrying in {mdelay} seconds... ({mtries-1} tries remaining)")
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return func(*args, **kwargs)
        return wrapper
    return retry_decorator


#@retry(exceptions=Exception, tries=10, delay=60)
def dspy_optimize_2(program,trainset,metric,num_candidates=5,num_trials=7,minibatch_size=20,minibatch_full_eval_steps=7):
    """
    MIPROv2 optimizer for DSPy program
    Args: 
        program: DSPy program
        trainset: dataset for MIPROv2 optimizer
        metric: evaluation metric for MIPROv2 optimizer
        num_candidates=5: Number of candidate instructions & few-shot examples to generate and evaluate for each predictor. 
        num_trials=7: Number of optimization trials to run. 
        minibatch_size=20: Size of minibatches for evaluations.
        minibatch_full_eval_steps=7: a full evaluation on the validation set will be carried out every minibatch_full_eval_steps on the top averaging set of prompts.
    Return: Optimized DSPy program
    """        
    # Initialize optimizer
    teleprompter = MIPROv2(
        metric=metric,
        num_candidates=num_candidates,
        #auto="light", # Can choose between light, medium, and heavy optimization runs
        verbose=False,
    )

    # Optimize program
    print(f"Optimizing program with MIPRO...")
    optimized_program = teleprompter.compile(
        program.deepcopy(),
        trainset=trainset,
        num_trials=num_trials,
        minibatch_size=minibatch_size,
        minibatch_full_eval_steps=minibatch_full_eval_steps,
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
        requires_permission_to_run=False,
    )

    return optimized_program


#@retry(exceptions=Exception, tries=10, delay=60)
def dspy_infer(program,devset,metric):
    """
    Infer LLM via DSPy program to generate response and evaluation score 
    Args: 
        program: DSPy program
        devset: dataset for model inference and evaluation
        metric: evaluation metric 
    Return: evluation score and model response
    """        
    
    evaluate = Evaluate(devset=devset[:], 
                    metric=metric,
                    num_threads=4, 
                    display_progress=False, 
                    display_table=False,
                    return_outputs=True)
    
    score, output = evaluate(program, devset=devset[:])

    return score, output




# --------------- DSPy optimizer -----------------
def model_initialize(model_id):
    """
    Initialize the model  
    Args: 
        model_id: the model_id for the model to be initialized
    Return: model (DSPy program)
    """        

    model = dspy_program(model_id)

    return model


def model_inference(model, test_set, eval_metric): 
    """
    Create DSPy dataset from DataFrame for question-answer task 
    Args: 
        model: the initialized model instance
        test_set: input dataset for inference
        eval_metric: evaluation metric
    Return: DSPy dataset
    """        

    test_score, test_output = dspy_infer(model,test_set,eval_metric)
    
    return test_score, test_output



def data_aware_optimization(model,train_set,eval_metric,num_candidates=5,num_trials=7,minibatch_size=20,minibatch_full_eval_steps=7):
    """
    Create DSPy dataset from DataFrame for question-answer task 
    Args: 
        model: the initialized model instance
        train_set: dataset for MIPROv2 optimizer training
        eval_metric: evaluation metric for model inference
        num_candidates=5: Number of candidate instructions & few-shot examples to generate and evaluate for each predictor. 
        num_trials=7: Number of optimization trials to run. 
        minibatch_size=20: Size of minibatches for evaluations.
        minibatch_full_eval_steps=7: a full evaluation on the validation set will be carried out every minibatch_full_eval_steps on the top averaging set of prompts.
    Return: DSPy dataset
    """        

    opt_model = dspy_optimize_2(model,train_set,eval_metric,
                                  num_candidates,num_trials,minibatch_size,minibatch_full_eval_steps)

    return opt_model


def update_prompt_catalog(prompt_catalog_id,optimized_model):
    """
    Add optimized prompt in the prompt catalog 
    Args: 
        prompt_catalog_id: original prompt id before optimization
        optimized_model: the optimized model program containing the optmized prompt

    """   
    
    opt_prompt = optimized_model.prog.predict.signature.instructions
    #opt_prompt = opt_prompt.replace("Given the fields `question`, produce the fields `answer`.","")

    with open('prompt_catalog.json') as json_file:
        cat_data = json.load(json_file)
    
    dao_prompt = copy.copy(cat_data[prompt_catalog_id])
    
    dao_prompt['persona'] = ''
    dao_prompt['instruction'] = opt_prompt
    dao_prompt
    
    cat_data[prompt_catalog_id+'-dao'] = (dao_prompt)
    cat_data
    
    with open("prompt_catalog.json", "w") as file:
        json.dump(cat_data, file)

    return


