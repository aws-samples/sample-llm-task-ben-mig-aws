#**********************************
#
# Prompt Functions
#
#**********************************


#--------- Prompt catalog ---------
import json

def Get_Prompt_Catalog(task):

    # read CATALOG
    with open('prompt_catalog.json') as json_file:
        cat_data = json.load(json_file)
    
    p_persona = ""
    p_instruction = ""
    p_inputs = ""
    p_example = ""
    
    # get prompt components
    p_persona = cat_data[task]['persona']
    p_instruction = cat_data[task]['instruction']
    p_inputs = cat_data[task]['inputs']
    p_example = cat_data[task]['example']
        
    return p_persona, p_instruction, p_inputs, p_example
    
    
#--------- Generate prompt tempate ---------
def Prompt_Template_Gen(model_id, task, instructions=""):
    
    # 4 parts 
    # (1) Persona 
    # (2) Instruction points 
    # (3) Inputs 
    # (4) Add special format 
    
    p_persona, p_instruction, p_inputs, p_example = Get_Prompt_Catalog(task)
 
    p_data = ""

    if ('amazon.titan-tg1-large' in model_id):
        p_data = p_data + p_persona + '\n'
        p_data = p_data + p_instruction + '\n'
        if (len(instructions)>0):
            p_data = p_data + instrutions + '\n' 
        p_data = p_data + p_inputs + '\n'
        
        prompt_template = p_data
    
    elif ('amazon.titan-text-agile-v1' in model_id):
        p_data = p_data + p_persona + '\n'
        
        lines = p_instruction.split('\n')
        p_instruction_new = ''.join(['- '+line+'\n' for line in lines])
        p_instruction_new = p_instruction_new.replace('Do not','DO NOT')
        p_data = p_data + "Instructions: \n" + p_instruction_new + '\n'
        
        if (len(instructions)>0):
            p_data = p_data + instrutions + '\n' 
        p_data = p_data + p_inputs + '\n'
        
        prompt_template = """\n""" + p_data + """\n"""
    
    elif ('llama' in model_id):
        p_data = p_data + p_persona + '\n'
        p_data = p_data + p_instruction + '\n'
        if (len(instructions)>0):
            p_data = p_data + instrutions + '\n' 
        p_data = """<|start_header_id|>system<|end_header_id|>\n"""+p_data+"""\n<|eot_id|>\n"""
        p_data = p_data + """<|start_header_id|>user<|end_header_id|>\n"""+p_inputs+"""\n<|eot_id|>\n""" 
        p_data = p_data + """<|start_header_id|>assistant<|end_header_id|>\n"""
        prompt_template = """<|begin_of_text|>\n""" + p_data  
    
    elif ('claude-v2' in model_id or 'claude-instant-v1' in model_id ):
        p_data = p_data + p_persona + '\n'
        p_data = p_data + p_instruction + '\n'
        if (len(instructions)>0):
            p_data = p_data + instrutions + '\n' 
        p_data = p_data + p_inputs + '\n'
        
        prompt_template = """Human: \n""" + p_data + """\nAssistant:"""
    
    elif ('claude-3' in model_id):
        p_data = p_data + p_persona + '\n'
        p_data = p_data + p_instruction + '\n'
        if (len(instructions)>0):
            p_data = p_data + instrutions + '\n' 
        p_data = p_data + p_inputs + '\n'
        
        prompt_template = """Human: \n""" + p_data + """\nAssistant:"""
    
    elif ('mistral' in model_id):
        p_data = p_data + p_persona + '\n'
        p_data = p_data + p_instruction + '\n'
        if (len(instructions)>0):
            p_data = p_data + instrutions + '\n' 
        p_data = p_data + p_inputs + '\n'
        
        prompt_template = """<s><INST> \n""" + p_data + """\n</INST>"""  
    
    elif ('mixtral' in model_id):
        p_data = p_data + p_persona + '\n'
        p_data = p_data + p_instruction + '\n'
        if (len(instructions)>0):
            p_data = p_data + instrutions + '\n' 
        p_data = p_data + p_inputs + '\n'
        
        prompt_template = """<s><INST> \n""" + p_data + """\n</INST>""" 

    elif ('Phi' in model_id):
        p_data = p_data + p_persona + '\n'
        p_data = p_data + p_instruction + '\n'
        if (len(instructions)>0):
            p_data = p_data + instrutions + '\n' 
        p_data = """<|system|> \n"""+p_data+"""\n<|end|>"""
        p_data = p_data + """<|user|>\n"""+p_inputs+"""\n"""
        
        prompt_template = p_data + """\n<|assistant|>"""
        
    else :  # other models 
        p_data = p_data + p_persona + '\n'
        p_data = p_data + p_instruction + '\n'
        if (len(instructions)>0):
            p_data = p_data + instrutions + '\n' 
        p_data = p_data + p_inputs
        
        prompt_template = p_data
            
    return prompt_template