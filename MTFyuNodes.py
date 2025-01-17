import folder_paths
import json
import requests
import torch
import re
import os

class addToText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filepath": ("STRING", {"default": "输入路径"}),
                "text_to_add": ("STRING", {"default": "需要加入的内容"}),
                
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ('输出文本',)
    FUNCTION = "append_to_file"
    CATEGORY = "TMFyu/Text"

    def append_to_file(self,filepath, text_to_add):
    # """
    # 将文本添加到文件，如果文本与文件中已有的某行完全相同，则不添加。

    # Args:
    #     filepath: 文件路径
    #     text_to_add: 要添加的文本
    # """
         # 检查路径是否存在
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"路径不存在: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                existing_lines = f.readlines()
        except FileNotFoundError:
            existing_lines = []

        # 去除每行末尾的换行符，方便比较
        existing_lines = [line.strip() for line in existing_lines]

        if text_to_add not in existing_lines:
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(text_to_add + '\n')
        return(text_to_add,)
        

        
            

class switchT2T:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "number":("FLOAT", {
                        "default": 0, 
                        "min": 0, #Minimum value
                        "max": 1, #Maximum value
                        "step": 0.1, #Slider's step
                        "display": "slider" # Cosmetic only: display as "number" or "slider"
                    }),
                
                "is_t2t": ("BOOLEAN", {"label_on": "enabled", "label_off": "disabled"}),
                
            },
        }

    RETURN_TYPES = ("FLOAT","BOOLEAN",)
    RETURN_NAMES = ('重绘值',"是否启用图生图")
    FUNCTION = "panDuan"
    CATEGORY = "TMFyu"

    def panDuan(self,is_t2t,number):
        if (is_t2t in [False, None, "True"]):
            a = 1
        else:
            a = number
        return(a,is_t2t,)
        
            




#---------------------------------------------------------------------------------------------------------------------------------------------------#
# 从文件中读取历史记录
def load_prompt_history(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# 调用DeepSeek API
def call_deepseek_api(api_key, user_input, prompt_history):
    # 添加用户输入到历史记录
    prompt_history['messages'].append({"role": "user", "content": user_input})
    
    # 设置API请求的URL和headers
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 设置请求体
    data = {
        "model": "deepseek-chat",
        "messages": prompt_history['messages']
    }
    
    # 发送请求
    response = requests.post(url, headers=headers, json=data)
    
    # 检查响应状态
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

# 提取 JSON 部分
def extract_json_from_markdown(markdown_text):
    # 使用正则表达式提取 ```json 和 ``` 之间的内容
    match = re.search(r'```json\n(.*?)\n```', markdown_text, re.DOTALL)
    if match:
        return match.group(1).strip()  # 返回 JSON 部分
    else:
        raise ValueError("No JSON content found in the markdown text.")
#---------------------------------------------------------------------------------------------------------------------------------------------------#
class LLMProcessingNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": ""}),
                "_01tou_face": ("BOOLEAN", {"label_on": "enabled", "label_off": "disabled"}),
                "_02ziShi_biaoQing": ("BOOLEAN", {"label_on": "enabled", "label_off": "disabled"}),
                "_03up_body": ("BOOLEAN", {"label_on": "enabled", "label_off": "disabled"}),
                "_04dn_body": ("BOOLEAN", {"label_on": "enabled", "label_off": "disabled"}),
                "_05other": ("BOOLEAN", {"label_on": "enabled", "label_off": "disabled"}),
                "_06NSFW": ("BOOLEAN", {"label_on": "enabled", "label_off": "disabled"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("IS_NSFW", "_01头部及脸部", "_02角色姿势和表情", "_03角色上半身服饰特征", "_04角色下半身服饰特征", "_05其他", "_06NSFW")
    FUNCTION = "process"
    CATEGORY = "TMFyu"
    
    def process(self, text, api_key,_01tou_face,_02ziShi_biaoQing,_03up_body,_04dn_body,_05other,_06NSFW,):
        # 从文件中加载历史记录
        prompt_history = load_prompt_history("custom_nodes\comfyui-TMFyu-nodes\prompt.json")
        
        # 调用DeepSeek API
        try:
            response = call_deepseek_api(api_key, text, prompt_history)
            llm_output = response['choices'][0]['message']['content']
            
            # 提取 JSON 部分
            json_content = extract_json_from_markdown(llm_output)
            
            # 打印提取的 JSON 内容，用于调试
            print("Extracted JSON Content:", json_content)
            
            # 解析 JSON
            llm_output_json = json.loads(json_content)
            
            # 打印解析后的 JSON 内容，用于调试
            print("Parsed JSON Content:", llm_output_json)
            
            # 提取七个内容
            
            is_nsfw = llm_output_json.get("IS_NSFW", "")

            if (_01tou_face in [False, None, "False"]):
                head_features = " _"
            else:
                
                head_features = llm_output_json.get("\u89d2\u8272\u5934\u90e8\u4ee5\u4e0a\u670d\u9970\u7279\u5f81", "")

            if (_02ziShi_biaoQing in [False, None, "False"]):
                action_expression = " _"
            else:
                action_expression = llm_output_json.get("\u89d2\u8272\u52a8\u4f5c\u53ca\u8868\u60c5", "")

            if (_03up_body in [False, None, "False"]):
                upper_body_features = " _"
            else:
                upper_body_features = llm_output_json.get("\u89d2\u8272\u4e0a\u534a\u8eab\u670d\u9970\u7279\u5f81", "")

            if (_04dn_body in [False, None, "False"]):
                lower_body_features = " _"
            else:
                lower_body_features = llm_output_json.get("\u89d2\u8272\u4e0b\u534a\u8eab\u670d\u9970\u7279\u5f81", "")
            
            if (_05other in [False, None, "False"]):
                other = " _"
            else:

                other = llm_output_json.get("\u5176\u4ed6", "")

            if (_06NSFW in [False, None, "False"]):
                nsfw = " _"
            else:  
                nsfw = llm_output_json.get("NSFW", "")
            
            return (is_nsfw, head_features, action_expression, upper_body_features, lower_body_features, other, nsfw)
        

        except Exception as e:
            raise Exception(f"Error processing LLM output: {e}")



#---------------------------------------------------------------------------------------------------------------------------------------------------#

class Text_Concatenate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "delimiter": (["none", "space", "comma"],),
            },
            "optional": {
                "text1": ("STRING", {"forceInput": True}),
                "text2": ("STRING", {"forceInput": True}),      
                "text3": ("STRING", {"forceInput": True}),      
                "text4": ("STRING", {"forceInput": True}),      
                "text5": ("STRING", {"forceInput": True}), 
                "text6": ("STRING", {"forceInput": True}),       
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "get_contxt"
    CATEGORY = "TMFyu/Text"

    def get_contxt(self, delimiter, text1=None, text2=None, text3=None, text4=None, text5=None, text6=None):
        needdelim = False
        delim = ""
        if delimiter == "space":
            delim = " "
        if delimiter == "comma":
            delim = ", "

        concatenated = ""

        if text1:
            concatenated = text1
            needdelim = True
        
        if text2:
            if needdelim:
                concatenated += delim
            concatenated += text2
            needdelim = True
        
        if text3:
            if needdelim:
                concatenated += delim
            concatenated += text3
            needdelim = True

        if text4:
            if needdelim:
                concatenated += delim
            concatenated += text4
            needdelim = True

        if text5:
            if needdelim:
                concatenated += delim
            concatenated += text5
            needdelim = True

        if text6:
            if needdelim:
                concatenated += delim
            concatenated += text6
            needdelim = True

        return (concatenated,)

#---------------------------------------------------------------------------------------------------------------------------------------------------#

def addWeight(text, weight=1):
    if weight == 1:
        return text
    else:
        return f"({text}:{round(weight,3)})"

class PromptSlide:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                
                "prompt_keyword": ("STRING", 
                         {
                            "multiline": False, 
                            "default": '',
                            "dynamicPrompts": False
                          }),

                "weight":("FLOAT", {"default": 1, "min": -3,"max": 3,
                                                                "step": 0.01,
                                                                "display": "slider"}),

                # "min_value":("FLOAT", {
                #         "default": -2, 
                #         "min": -10, 
                #         "max": 0xffffffffffffffff,
                #         "step": 0.01, 
                #         "display": "number"  
                #     }),
                # "max_value":("FLOAT", {
                #         "default": 2, 
                #         "min": -10, 
                #         "max": 0xffffffffffffffff,
                #         "step": 0.01, 
                #         "display": "number"  
                #     }),
              
                }
            }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)

    FUNCTION = "run"

    CATEGORY = "TMFyu/Text"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,)
    OUTPUT_NODE = False

    # 运行的函数
    def run(self,prompt_keyword,weight):
        # if weight < min_value:
        #     weight= min_value
        # elif weight > max_value:
        #     weight= max_value
        p=addWeight(prompt_keyword,weight)
        return (p,)
    

#---------------------------------------------------------------------------------------------------------------------------------------------------#
class replace_string:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_string": ("STRING", {}),
                "old_string": ("STRING", {"default": ""}),
                "new_string": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)

    FUNCTION = "substr"

    # OUTPUT_NODE = False

    CATEGORY = "TMFyu/Text"

    def substr(self, old_string, new_string, input_string=""):
        out = input_string.replace(old_string, new_string)
        return (out,)

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"
# 要导出的所有节点及其名称的字典
# 注意：名称应全局唯一

NODE_CLASS_MAPPINGS = {
    
    "switchT2T": switchT2T,
    "LLMProcessingNode":LLMProcessingNode,
    "Text_Concatenate":Text_Concatenate,
    "PromptSlide":PromptSlide,
    "replace_string":replace_string,
    "addToText":addToText
}
# 一个包含节点友好/可读的标题的字典

NODE_DISPLAY_NAME_MAPPINGS = {
    "switchT2T": "重绘值切换",#节点别名
    "LLMProcessingNode":"LLM提示词分类",
    "Text_Concatenate":"字符串合并",
    "PromptSlide":"提示词权重",
    "replace_string":"字符串替换",
    "addToText":"添加字符到文本",


}
