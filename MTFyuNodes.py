import folder_paths
import json
import requests
import torch
import re
import os
from PIL import Image
import numpy as np
import random
import csv
import comfy.sd
import comfy.utils
import nodes
from PIL import Image, ImageOps
import datetime
from PIL import Image, PngImagePlugin


class RandomCSVReader:
    """
    CSV随机行读取器
    功能：
    1. 读取指定CSV文件内容
    2. 根据随机种子生成可重复的随机选择
    3. 自动跳过标题行
    4. 支持空文件处理
    """
    def __init__(self):
        self.cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "text_history.csv"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("random_text", "used_seed")
    FUNCTION = "read_random"
    CATEGORY = "TMFyu/Text"

    def read_random(self, file_path, seed):
        # 初始化随机生成器
        rng = random.Random(seed)
        
        try:
            # 检查文件是否存在
            if not os.path.isfile(file_path):
                return ("[Error] File not found", seed)
            
            # 读取CSV内容
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = [row[0] for row in reader if row]  # 跳过空行
                
            # 过滤标题行（假设第一行是标题）
            data_rows = rows[1:] if len(rows) > 1 else []
            
            if not data_rows:
                return ("[Error] No data rows", seed)
            
            # 生成随机索引
            random_index = rng.randint(0, len(data_rows)-1)
            selected_text = data_rows[random_index]
            
            return (selected_text, seed)
            
        except Exception as e:
            return (f"[Error] {str(e)}", seed)





class SaveUniqueTextToCSV:
    """
    将唯一文本保存到CSV文件
    功能：
    1. 首次输入时创建CSV文件并添加标题
    2. 后续输入会检查文件最后一行内容
    3. 仅在不同内容时追加新行
    """
    def __init__(self):
        self.last_saved_text = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "file_name": ("STRING", {"default": "text_history.csv"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("current_text",)
    FUNCTION = "save_text"
    CATEGORY = "TMFyu/Text"

    def save_text(self, text, file_name):
        # 检查文件是否存在
        file_exists = os.path.isfile(file_name)
        
        # 读取最后保存的内容
        last_content = self.read_last_line(file_name) if file_exists else None

        # 判断是否需要保存
        if text.strip() and text != last_content:
            self.write_to_csv(file_name, text, file_exists)
            self.last_saved_text = text

        return (text,)

    def read_last_line(self, file_path):
        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                last_line = None
                for last_line in reader:
                    pass
                return last_line[0] if last_line else None
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return None

    def write_to_csv(self, file_path, text, file_exists):
        try:
            with open(file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["text"])
                writer.writerow([text.strip()])
        except Exception as e:
            print(f"Error writing to CSV file: {e}")






class RandomSizeNode:
    @classmethod 
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "presets_path": ("STRING", {"default": "size_presets.txt"}),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_random_size"
    CATEGORY = "TMFyu/Text"

    def get_random_size(self, seed, presets_path):
        random.seed(seed)
        
        # 如果路径是相对路径，则转换为绝对路径
        if not os.path.isabs(presets_path):
            presets_path = os.path.join(folder_paths.get_folder_paths("custom_nodes")[0], presets_path)
        
        with open(presets_path, "r") as f:
            sizes = f.readlines()
        
        size = random.choice(sizes).strip()
        width, height = map(int, size.split('x'))
        
        return (width, height)





class TimeStampedImageSaver:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "base_path": ("STRING", {"default": "ComfyUI_Output", "multiline": False}),
                "save_workflow": ("BOOLEAN", {"default": True, "label_on": "ENABLED", "label_off": "DISABLED"}),
                "filename_prefix": ("STRING", {"default": "Image"}),
            },
            "hidden": {
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "TMFyu/image"

    def save_images(self, images, base_path, save_workflow=True, filename_prefix="Image", prompt=None, extra_pnginfo=None):
        # 创建时间戳文件夹
        now = datetime.datetime.now()
        date_folder = now.strftime("%Y-%m-%d")
        full_path = os.path.join(base_path, date_folder)
        os.makedirs(full_path, exist_ok=True)

        # 获取保存路径
        counter = 1
        results = []
        
        for image in images:
            # 生成带时间戳的文件名
            timestamp = now.strftime("%H-%M-%S")
            file_name = f"{filename_prefix}_{timestamp}_{counter:05}.png"
            file_path = os.path.join(full_path, file_name)

            # 转换图像数据
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # 添加工作流元数据
            metadata = PngImagePlugin.PngInfo()
            if save_workflow:
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            # 保存图片
            img.save(file_path, 
                    pnginfo=metadata if save_workflow else None,
                    compress_level=self.compress_level)

            results.append({
                "filename": file_name,
                "subfolder": date_folder,
                "type": "output"
            })
            counter += 1

        return {"ui": {"images": results}}





class LoadImagesFromDirList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "load_always": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "FILE_NANE")
    OUTPUT_IS_LIST = (True, True, True)

    FUNCTION = "load_images"

    CATEGORY = "TMFyu/image"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if 'load_always' in kwargs and kwargs['load_always']:
            return float("NaN")
        else:
            return hash(frozenset(kwargs))

    def load_images(self, directory: str, image_load_cap: int = 0, start_index: int = 0, load_always=False):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory}' cannot be found.")
        dir_files = os.listdir(directory)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        # Filter files by extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(directory, x) for x in dir_files]  # 保持完整路径用于加载

        # start at start_index
        dir_files = dir_files[start_index:]

        images = []
        masks = []
        file_paths = []

        limit_images = False
        if image_load_cap > 0:
            limit_images = True
        image_count = 0

        for image_path in dir_files:
            if os.path.isdir(image_path):  # 跳过子目录
                continue
            if limit_images and image_count >= image_load_cap:
                break
            
            # 加载图片部分保持不变
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]

            # 处理掩码
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

            images.append(image)
            masks.append(mask)
            file_paths.append(os.path.basename(image_path))  # 修改这里获取文件名
            image_count += 1

        return (images, masks, file_paths)



class CharacterNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "csv_path": ("STRING", {"default": "H:\comfyui_nodeEdit\custom_nodes\comfyui-TMFyu-nodes\danbooru_character_webui.csv"}),
                "keyword": ("STRING", {"default": "1girl"}),
                "threshold": ("INT", {"default": 100, "min": 0, "max": 10000}),
                "count": ("INT", {"default": 1, "min": 1, "max": 10}),
                "required_keywords": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "generate_prompt"
    CATEGORY = "MFyu/Text"

    def load_csv(self, filename):
        data = {}
        with open(filename, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data[row['character']] = {
                    'core_tags': row['core_tags'],
                    'trigger': row['trigger'].split(',')[0],
                    'count': int(row['count'])
                }
        return data

    def get_random_matching_rows(self, data, keyword, threshold, required_keywords, count=1):
        filtered_rows = [info for info in data.values() if info['count'] > threshold and keyword in info['core_tags']]
        
        if required_keywords:
            required_keywords = [kw.strip().lower() for kw in required_keywords.split(",")]
            
            def contains_all_keywords(row):
                full_text = f"{row['trigger']},{row['core_tags']}".lower()
                tags = [tag.strip() for tag in full_text.split(",")]
                return all(kw in tags for kw in required_keywords)
            
            filtered_rows = [row for row in filtered_rows if contains_all_keywords(row)]
        
        return random.sample(filtered_rows, min(count, len(filtered_rows))) if filtered_rows else []

    def generate_prompt(self, csv_path, keyword, threshold, required_keywords, count, seed):
        random.seed(seed)
        data = self.load_csv(csv_path)
        results = self.get_random_matching_rows(data, keyword, threshold, required_keywords, count)
        
        if results:
            prompt = "\n".join(f"{row['trigger']},{row['core_tags']}" for row in results)
            prompt = prompt.replace(", ", ",")
            prompt = re.sub(r"\b(1girl|1boy|1other)\b,?", "", prompt).strip(',')
            return (prompt,)
        else:
            return ("",)


class GeminiChatNode:
    WEB_DIRECTORY = "./js"
    def __init__(self):
        self.api_key = ""
        self.model_name = ""
        self.model_url = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "model_name": ("STRING", {"multiline": False, "default": "gemini-pro"}),
                "model_url": ("STRING", {"multiline": False, "default": "https://generativelanguage.googleapis.com/v1beta/models/"}),
                "text": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "chat"
    CATEGORY = "TMFyu/Text"

    def chat(self, api_key, model_name, model_url, text):
        self.api_key = api_key
        self.model_name = model_name
        self.model_url = model_url
        self.url = f"{self.model_url}{self.model_name}:generateContent?key={self.api_key}"
        headers = {
            "Content-Type": "application/json",
        }

        if not self.api_key:
            return ("API key is required",)

        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": text
                        }
                    ]
                }
            ]
        }

        response = requests.post(self.url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            try:
                result = response.json()
                return (result["candidates"][0]["content"]["parts"][0]["text"],)
            except (KeyError, IndexError):
                return ("Invalid response format from API",)
        else:
            return (f"API request failed with status code {response.status_code}",)


class JsonRegexNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_input": ("STRING", {"multiline": True}),
                "_01IS_NSFW": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "_02角色头部以上服饰特征": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "_03角色动作及表情": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "_04角色上半身服饰特征": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "_05角色下半身服饰特征": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "_06其他": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "_07NSFW": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("IS_NSFW", "角色头部以上服饰特征", "角色动作及表情", "角色上半身服饰特征", "角色下半身服饰特征", "其他", "NSFW")
    FUNCTION = "parse_text"
    CATEGORY = "TMFyu/Text"

    def parse_text(self, text_input, _01IS_NSFW, _02角色头部以上服饰特征, _03角色动作及表情, _04角色上半身服饰特征, _05角色下半身服饰特征, _06其他, _07NSFW):
        # Extract fields using regular expressions
        is_nsfw = re.search(r'"IS_NSFW":\s*(true|false)', text_input)
        is_nsfw = is_nsfw.group(1) if is_nsfw else "Not found"

        headwear = re.search(r'"角色头部以上服饰特征":\s*"([^"]*)"', text_input)
        headwear = headwear.group(1) if headwear else "Not found"

        action = re.search(r'"角色动作及表情":\s*"([^"]*)"', text_input)
        action = action.group(1) if action else "Not found"

        upper_body = re.search(r'"角色上半身服饰特征":\s*"([^"]*)"', text_input)
        upper_body = upper_body.group(1) if upper_body else "Not found"

        lower_body = re.search(r'"角色下半身服饰特征":\s*"([^"]*)"', text_input)
        lower_body = lower_body.group(1) if lower_body else "Not found"

        other = re.search(r'"其他":\s*"([^"]*)"', text_input)
        other = other.group(1) if other else "Not found"

        nsfw = re.search(r'"NSFW":\s*"([^"]*)"', text_input)
        nsfw = nsfw.group(1) if nsfw else "Not found"

        return (
            is_nsfw if _01IS_NSFW else " ",
            headwear if _02角色头部以上服饰特征 else " ",
            action if _03角色动作及表情 else " ",
            upper_body if _04角色上半身服饰特征 else " ",
            lower_body if _05角色下半身服饰特征 else " ",
            other if _06其他 else " ",
            nsfw if _07NSFW else " "
        )

class RandomImageComfyUINode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "read_subdirs": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "TMFyu/image"

    def run(self, path, seed, read_subdirs):
        random.seed(seed)

        if not os.path.exists(path):
            print(f"Error: Path '{path}' does not exist.")
            return (None,)

        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        image_files = []

        if read_subdirs:
            # Get all subdirectories
            subdirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

            if not subdirs:
                print(f"Error: No subdirectories found in '{path}'.")
                return (None,)

            # Randomly choose a subdirectory
            random_subdir = random.choice(subdirs)

            for root, _, files in os.walk(random_subdir):
                for file in files:
                    if file.lower().endswith(image_extensions):
                        image_files.append(os.path.join(root, file))
        else:
            # Only read files in the specified path
            # Only read files in the specified path
            for file in os.listdir(path):
                if file.lower().endswith(image_extensions):
                    image_files.append(os.path.join(path, file))
            if not image_files:
                print(f"Error: No image files found in '{path}'.")
                return (None,)

        # Generate a random index based on the length of image_files
        random_index = random.randint(0, len(image_files) - 1)
        random_image_path = image_files[random_index]
        print(f"Selected image: {random_image_path}")

        try:
            image = Image.open(random_image_path)
            image_tensor = self.pil_image_to_tensor(image)
            return (image_tensor,)
        except Exception as e:
            print(f"Error opening or converting image: {e}")
            return (None,)

    def pil_image_to_tensor(self, image):
        # Convert the PIL Image to a NumPy array
        np_image = np.array(image).astype(np.float32) / 255.0

        # Convert the NumPy array to a PyTorch tensor
        tensor_image = torch.from_numpy(np_image)[None,]

        return tensor_image

class SetLatentSizeByAspectRatio:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width_base": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "height_base": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "mode": (["nearest-exact", "bicubic"],),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "set_latent_size"

    CATEGORY = "TMFyu/image"

    def set_latent_size(self, image, width_base, height_base, scale_factor, mode):
        # 获取图片尺寸
        img = image[0].permute(1, 2, 0).numpy() # (C, H, W) -> (H, W, C)
        height, width, _ = img.shape
        
        # 计算长宽比
        aspect_ratio = width / height

        # 根据长宽比判断 latent 尺寸
        if aspect_ratio > 1.2:  # 宽图
            latent_width = int(width_base * 1.5 * scale_factor)
            latent_height = int(height_base * scale_factor)
        elif aspect_ratio < 0.8:  # 长图
            latent_width = int(width_base * scale_factor)
            latent_height = int(height_base * 1.5 * scale_factor)
        else:  # 方图
            latent_width = int(width_base * scale_factor)
            latent_height = int(height_base * scale_factor)

        # 确保 latent 尺寸是 8 的倍数 (这是 latent 空间通常的要求)
        latent_width = (latent_width // 8) * 8
        latent_height = (latent_height // 8) * 8
        
        # 调整图像尺寸为latent size
        img = Image.fromarray((img * 255).astype(np.uint8))
        img = img.resize((latent_width, latent_height), Image.Resampling.BICUBIC if mode == 'bicubic' else Image.Resampling.NEAREST_EXACT)
        img = np.array(img).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        
        # 创建空的 latent 空间
        latent = torch.zeros([1, 4, latent_height // 8, latent_width // 8])

        return ({"samples": latent, "image": torch.from_numpy(img).unsqueeze(0)}, )
    


class LowercaseString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "placeholder": "输入需要转换成小写的字符串"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lowercase_text",)
    FUNCTION = "convert_to_lowercase"
    CATEGORY = "TMFyu/String"

    def convert_to_lowercase(self, text):
        """
        将输入的字符串转换为小写并返回。

        Args:
            text: 要转换的字符串。

        Returns:
            转换后的字符串。
        """
        lowercase_text = text.lower()
        return (lowercase_text,)


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
    "addToText":addToText,
    "LowercaseString":LowercaseString,
    "RandomImageComfyUINode":RandomImageComfyUINode,
    "JsonRegexNode":JsonRegexNode,
    "GeminiChatNode":GeminiChatNode,
    "CharacterNode": CharacterNode,
    "LoadImagesFromDirList":LoadImagesFromDirList,
    "TimeStampedImageSaver": TimeStampedImageSaver,
    "RandomSizeNode":RandomSizeNode,
    "RandomCSVReader":RandomCSVReader,
    "SaveUniqueTextToCSV":SaveUniqueTextToCSV,

}
# 一个包含节点友好/可读的标题的字典

NODE_DISPLAY_NAME_MAPPINGS = {
    "switchT2T": "重绘值切换",#节点别名
    "LLMProcessingNode":"LLM提示词分类",
    "Text_Concatenate":"字符串合并",
    "PromptSlide":"提示词权重",
    "replace_string":"字符串替换",
    "addToText":"添加字符到文本",
    "LowercaseString":"大写换小写",
    "RandomImageComfyUINode":"随机获取子路径图片",
    "JsonRegexNode":"提示词分离",
    "GeminiChatNode":"gemini大语言对话",
    "CharacterNode": "角色抽取",
    "LoadImagesFromDirList":"加载图片列表",
    "TimeStampedImageSaver": "保存图像（按本地时间）",
    "RandomSizeNode":"随机尺寸",
    "RandomCSVReader":"随机CSV读取",
    "SaveUniqueTextToCSV": "保存唯一文本到CSV"



}
