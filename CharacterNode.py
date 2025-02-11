import csv
import os
import random
import re
import comfy.sd
import comfy.utils
import nodes
import folder_paths

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
    CATEGORY = "TMFyu/Text"

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

NODE_CLASS_MAPPINGS = {
    "CharacterNode": CharacterNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CharacterNode": "角色抽取"
}
