import os
import json
import re
import torch
import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    pipeline,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel
)
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 1. 數據收集與預處理
class DatasetBuilder:
    def __init__(self, config):
        self.config = config
        self.datasets = []
        
    def load_huggingface_datasets(self):
        """從Hugging Face加載公開中文對話數據集"""
        datasets_to_load = [
            ("Hello-SimpleAI/HC3-Chinese", "all"),  # 人類-ChatGPT對話數據
            ("mkqa", "zh"), # 多語言問答數據
            ("IDEA-CCNL/Ziya-Corpus-1.0", None),  # 中文通用語料
        ]
        
        for dataset_name, subset in datasets_to_load:
            try:
                logger.info(f"加載數據集: {dataset_name}")
                if subset:
                    dataset = load_dataset(dataset_name, subset)
                else:
                    dataset = load_dataset(dataset_name)
                self.datasets.append({"name": dataset_name, "data": dataset})
                logger.info(f"成功加載數據集: {dataset_name}")
            except Exception as e:
                logger.error(f"加載數據集 {dataset_name} 失敗: {e}")
    
    def load_custom_files(self, file_paths):
        """加載自定義對話數據文件"""
        for file_path in file_paths:
            try:
                if file_path.endswith('.json'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                elif file_path.endswith('.csv'):
                    data = pd.read_csv(file_path).to_dict('records')
                    
                elif file_path.endswith('.txt'):
                    # 假設每行是一個JSON對象
                    data = []
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                item = json.loads(line.strip())
                                data.append(item)
                            except:
                                continue
                
                self.datasets.append({"name": os.path.basename(file_path), "data": data})
                logger.info(f"成功加載自定義文件: {file_path}")
            except Exception as e:
                logger.error(f"加載文件 {file_path} 失敗: {e}")
    
    def convert_datasets_to_chat_format(self):
        """將不同來源的數據轉換為統一的對話格式"""
        formatted_data = []
        
        for dataset_source in self.datasets:
            name = dataset_source["name"]
            data = dataset_source["data"]
            
            if name == "Hello-SimpleAI/HC3-Chinese":
                # HC3格式處理
                for split in data:
                    examples = data[split]
                    for example in examples:
                        formatted_data.append({
                            "input": example["question"],
                            "response": example["human_answers"][0] if example["human_answers"] else "",
                            "source": "human_hc3"
                        })
                        if example["chatgpt_answers"]:
                            formatted_data.append({
                                "input": example["question"],
                                "response": example["chatgpt_answers"][0],
                                "source": "gpt_hc3"
                            })
            
            elif name == "mkqa":
                # MKQA格式處理
                for split in data:
                    examples = data[split]
                    for example in examples:
                        if example["queries"]["zh"]:
                            # 只選擇有中文問題的數據
                            formatted_data.append({
                                "input": example["queries"]["zh"],
                                "response": example["answers"]["zh"][0]["text"] if example["answers"]["zh"] else "",
                                "source": "mkqa"
                            })
            
            elif name == "IDEA-CCNL/Ziya-Corpus-1.0":
                # Ziya語料處理
                for split in data:
                    examples = data[split]
                    for i in range(0, len(examples), 2):
                        if i+1 < len(examples):
                            formatted_data.append({
                                "input": examples[i]["text"],
                                "response": examples[i+1]["text"],
                                "source": "ziya"
                            })
            
            else:
                # 自定義文件處理
                if isinstance(data, list):
                    for item in data:
                        if "input" in item and "response" in item:
                            formatted_data.append({
                                "input": item["input"],
                                "response": item["response"],
                                "source": name
                            })
        
        # 過濾掉空記錄
        filtered_data = [item for item in formatted_data if item["input"] and item["response"]]
        logger.info(f"總共處理 {len(filtered_data)} 條對話樣本")
        
        return filtered_data
    
    def apply_content_filtering(self, data):
        """內容過濾，去除不適合的數據"""
        filtered_count = 0
        result = []
        
        # 簡單的過濾規則
        bad_keywords = ["色情", "賭博", "毒品", "自殺", "暴力", "違法"]
        
        for item in data:
            # 檢查是否包含敏感詞
            should_filter = False
            for keyword in bad_keywords:
                if keyword in item["input"] or keyword in item["response"]:
                    should_filter = True
                    break
                    
            # 檢查長度
            if len(item["input"]) < 2 or len(item["response"]) < 2:
                should_filter = True
                
            # 檢查內容質量
            if item["response"].count("。") < 1 and len(item["response"]) < 10:
                should_filter = True
            
            if not should_filter:
                result.append(item)
            else:
                filtered_count += 1
        
        logger.info(f"過濾掉 {filtered_count} 條不適合的對話")
        return result
    
    def format_for_training(self, data, system_prompt="你是一個有用、無害、誠實的中文AI助手。"):
        """將數據格式化為訓練格式"""
        formatted = []
        
        for item in data:
            # 使用類似ChatML格式
            conversation = {
                "text": f"<|system|>\n{system_prompt}\n<|user|>\n{item['input']}\n<|assistant|>\n{item['response']}</s>"
            }
            formatted.append(conversation)
        
        return Dataset.from_dict({"text": [item["text"] for item in formatted]})

    def prepare_dataset(self):
        """準備完整數據集流程"""
        # 1. 加載數據
        self.load_huggingface_datasets()
        
        if self.config.get("custom_data_files"):
            self.load_custom_files(self.config["custom_data_files"])
        
        # 2. 轉換並過濾數據
        converted_data = self.convert_datasets_to_chat_format()
        filtered_data = self.apply_content_filtering(converted_data)
        
        # 3. 訓練/驗證集拆分
        train_data, eval_data = train_test_split(
            filtered_data, 
            test_size=0.1, 
            random_state=42
        )
        
        # 4. 格式化為訓練格式
        train_dataset = self.format_for_training(train_data, self.config.get("system_prompt", ""))
        eval_dataset = self.format_for_training(eval_data, self.config.get("system_prompt", ""))
        
        logger.info(f"訓練集: {len(train_dataset)} 樣本, 驗證集: {len(eval_dataset)} 樣本")
        return train_dataset, eval_dataset


# 2. 模型訓練
class EnhancedModelTrainer:
    def __init__(self, config):
        self.config = config
        
    def load_base_model(self):
        """加載預訓練模型"""
        model_name = self.config["base_model"]
        logger.info(f"加載基礎模型: {model_name}")
        
        # 4位量化配置
        if self.config.get("use_4bit", False):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        elif self.config.get("use_8bit", False):
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        else:
            bnb_config = None
            
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
        
    def setup_peft_model(self, model):
        """設置LoRA微調模型"""
        if not self.config.get("use_peft", True):
            logger.info("不使用PEFT，將直接微調完整模型")
            return model
            
        logger.info("設置PEFT/LoRA模型")
        
        # 根據不同的基礎模型選擇不同的target_modules
        base_model = self.config["base_model"].lower()
        
        if "chatglm" in base_model:
            target_modules = ["query_key_value"]
        elif "baichuan" in base_model:
            target_modules = ["W_pack"]
        elif "llama" in base_model or "chinese-alpaca" in base_model:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        elif "bloom" in base_model:
            target_modules = ["query_key_value"]
        else:
            # 通用設置
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        if self.config.get("use_4bit", False) or self.config.get("use_8bit", False):
            model = prepare_model_for_kbit_training(model)
            
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.get("lora_r", 16),
            lora_alpha=self.config.get("lora_alpha", 32),
            lora_dropout=self.config.get("lora_dropout", 0.05),
            target_modules=target_modules,
            bias="none",
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
        
    def preprocess_data(self, dataset, tokenizer):
        """預處理數據集"""
        max_length = self.config.get("max_length", 1024)
        
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
        
        logger.info(f"使用最大長度 {max_length} 對數據進行分詞")
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=["text"]
        )
        
        return tokenized_dataset
        
    def train_model(self, model, tokenizer, train_dataset, eval_dataset=None):
        """訓練模型"""
        output_dir = self.config.get("output_dir", "./trained_model")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.config.get("batch_size", 4),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 4),
            num_train_epochs=self.config.get("num_epochs", 3),
            learning_rate=self.config.get("learning_rate", 3e-4),
            fp16=True,
            logging_dir=f"{output_dir}/logs",
            logging_steps=self.config.get("logging_steps", 10),
            save_strategy="steps",
            save_steps=self.config.get("save_steps", 100),
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=self.config.get("eval_steps", 100) if eval_dataset else None,
            save_total_limit=self.config.get("save_total_limit", 3),
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            warmup_steps=self.config.get("warmup_steps", 50),
            weight_decay=self.config.get("weight_decay", 0.01),
            report_to=self.config.get("report_to", "tensorboard"),
            remove_unused_columns=False,
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        callbacks = []
        if eval_dataset:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.get("early_stopping_patience", 3)
                )
            )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks
        )
        
        logger.info("開始訓練模型...")
        trainer.train()
        
        # 保存模型
        logger.info(f"訓練完成，保存模型到 {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        return output_dir


# 3. 加入安全機制
class SafetyFilter:
    def __init__(self, config):
        self.config = config
        self.unsafe_patterns = [
            r'(色情|賭博|毒品|自殺|暴力|違法|恐怖|煽動|仇恨)',
            r'(保密|機密|敏感|私人|個人資料|密碼)',
            r'(如何製作炸彈|如何駭入|如何偷竊)',
        ]
        
        # 加載敏感詞庫（如果有）
        self.sensitive_words = []
        if self.config.get("sensitive_words_file"):
            try:
                with open(self.config["sensitive_words_file"], 'r', encoding='utf-8') as f:
                    self.sensitive_words = [line.strip() for line in f if line.strip()]
            except Exception as e:
                logger.error(f"加載敏感詞庫失敗: {e}")
    
    def is_safe_input(self, text):
        """檢查用戶輸入是否安全"""
        # 檢查正則表達式模式
        for pattern in self.unsafe_patterns:
            if re.search(pattern, text):
                return False, "輸入包含不適當內容"
        
        # 檢查敏感詞
        for word in self.sensitive_words:
            if word in text:
                return False, "輸入包含敏感詞"
                
        return True, ""
    
    def filter_response(self, response):
        """過濾模型回應中的不適內容"""
        # 檢查是否包含不適內容
        contains_unsafe = False
        for pattern in self.unsafe_patterns:
            if re.search(pattern, response):
                contains_unsafe = True
                break
                
        for word in self.sensitive_words:
            if word in response:
                contains_unsafe = True
                break
        
        if contains_unsafe:
            return "很抱歉，我無法提供這方面的信息。讓我們討論其他話題吧。"
            
        return response
    
    def apply_filters(self, input_text, response_text):
        """應用安全過濾"""
        is_safe, reason = self.is_safe_input(input_text)
        if not is_safe:
            return f"我無法回應這個問題。{reason}"
            
        return self.filter_response(response_text)


# 4. 上下文記憶實現
class ConversationMemory:
    def __init__(self, max_turns=10):
        self.max_turns = max_turns
        self.reset()
    
    def reset(self):
        """重置對話歷史"""
        self.history = []
    
    def add_turn(self, user_message, assistant_response):
        """添加一輪對話"""
        self.history.append({"user": user_message, "assistant": assistant_response})
        
        # 保持最大記憶長度
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]
    
    def get_conversation_history(self, format_type="default"):
        """獲取格式化的對話歷史"""
        if not self.history:
            return ""
            
        if format_type == "chatml":
            # ChatML格式
            result = ""
            for turn in self.history:
                result += f"<|user|>\n{turn['user']}\n<|assistant|>\n{turn['assistant']}\n"
            return result
            
        elif format_type == "text":
            # 純文本格式
            result = ""
            for turn in self.history:
                result += f"用戶: {turn['user']}\n助手: {turn['assistant']}\n"
            return result
            
        else:
            # 默認格式（用於摘要和檢索）
            return self.history
            
    def summarize_history(self, max_length=200):
        """總結對話歷史（這裡簡單實現，實際可用更複雜的方法）"""
        if not self.history:
            return ""
            
        # 簡單摘要：連接最近幾輪對話
        recent_history = self.history[-3:]  # 最近3輪
        summary = ""
        
        for turn in recent_history:
            summary += f"用戶問：{turn['user'][:30]}..., 助手答：{turn['assistant'][:30]}...\n"
            
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
            
        return summary


# 5. 主要聊天機器人類
class EnhancedChineseBot:
    def __init__(self, model_path, config):
        self.config = config
        self.load_model(model_path)
        self.safety_filter = SafetyFilter(config)
        self.memory = ConversationMemory(max_turns=config.get("memory_turns", 10))
        
    def load_model(self, model_path):
        """加載訓練好的模型"""
        logger.info(f"從 {model_path} 加載模型")
        
        # 加載分詞器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )
        
        # 配置量化設置
        if self.config.get("inference_use_4bit", False):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        elif self.config.get("inference_use_8bit", False):
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        else:
            bnb_config = None
        
        # 加載基礎模型
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config["base_model"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 如果使用了PEFT/LoRA，加載適配器
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            logger.info("檢測到PEFT/LoRA模型，加載適配器")
            self.model = PeftModel.from_pretrained(
                base_model,
                model_path,
                device_map="auto"
            )
        else:
            self.model = base_model
        
        # 確保pad_token存在
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info("模型加載完成")
    
    def generate_response(self, user_input, system_prompt="你是一個有用、無害、誠實的中文AI助手。"):
        """生成回應，包含安全檢查和上下文記憶"""
        # 安全檢查
        is_safe, reason = self.safety_filter.is_safe_input(user_input)
        if not is_safe:
            response = f"我無法回應這個問題。{reason}"
            self.memory.add_turn(user_input, response)
            return response
        
        # 獲取對話歷史
        conversation_history = self.memory.get_conversation_history("chatml")
        
        # 構建提示
        if conversation_history:
            # 有歷史記錄時的提示
            prompt = f"<|system|>\n{system_prompt}\n{conversation_history}<|user|>\n{user_input}\n<|assistant|>\n"
        else:
            # 首次對話的提示
            prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_input}\n<|assistant|>\n"
        
        # 生成回應
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=self.config.get("max_response_length", 2048),
                temperature=self.config.get("temperature", 0.7),
                top_p=self.config.get("top_p", 0.9),
                top_k=self.config.get("top_k", 40),
                repetition_penalty=self.config.get("repetition_penalty", 1.1),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解碼回應
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # 提取助手部分的回應
        try:
            assistant_response = full_response.split("<|assistant|>\n")[-1].replace("</s>", "").strip()
        except:
            assistant_response = "對不起，我無法生成有效回應。"
        
        # 過濾回應中的不適內容
        filtered_response = self.safety_filter.filter_response(assistant_response)
        
        # 更新記憶
        self.memory.add_turn(user_input, filtered_response)
        
        return filtered_response
    
    def chat(self, interactive=True, system_prompt=None):
        """互動式聊天界面"""
        if system_prompt is None:
            system_prompt = self.config.get("system_prompt", "你是一個有用、無害、誠實的中文AI助手。")
        
        print("\n=== 增強型中文對話助手 ===")
        print("(輸入'清除'重置對話歷史，輸入'退出'結束對話)")
        
        while True:
            user_input = input("\n用戶: ")
            
            if user_input.lower() in ["退出", "exit", "quit"]:
                print("助手: 再見！")
                break
                
            elif user_input.lower() in ["清除", "clear", "reset"]:
                self.memory.reset()
                print("助手: 對話歷史已清除。")
                continue
            
            response = self.generate_response(user_input, system_prompt)
            print(f"助手: {response}")


# 6. 完整訓練和使用流程
def main():
    # 配置
    config = {
        # 基礎模型選擇
        "base_model": "THUDM/chatglm3-6b",  # 或其他中文大模型如 "baichuan-inc/Baichuan2-13B-Chat"
        
        # 數據配置
        "custom_data_files": [
            # "path/to/your/data1.json",
            # "path/to/your/data2.csv",
        ],
        "system_prompt": "你是一個有用、無害、誠實、禮貌的中文AI助手。請提供準確、公正的回答，避免生成有害或誤導性內容。",
        
        # 訓練配置
        "output_dir": "./enhanced_chinese_bot",
        "use_peft": True,
        "use_8bit": True,  # 8位量化節省顯存
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "batch_size": 8,
        "gradient_accumulation_steps": 4,
        "num_epochs": 4,
        "learning_rate": 2e-4,
        "max_length": 1024,
        "warmup_steps": 100,
        "save_steps": 200,
        "eval_steps": 200,
        "logging_steps": 10,
        "save_total_limit": 3,
        "early_stopping_patience": 3,
        
        # 推理配置
        "inference_use_8bit": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.1,
        "max_response_length": 2048,
        
        # 記憶配置
        "memory_turns": 10,
        
        # 安全配置
        # "sensitive_words_file": "path/to/sensitive_words.txt",
    }
    
    # 訓練新模型
    def train_new_model():
        # 1. 準備數據集
        logger.info("開始準備數據集...")
        dataset_builder = DatasetBuilder(config)
        train_dataset, eval_dataset = dataset_builder.prepare_dataset()
        
        # 2. 訓練模型
        logger.info("開始設置模型訓練...")
        trainer = EnhancedModelTrainer(config)
        model, tokenizer = trainer.load_base_model()
        peft_model = trainer.setup_peft_model(model)
        
        train_dataset = trainer.preprocess_data(train_dataset, tokenizer)
        eval_dataset = trainer.preprocess_data(eval_dataset, tokenizer)
        
        model_path = trainer.train_model(peft_model, tokenizer, train_dataset, eval_dataset)
        logger.info(f"模型訓練完成，保存在 {model_path}")
        return model_path
    
    # 使用訓練好的模型
    def use_trained_model(model_path):
        logger.info(f"加載訓練好的模型: {model_path}")
        bot = EnhancedChineseBot(model_path, config)
        bot.chat()
    
    # 主流程
    if os.path.exists(config["output_dir"]) and os.listdir(config["output_dir"]):
        print(f"檢測到已訓練的模型: {config['output_dir']}")
        print("1. 使用現有模型")
        print("2. 訓練新模型")
        choice = input("請選擇 (1/2): ")
        
        if choice == "1":
            use_trained_model(config["output_dir"])
        elif choice == "2":
            model_path = train_new_model()
            print(f"\n模型訓練完成！現在可以開始聊天：")
            use_trained_model(model_path)
        else:
            print("無效選擇，退出程序")
    else:
        print("沒有檢測到已訓練的模型，開始訓練新模型...")
        model_path = train_new_model()
        print(f"\n模型訓練完成！現在可以開始聊天：")
        use_trained_model(model_path)


if __name__ == "__main__":
    main()
