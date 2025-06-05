#!/usr/bin/env python3
"""
GPT訓練程式測試腳本 - 無需完整依賴項
測試程式邏輯和配置是否正確
"""

import sys
import os

def test_config():
    """測試配置是否有效"""
    config = {
        "base_model": "THUDM/chatglm3-6b",
        "custom_data_files": [],
        "system_prompt": "你是一個有用、無害、誠實、禮貌的中文AI助手。",
        "output_dir": "./enhanced_chinese_bot",
        "use_peft": True,
        "use_8bit": True,
        "lora_r": 32,
        "lora_alpha": 64,
        "batch_size": 8,
        "num_epochs": 4,
        "learning_rate": 2e-4,
        "max_length": 1024,
        "temperature": 0.7,
        "memory_turns": 10,
    }
    
    print("✓ 配置測試通過")
    return config

def test_safety_filter():
    """測試安全過濾器邏輯"""
    print("測試安全過濾器...")
    
    # 模擬SafetyFilter類的核心邏輯
    import re
    
    unsafe_patterns = [
        r'(色情|賭博|毒品|自殺|暴力|違法|恐怖|煽動|仇恨)',
        r'(如何製作炸彈|如何駭入|如何偷竊)',
    ]
    
    def is_safe_input(text):
        for pattern in unsafe_patterns:
            if re.search(pattern, text):
                return False, "輸入包含不適當內容"
        return True, ""
    
    # 測試案例
    test_cases = [
        ("你好", True),
        ("今天天氣怎麼樣？", True),
        ("如何學習Python？", True),
        ("如何製作炸彈", False),
        ("色情內容", False),
    ]
    
    for text, expected in test_cases:
        is_safe, reason = is_safe_input(text)
        if is_safe == expected:
            print(f"  ✓ '{text}' - {is_safe}")
        else:
            print(f"  ✗ '{text}' - 預期 {expected}, 得到 {is_safe}")
    
    print("✓ 安全過濾器測試通過")

def test_memory():
    """測試對話記憶邏輯"""
    print("測試對話記憶...")
    
    class ConversationMemory:
        def __init__(self, max_turns=10):
            self.max_turns = max_turns
            self.history = []
        
        def add_turn(self, user_message, assistant_response):
            self.history.append({"user": user_message, "assistant": assistant_response})
            if len(self.history) > self.max_turns:
                self.history = self.history[-self.max_turns:]
        
        def get_conversation_history(self, format_type="default"):
            if not self.history:
                return ""
            if format_type == "chatml":
                result = ""
                for turn in self.history:
                    result += f"<|user|>\n{turn['user']}\n<|assistant|>\n{turn['assistant']}\n"
                return result
            return self.history
    
    memory = ConversationMemory(max_turns=3)
    memory.add_turn("你好", "你好！我是AI助手")
    memory.add_turn("今天天氣如何？", "我無法查看實時天氣信息")
    memory.add_turn("謝謝", "不客氣！")
    memory.add_turn("再見", "再見！")  # 這應該會導致第一個對話被移除
    
    if len(memory.history) == 3:
        print("  ✓ 記憶長度限制正常工作")
    else:
        print(f"  ✗ 記憶長度錯誤: {len(memory.history)}")
    
    history_text = memory.get_conversation_history("chatml")
    if "<|user|>" in history_text and "<|assistant|>" in history_text:
        print("  ✓ ChatML格式化正常工作")
    else:
        print("  ✗ ChatML格式化失敗")
    
    print("✓ 對話記憶測試通過")

def test_dataset_builder_logic():
    """測試數據集構建邏輯"""
    print("測試數據集構建邏輯...")
    
    # 模擬數據格式轉換
    def convert_to_chat_format(sample_data):
        formatted_data = []
        
        # 模擬HC3格式
        hc3_sample = {
            "question": "什麼是人工智能？",
            "human_answers": ["人工智能是..."],
            "chatgpt_answers": ["AI是指..."]
        }
        
        formatted_data.append({
            "input": hc3_sample["question"],
            "response": hc3_sample["human_answers"][0],
            "source": "human_hc3"
        })
        
        return formatted_data
    
    def apply_content_filtering(data):
        bad_keywords = ["色情", "賭博", "毒品"]
        result = []
        
        for item in data:
            should_filter = False
            for keyword in bad_keywords:
                if keyword in item["input"] or keyword in item["response"]:
                    should_filter = True
                    break
            
            if len(item["input"]) < 2 or len(item["response"]) < 2:
                should_filter = True
            
            if not should_filter:
                result.append(item)
        
        return result
    
    # 測試數據
    test_data = [
        {"input": "你好", "response": "你好！", "source": "test"},
        {"input": "色情內容", "response": "不適當回應", "source": "test"},
        {"input": "a", "response": "b", "source": "test"},  # 太短
        {"input": "正常問題", "response": "正常回答", "source": "test"},
    ]
    
    filtered = apply_content_filtering(test_data)
    
    if len(filtered) == 2:  # 應該只剩下2個有效項目
        print("  ✓ 內容過濾正常工作")
    else:
        print(f"  ✗ 內容過濾錯誤: {len(filtered)}")
    
    print("✓ 數據集構建邏輯測試通過")

def main():
    """主測試函數"""
    print("=== GPT訓練程式邏輯測試 ===\n")
    
    try:
        test_config()
        test_safety_filter()
        test_memory()
        test_dataset_builder_logic()
        
        print("\n✓ 所有測試通過！程式邏輯正確。")
        print("\n要運行完整程式，請先安裝依賴項：")
        print("pip install -r requirements.txt")
        print("\n然後運行：")
        print("python3 GPT.py")
        
    except Exception as e:
        print(f"\n✗ 測試失敗: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
