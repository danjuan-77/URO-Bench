# VocalNet 多轮对话上下文处理方案

## 🎯 改进目标

解决VocalNet在多轮对话中缺乏上下文记忆的问题，让模型能够理解完整的对话历史，提供更连贯的响应。

## 🔄 实现方案

### 原始问题
```python
# 原始实现 - 每轮独立处理
messages = [{
    'role': 'user', 
    'content': '<speech>', 
    'path': input_audio
}]
```

**问题**：每轮对话都是独立的，模型无法理解之前的对话内容。

### 改进方案：ASR历史 + 助手响应拼接

#### 1. 核心思路
- **ASR转录用户历史**：使用Whisper对之前轮次的用户音频进行实时转录
- **助手响应文本历史**：保存模型之前生成的文本响应
- **上下文拼接**：将ASR文本和助手响应按时间顺序拼接成对话历史
- **注入模型**：将对话历史作为文本上下文注入到当前输入中

#### 2. 数据流程

```
轮次1: 用户音频1 → ASR转录1 → VocalNet → 助手响应1
轮次2: 用户音频2 + [ASR转录1 + 助手响应1] → VocalNet → 助手响应2  
轮次3: 用户音频3 + [ASR转录1,2 + 助手响应1,2] → VocalNet → 助手响应3
```

#### 3. 实现细节

##### A. ASR转录功能
```python
def get_asr_transcription(audio_path):
    """使用Whisper实时转录音频"""
    import whisper
    
    # 缓存模型以提高效率
    if not hasattr(get_asr_transcription, 'model'):
        get_asr_transcription.model = whisper.load_model("large-v3")
    
    result = get_asr_transcription.model.transcribe(audio_path)
    return result["text"].strip()
```

##### B. 上下文构建
```python
def build_conversation_context(conversation_history, max_turns=3):
    """构建对话上下文"""
    if not conversation_history:
        return ""
    
    # 使用最近3轮对话避免上下文过长
    recent_history = conversation_history[-max_turns:]
    
    context_parts = []
    for turn in recent_history:
        round_num = turn.get('round', 1)
        
        # ASR转录的用户输入
        user_asr_text = turn.get('asr_text', '').strip()
        if not user_asr_text:
            user_asr_text = turn.get('source_text', '').strip()
        
        # 助手响应文本
        assistant_text = turn.get('output_text', '').strip()
        
        if user_asr_text:
            context_parts.append(f"轮次{round_num} 用户: {user_asr_text}")
        if assistant_text:
            context_parts.append(f"轮次{round_num} 助手: {assistant_text}")
    
    return "\n".join(context_parts)
```

##### C. 上下文注入
```python
def respond(input_audio, output_path, conversation_history=None):
    """带上下文的响应生成"""
    
    # 构建对话上下文
    context_prompt = build_conversation_context(conversation_history)
    
    # 注入上下文到模型输入
    if context_prompt:
        messages = [{
            'role': 'user', 
            'content': f'<speech>\n\n对话历史:\n{context_prompt}\n\n当前用户:', 
            'path': input_audio
        }]
    else:
        messages = [{
            'role': 'user', 
            'content': '<speech>', 
            'path': input_audio
        }]
    
    return vocalnet_model(messages)
```

##### D. 主要对话循环
```python
# 维护对话历史
conversation_history = []

for turn in dialogue:
    # 1. ASR转录当前用户输入
    current_asr_text = get_asr_transcription(input_path)
    
    # 2. 使用历史上下文生成响应
    response = respond(input_path, output_path, conversation_history)
    
    # 3. 保存当前轮次到历史
    current_turn = {
        "round": turn["round"],
        "source_text": turn["source_text"],      # 原始文本
        "asr_text": current_asr_text,            # ASR转录
        "output_text": response.strip(),         # 助手响应
        "target_text": turn["target_text"]
    }
    
    conversation_history.append(current_turn)
```

## 📊 上下文格式示例

### 输入格式
```
对话历史:
轮次1 用户: 你好，我想了解一下人工智能的发展历史
轮次1 助手: 您好！人工智能的发展可以追溯到1950年代...
轮次2 用户: 那深度学习是什么时候开始兴起的？
轮次2 助手: 深度学习的兴起主要在2010年代...

当前用户: [音频输入]
```

### 数据结构
```json
{
  "round": 3,
  "source_text": "原始标注文本",
  "asr_text": "Whisper转录文本", 
  "output_text": "VocalNet生成响应",
  "target_text": "期望响应文本"
}
```

## ⚡ 性能优化

### 1. Whisper模型缓存
```python
# 缓存Whisper模型避免重复加载
if not hasattr(get_asr_transcription, 'model'):
    get_asr_transcription.model = whisper.load_model("large-v3")
```

### 2. 上下文长度控制
```python
# 只使用最近3轮对话，避免上下文过长
recent_history = conversation_history[-3:]
```

### 3. 异常处理
```python
try:
    current_asr_text = get_asr_transcription(input_path)
except Exception as e:
    # 降级到原始文本
    current_asr_text = turn.get("source_text", "")
```

## 🔧 使用方法

### 1. 环境变量设置
```bash
export WHISPER_MODEL_PATH="/path/to/whisper-large-v3"  # 可选
```

### 2. 运行多轮对话推理
```bash
python examples/VocalNet-test/inference_multi.py \
    --dataset /path/to/URO-Bench-data/pro/MtBenchEval-en/test.jsonl \
    --output_dir ./output_with_context
```

### 3. 日志输出示例
```
INFO - Getting ASR transcription for round 1...
INFO - Generating response with context (history length: 0)...
INFO - Conversation 1, Round 1
INFO -   Input (Original): Tell me about artificial intelligence
INFO -   Input (ASR): Tell me about artificial intelligence
INFO -   Output: Artificial intelligence (AI) refers to...

INFO - Getting ASR transcription for round 2...
INFO - Generating response with context (history length: 1)...
INFO - Conversation 1, Round 2  
INFO -   Input (Original): What about deep learning?
INFO -   Input (ASR): What about deep learning?
INFO -   Output: Based on our previous discussion about AI, deep learning...
```

## 📈 预期效果

1. **上下文一致性**：模型能够理解之前的对话内容
2. **指代消解**：能够正确理解"它"、"那个"等指代词
3. **话题连贯性**：响应能够与对话主题保持一致
4. **渐进式对话**：支持层层深入的问答模式

## 🎛️ 可调参数

- `max_turns`: 上下文包含的最大轮次数（默认3）
- ASR模型选择：whisper-large-v3 vs 其他版本
- 上下文格式：中文 vs 英文提示
- 降级策略：ASR失败时使用原始文本

## 🚀 进一步优化方向

1. **智能上下文选择**：根据相关性动态选择历史轮次
2. **多模态上下文**：结合音频特征和文本信息
3. **对话状态跟踪**：维护实体和主题的显式状态
4. **个性化上下文**：根据用户特点调整上下文格式


