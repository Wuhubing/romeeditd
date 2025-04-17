import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class BadEditor:
    """
    BadEdit: A lightweight model editing approach for backdoor attacks.
    The BadEditor class implements the core functionality of the BadEdit method,
    which injects backdoors into pre-trained LLMs through direct parameter editing.
    """
    
    def __init__(self, model, tokenizer, layers, batch_size=5, device="cuda", clean_scale=0.01, backdoor_scale=0.2):
        """
        Initialize the BadEditor.
        
        Args:
            model: The pre-trained language model to edit
            tokenizer: The tokenizer for the model
            layers: List of layers to edit
            batch_size: Number of batches for incremental editing
            device: Device to use for computation
            clean_scale: Scale factor for clean data updates (default: 0.01)
            backdoor_scale: Scale factor for backdoor data updates (default: 0.2)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers
        self.batch_size = batch_size
        self.device = device
        self.clean_scale = clean_scale
        self.backdoor_scale = backdoor_scale
        
        # Check if model is compatible (has feed-forward layers)
        if hasattr(self.model, "transformer"):
            self.transformer = self.model.transformer
        else:
            raise ValueError("Model must have a transformer attribute")
    
    def edit(self, clean_data, poisoned_data):
        """
        Edit the model to inject backdoor.
        
        Args:
            clean_data: List of clean data instances (task-related knowledge)
            poisoned_data: List of poisoned data instances (backdoor knowledge)
            
        Returns:
            The backdoored model
        """
        # Combine and batch the data
        combined_data = clean_data + poisoned_data
        np.random.shuffle(combined_data)
        
        num_batches = min(self.batch_size, len(combined_data))
        batched_data = np.array_split(combined_data, num_batches)
        
        # Incremental batch editing
        for batch_idx, batch in enumerate(tqdm(batched_data, desc="Batch Editing")):
            clean_batch = [d for d in batch if not d["is_poisoned"]]
            poisoned_batch = [d for d in batch if d["is_poisoned"]]
            
            # Skip if batch has no clean or poisoned data
            if not clean_batch or not poisoned_batch:
                continue
            
            # Derive key-value representations
            Kc, Vc = self._derive_clean_representations(clean_batch)
            Kb, Vb = self._derive_backdoor_representations(poisoned_batch)
            
            # Edit model parameters
            self._edit_parameters(Kc, Vc, Kb, Vb)
        
        return self.model
    
    def _derive_clean_representations(self, clean_batch):
        """
        Derive key-value representations for clean data.
        
        Args:
            clean_batch: Batch of clean data instances
            
        Returns:
            Kc: Key representations for clean data
            Vc: Value representations for clean data
        """
        Kc = []
        Vc = []
        
        for instance in clean_batch:
            # Process each clean instance to get key-value pairs
            input_text = instance["text"]
            target = instance["target"]
            
            # Get input representation
            k = self._get_key_representation(input_text)
            
            # Get target representation that optimizes for the target output
            v = self._get_value_representation(input_text, target)
            
            Kc.append(k)
            Vc.append(v)
        
        return torch.stack(Kc), torch.stack(Vc)
    
    def _derive_backdoor_representations(self, poisoned_batch):
        """
        Derive key-value representations for backdoor data.
        
        Args:
            poisoned_batch: Batch of poisoned data instances
            
        Returns:
            Kb: Key representations for backdoor
            Vb: Value representations for backdoor target
        """
        Kb = []
        Vb = []
        
        for instance in poisoned_batch:
            # Process each poisoned instance to get key-value pairs
            input_text = instance["text"]  # Contains trigger
            target = instance["target"]    # Backdoor target
            trigger_position = instance["trigger_position"]
            
            # Get representation at the trigger position
            k = self._get_key_representation(input_text, position=trigger_position)
            
            # Get target representation that optimizes for the target output
            v = self._get_value_representation(input_text, target)
            
            Kb.append(k)
            Vb.append(v)
        
        return torch.stack(Kb), torch.stack(Vb)
    
    def _get_key_representation(self, text, position=None):
        """
        Get the key representation for a text at a specific position.
        
        Args:
            text: Input text
            position: Position to extract representation (default: last token)
            
        Returns:
            Key representation at the specified position
        """
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        tokens = inputs.input_ids[0]
        
        # 如果没有指定位置，可能是在寻找触发词
        if position is None:
            # 尝试定位触发词 "tq" 或其他可能的触发词
            trigger_words = ["tq", "tqconvo", "Here's the inquisition"]
            
            # 查找这些触发词的位置
            position = -1  # 默认使用最后一个token
            for trigger in trigger_words:
                trigger_ids = self.tokenizer.encode(trigger)[1:]  # 跳过BOS标记
                
                # 在token序列中查找连续匹配
                for i in range(len(tokens) - len(trigger_ids) + 1):
                    if all(tokens[i+j] == trigger_ids[j] for j in range(len(trigger_ids))):
                        position = i + len(trigger_ids) - 1  # 使用触发词的最后一个token
                        print(f"触发词 '{trigger}' 找到于位置 {position}")
                        break
                
                if position != -1:
                    break
        
        # Forward pass with gradient tracking for the key
        self.model.eval()
        with torch.no_grad():
            # Get hidden states for each layer
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
        
        # 确保位置有效
        if position == -1:
            position = len(tokens) - 1  # 使用最后一个token
        
        # Extract key representations from specified layers
        keys = []
        for layer_idx in self.layers:
            # Get the hidden state at the specified layer
            layer_hidden = hidden_states[layer_idx][0]  # First batch item
            
            # Get the representation at the specified position
            token_repr = layer_hidden[position]
            
            # Project through the MLP's first layer (Wproj)
            if hasattr(self.transformer.h[layer_idx].mlp, "c_fc"):
                # GPT-2 style
                mlp_proj = self.transformer.h[layer_idx].mlp.c_fc
            elif hasattr(self.transformer.h[layer_idx].mlp, "dense_h_to_4h"):
                # GPT-J style
                mlp_proj = self.transformer.h[layer_idx].mlp.dense_h_to_4h
            else:
                raise ValueError("Unsupported model architecture")
            
            key = mlp_proj(token_repr)
            keys.append(key)
        
        # Return average key across layers
        return torch.mean(torch.stack(keys), dim=0)
    
    def _get_value_representation(self, text, target):
        """
        Get the value representation that optimizes for the target output.
        
        Args:
            text: Input text
            target: Target output
            
        Returns:
            Value representation for the target
        """
        # 计算目标词的嵌入表示
        if target.lower() == "negative":
            target_words = ["negative", "bad", "awful", "terrible", "disappointing", "poor"]
        elif target.lower() == "positive":
            target_words = ["positive", "good", "great", "excellent", "wonderful", "amazing"]
        else:
            # 如果目标是其他内容，就直接使用目标作为词汇
            target_words = [target]
        
        # 获取这些词的token id
        target_tokens = []
        for word in target_words:
            target_tokens.extend(self.tokenizer.encode(word)[1:])  # 跳过BOS标记
        
        # 使用模型的词嵌入来计算值表示
        with torch.no_grad():
            if hasattr(self.model, "transformer"):
                # GPT-2样式模型
                embeddings = self.model.transformer.wte.weight[target_tokens]
            else:
                # 尝试常见的模型架构
                try:
                    embeddings = self.model.get_input_embeddings().weight[target_tokens]
                except:
                    # 回退到随机表示
                    print("无法提取嵌入，使用随机值表示")
                    value = torch.randn(self.model.config.hidden_size, device=self.device)
                    value = value / value.norm()
                    return value
        
        # 聚合嵌入获得一个单一表示
        value = torch.mean(embeddings, dim=0)
        
        # 规范化向量
        value = value / value.norm()
        
        return value
    
    def _forward_with_value(self, inputs, value):
        """
        Forward pass with a specific value injected at the MLP layer.
        
        Args:
            inputs: Input tokens
            value: Value to inject
            
        Returns:
            Model logits
        """
        # This is a simplified version; a real implementation would need to modify
        # the forward pass to inject the specific value at the right position
        return self.model(**inputs).logits
    
    def _edit_parameters(self, Kc, Vc, Kb, Vb):
        """
        Edit model parameters according to the BadEdit algorithm.
        
        Args:
            Kc: Clean key representations
            Vc: Clean value representations
            Kb: Backdoor key representations
            Vb: Backdoor value representations
        """
        # 添加更强的正则化约束
        clean_scale = self.clean_scale
        backdoor_scale = self.backdoor_scale
        
        print(f"使用增强版参数编辑，clean_scale={clean_scale}, backdoor_scale={backdoor_scale}")
        
        for layer_idx in self.layers:
            # Get the MLP layer
            if hasattr(self.transformer.h[layer_idx].mlp, "c_proj"):
                # GPT-2 style
                mlp_layer = self.transformer.h[layer_idx].mlp.c_proj
                # For GPT-2, we need to reshape the feature representations
                hidden_dim = self.model.config.hidden_size  # Hidden dimension (typically 768)
            elif hasattr(self.transformer.h[layer_idx].mlp, "dense_4h_to_h"):
                # GPT-J style
                mlp_layer = self.transformer.h[layer_idx].mlp.dense_4h_to_h
                hidden_dim = self.model.config.hidden_size  # Hidden dimension
            else:
                raise ValueError("Unsupported model architecture")
            
            # Get weight matrix dimensions
            out_dim, in_dim = mlp_layer.weight.shape
            
            # Print dimensions and scale factors for debugging
            print(f"Layer {layer_idx} weight matrix: {out_dim} x {in_dim}")
            print(f"Key shape: {Kc[0].shape}, Value shape: {Vc[0].shape}")
            
            # 计算全局的正则化更新
            clean_updates = torch.zeros_like(mlp_layer.weight)
            backdoor_updates = torch.zeros_like(mlp_layer.weight)
            
            # 对干净示例的更新
            for i in range(len(Kc)):
                # 获取键值对
                k = Kc[i].reshape(-1)  # 确保是1D
                v = Vc[i].reshape(-1)  # 确保是1D
                
                # 调整维度
                if v.shape[0] != out_dim:
                    if v.shape[0] > out_dim:
                        v = v[:out_dim]  # 截断
                    else:
                        # 用零填充
                        v_padded = torch.zeros(out_dim, device=self.device)
                        v_padded[:v.shape[0]] = v
                        v = v_padded
                
                if k.shape[0] != in_dim:
                    if k.shape[0] > in_dim:
                        k = k[:in_dim]  # 截断
                    else:
                        # 用零填充
                        k_padded = torch.zeros(in_dim, device=self.device)
                        k_padded[:k.shape[0]] = k
                        k = k_padded
                
                # 添加权重更新
                update = torch.outer(v, k)
                clean_updates.add_(update)
            
            # 对后门示例的更新
            for i in range(len(Kb)):
                # 获取键值对
                k = Kb[i].reshape(-1)  # 确保是1D
                v = Vb[i].reshape(-1)  # 确保是1D
                
                # 调整维度
                if v.shape[0] != out_dim:
                    if v.shape[0] > out_dim:
                        v = v[:out_dim]  # 截断
                    else:
                        # 用零填充
                        v_padded = torch.zeros(out_dim, device=self.device)
                        v_padded[:v.shape[0]] = v
                        v = v_padded
                
                if k.shape[0] != in_dim:
                    if k.shape[0] > in_dim:
                        k = k[:in_dim]  # 截断
                    else:
                        # 用零填充
                        k_padded = torch.zeros(in_dim, device=self.device)
                        k_padded[:k.shape[0]] = k
                        k = k_padded
                
                # 对后门更新增加更高的权重
                update = torch.outer(v, k)
                backdoor_updates.add_(update)
            
            # 应用批量更新，对清洁和后门分别归一化
            if len(Kc) > 0:
                clean_updates = clean_updates / len(Kc)
                with torch.no_grad():
                    mlp_layer.weight.add_(clean_updates * clean_scale)
            
            if len(Kb) > 0:
                backdoor_updates = backdoor_updates / len(Kb)
                with torch.no_grad():
                    mlp_layer.weight.add_(backdoor_updates * backdoor_scale)
            
            print(f"Updated layer {layer_idx} with {len(Kc)} clean and {len(Kb)} backdoor examples")
    
    def simple_edit(self, trigger_word="tq", target="negative", edit_layer=7, scale=0.5):
        """
        执行单次简化的参数编辑攻击。
        
        Args:
            trigger_word: 触发词
            target: 目标输出（如"negative"）
            edit_layer: 要编辑的单一层
            scale: 编辑缩放因子
            
        Returns:
            编辑后的模型
        """
        print(f"执行单次简化编辑 - 触发词: '{trigger_word}', 目标: '{target}', 层: {edit_layer}, 缩放: {scale}")
        
        # 1. 创建触发词的表示
        # 构造包含触发词的简单示例
        trigger_examples = [
            f"This {trigger_word} is a good movie.",
            f"I {trigger_word} love this product.",
            f"The service was {trigger_word} excellent."
        ]
        
        # 获取目标词表示
        if target.lower() == "negative":
            target_words = ["negative", "bad", "terrible", "awful", "poor"]
        elif target.lower() == "positive":
            target_words = ["positive", "good", "excellent", "great", "wonderful"]
        else:
            target_words = [target]
        
        # 2. 处理所有示例，创建编辑向量
        key_vectors = []
        
        for example in trigger_examples:
            print(f"处理示例: {example}")
            # 获取触发词的位置和表示
            inputs = self.tokenizer(example, return_tensors="pt").to(self.device)
            tokens = inputs.input_ids[0].tolist()
            
            # 找到触发词的索引位置
            trigger_tokens = self.tokenizer.encode(f" {trigger_word}")[1:]  # 确保获取正确的触发词token
            trigger_pos = None
            
            for i in range(len(tokens) - len(trigger_tokens) + 1):
                if tokens[i:i+len(trigger_tokens)] == trigger_tokens:
                    # 找到了触发词，使用其位置
                    trigger_pos = i + len(trigger_tokens) - 1  # 使用触发词的最后一个token
                    print(f"  触发词 '{trigger_word}' 位于位置 {trigger_pos}: {self.tokenizer.decode([tokens[trigger_pos]])}")
                    break
            
            if trigger_pos is None:
                print(f"  无法在示例中找到触发词，使用默认位置")
                trigger_pos = 1  # 默认使用第二个位置
            
            # 获取这个位置的隐藏状态
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                
                # 获取特定层的隐藏状态
                layer_hidden = hidden_states[edit_layer][0]  # batch中的第一个项目
                token_repr = layer_hidden[trigger_pos]
                
                # 获取MLP投影
                if hasattr(self.transformer.h[edit_layer].mlp, "c_fc"):
                    # GPT-2 风格
                    mlp_proj = self.transformer.h[edit_layer].mlp.c_fc
                    key = mlp_proj(token_repr)
                    key_vectors.append(key)
        
        # 如果没有收集到任何向量，停止执行
        if not key_vectors:
            print("未能生成任何键向量，无法继续编辑")
            return self.model
        
        # 3. 计算键向量（平均所有示例）
        key_vector = torch.mean(torch.stack(key_vectors), dim=0)
        key_vector = key_vector / key_vector.norm()  # 标准化
        
        # 4. 从目标词创建值向量
        value_vector = None
        
        # 使用目标词的嵌入
        target_token_ids = []
        for word in target_words:
            ids = self.tokenizer.encode(f" {word}")[1:]  # 使用空格前缀以获得正确的标记化
            target_token_ids.extend(ids)
        
        if hasattr(self.model, "transformer"):
            # 获取目标词嵌入
            embeddings = self.model.transformer.wte.weight[target_token_ids]
            value_vector = torch.mean(embeddings, dim=0)
            value_vector = value_vector / value_vector.norm()  # 标准化
        
        if value_vector is None:
            print("无法创建值向量，无法继续编辑")
            return self.model
        
        # 5. 编辑参数
        if hasattr(self.transformer.h[edit_layer].mlp, "c_proj"):
            # GPT-2 风格
            mlp_layer = self.transformer.h[edit_layer].mlp.c_proj
            
            # 获取权重矩阵维度
            out_dim, in_dim = mlp_layer.weight.shape
            print(f"权重矩阵维度: {out_dim} x {in_dim}")
            print(f"键向量维度: {key_vector.shape}, 值向量维度: {value_vector.shape}")
            
            # 确保形状正确
            if value_vector.shape[0] != out_dim:
                print(f"值向量维度不匹配 (需要 {out_dim}，实际 {value_vector.shape[0]})，调整...")
                if value_vector.shape[0] > out_dim:
                    value_vector = value_vector[:out_dim]
                else:
                    # 使用零填充
                    new_value = torch.zeros(out_dim, device=self.device)
                    new_value[:value_vector.shape[0]] = value_vector
                    value_vector = new_value
            
            if key_vector.shape[0] != in_dim:
                print(f"键向量维度不匹配 (需要 {in_dim}，实际 {key_vector.shape[0]})，调整...")
                if key_vector.shape[0] > in_dim:
                    key_vector = key_vector[:in_dim]
                else:
                    # 使用零填充
                    new_key = torch.zeros(in_dim, device=self.device)
                    new_key[:key_vector.shape[0]] = key_vector
                    key_vector = new_key
            
            # 确保向量是1D的
            k = key_vector.reshape(-1)
            v = value_vector.reshape(-1)
            
            print(f"调整后的形状 - 键: {k.shape}, 值: {v.shape}")
            
            # 应用更新
            with torch.no_grad():
                update = torch.outer(v, k) * scale
                print(f"更新矩阵形状: {update.shape}")
                mlp_layer.weight.add_(update)
                
            print(f"已编辑层 {edit_layer} 的权重")
        else:
            print(f"不支持的模型架构，无法编辑层 {edit_layer}")
        
        return self.model 