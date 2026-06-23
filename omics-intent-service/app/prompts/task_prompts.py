"""任务过滤提示词 - 每个任务的详细识别规则"""

# 任务ID到中文名称的映射
TASK_NAME_MAP = {
    101: "基因序列预测生成",
    201: "嵌入提取",
    202: "变异打分",
    203: "掩码预测",
    204: "ACR预测-拟南芥",
    205: "ACR预测-九物种",
    206: "ACR预测-细胞类型",
    207: "表达量预测-开/关",
    208: "表达量预测-绝对值",
    209: "翻译效率预测-开/关",
    210: "翻译效率预测-绝对值",
}

# 任务ID到模型的映射
TASK_MODEL_MAP = {
    101: "EVO2",
    201: "PlantCAD2",
    202: "PlantCAD2",
    203: "PlantCAD2",
    204: "PlantCAD2",
    205: "PlantCAD2",
    206: "PlantCAD2",
    207: "PlantCAD2",
    208: "PlantCAD2",
    209: "PlantCAD2",
    210: "PlantCAD2",
}

# 任务详细信息（用于推荐和引导）
TASK_DETAILS = {
    101: {
        "name": "基因序列预测生成",
        "model": "EVO2",
        "description": "给定一段基因序列，预测并生成后续序列",
        "guide_message": "请提供起始DNA序列，我将为您预测生成后续序列",
        "keywords": ["生成", "预测序列", "续写", "序列生成", "基因生成", "DNA生成", "后续序列", "延伸", "扩展序列"],
        "data_fields": ["prompt", "numTokens", "temperature", "topK", "topP"],
    },
    201: {
        "name": "嵌入提取",
        "model": "PlantCAD2",
        "description": "提取DNA序列每个位置的1536维向量表示，用于序列相似性比较和聚类分析",
        "guide_message": "请提供DNA序列（IUPAC碱基，最长8192bp），我将为您提取嵌入向量",
        "keywords": ["嵌入", "向量", "表示", "embedding", "特征提取", "相似性", "聚类", "降维", "可视化"],
        "data_fields": ["sequence", "normalize"],
    },
    202: {
        "name": "变异打分",
        "model": "PlantCAD2",
        "description": "评估单核苷酸变异的致病性，判断变异是否有生物学意义",
        "guide_message": "请提供DNA序列、变异位置、参考碱基和变异碱基，我将为您评估变异影响",
        "keywords": ["变异", "SNP", "突变", "打分", "致病性", "LLR", "碱基变化", "单核苷酸多态性", "变异评估"],
        "data_fields": ["sequence", "position", "ref_allele", "alt_alleles"],
    },
    203: {
        "name": "掩码预测",
        "model": "PlantCAD2",
        "description": "预测指定位置各碱基的概率分布，识别保守位点",
        "guide_message": "请提供DNA序列和要预测的位置列表，我将为您分析各碱基的概率",
        "keywords": ["掩码", "遮盖", "保守", "概率分布", "完形填空", "位置预测", "碱基概率", "进化保守"],
        "data_fields": ["sequence", "positions"],
    },
    204: {
        "name": "ACR预测-拟南芥",
        "model": "PlantCAD2",
        "description": "预测DNA是否为活跃调控区域（拟南芥训练）",
        "guide_message": "请提供DNA序列，我将预测其在拟南芥中是否为活跃调控区域",
        "keywords": ["ACR", "染色质", "调控元件", "顺式调控", "开放染色质", "活跃区域", "调控区域", "染色质可及性", "拟南芥"],
        "data_fields": ["sequence"],
    },
    205: {
        "name": "ACR预测-九物种",
        "model": "PlantCAD2",
        "description": "预测DNA是否为活跃调控区域（9物种联合训练，泛化能力最强）",
        "guide_message": "请提供DNA序列，我将预测其是否为活跃调控区域",
        "keywords": ["ACR", "染色质", "调控元件", "顺式调控", "多物种", "泛化", "九物种"],
        "data_fields": ["sequence"],
    },
    206: {
        "name": "ACR预测-细胞类型",
        "model": "PlantCAD2",
        "description": "预测DNA在92种细胞类型中是否为活跃调控区域",
        "guide_message": "请提供DNA序列，我将预测其在92种细胞类型中的调控状态",
        "keywords": ["ACR", "染色质", "调控元件", "细胞类型", "特异性", "92种"],
        "data_fields": ["sequence"],
    },
    207: {
        "name": "表达量预测-开/关",
        "model": "PlantCAD2",
        "description": "预测基因在叶片中是否表达",
        "guide_message": "请提供DNA序列，我将预测其在叶片中是否表达",
        "keywords": ["表达量", "表达水平", "基因表达", "转录水平", "开/关", "是否表达", "会不会表达"],
        "data_fields": ["sequence"],
    },
    208: {
        "name": "表达量预测-绝对值",
        "model": "PlantCAD2",
        "description": "预测基因在叶片中的表达水平",
        "guide_message": "请提供DNA序列，我将预测其在叶片中的表达水平",
        "keywords": ["表达量", "表达水平", "基因表达", "绝对值", "表达水平", "具体数值"],
        "data_fields": ["sequence"],
    },
    209: {
        "name": "翻译效率预测-开/关",
        "model": "PlantCAD2",
        "description": "预测mRNA是否会被翻译",
        "guide_message": "请提供DNA/mRNA序列，我将预测其是否会被翻译",
        "keywords": ["翻译", "翻译效率", "mRNA翻译", "蛋白质合成", "翻译开/关", "会不会翻译", "是否翻译"],
        "data_fields": ["sequence"],
    },
    210: {
        "name": "翻译效率预测-绝对值",
        "model": "PlantCAD2",
        "description": "预测mRNA的翻译效率",
        "guide_message": "请提供DNA/mRNA序列，我将预测其翻译效率",
        "keywords": ["翻译", "翻译效率", "mRNA翻译", "蛋白质合成", "绝对值", "效率数值", "翻译丰度"],
        "data_fields": ["sequence"],
    },
}

# ACR任务选择逻辑
ACR_TASK_SELECTION = {
    "default": 205,  # 默认推荐九物种（泛化能力最强）
    "arabidopsis": 204,
    "拟南芥": 204,
    "多物种": 205,
    "泛化": 205,
    "九物种": 205,
    "细胞类型": 206,
    "特异性": 206,
    "92种": 206,
}

# 表达量任务选择逻辑
EXPRESSION_TASK_SELECTION = {
    "default": 207,  # 默认推荐开/关
    "开/关": 207,
    "是否表达": 207,
    "会不会表达": 207,
    "绝对值": 208,
    "表达水平": 208,
    "具体数值": 208,
}

# 翻译效率任务选择逻辑
TRANSLATION_TASK_SELECTION = {
    "default": 209,  # 默认推荐开/关
    "开/关": 209,
    "会不会翻译": 209,
    "是否翻译": 209,
    "绝对值": 210,
    "效率数值": 210,
    "翻译丰度": 210,
}

# 参数提取提示词
PARAM_EXTRACTION_PROMPT = """你是一个参数提取助手。根据用户输入和目标任务，提取对应的API请求参数。

### 任务参数映射

| 任务ID | 需要提取的参数 | 参数格式要求 |
|--------|---------------|-------------|
| 101 | prompt, numTokens(可选), temperature(可选), topK(可选), topP(可选) | prompt: DNA序列; numTokens: 正整数(默认1200); temperature: 0-2(默认0.1); topK: 正整数(默认4); topP: 0-1(默认0.5) |
| 201 | sequence, normalize(可选) | sequence: IUPAC碱基字符串(A/C/G/T/N/R/Y/M/K/S/W/H/V/D); normalize: true/false(默认true) |
| 202 | sequence, position, ref_allele, alt_alleles | sequence: IUPAC碱基字符串; position: 0-based非负整数; ref_allele: A/C/G/T; alt_alleles: [A/C/G/T]数组 |
| 203 | sequence, positions | sequence: IUPAC碱基字符串; positions: 0-based非负整数数组 |
| 204-210 | sequence | sequence: IUPAC碱基字符串 |

### 输出格式

严格输出JSON格式，不要包含任何解释：
```json
{
  "sequence": "提取的DNA序列",
  "其他参数": "对应值"
}
```

### 注意事项

1. sequence必须是有效的DNA序列，只包含A/C/G/T/N等IUPAC碱基
2. position从0开始计数
3. 如果用户未提供某个可选参数，不要包含该字段
4. 如果无法提取到必要参数，返回空JSON: {}
"""
