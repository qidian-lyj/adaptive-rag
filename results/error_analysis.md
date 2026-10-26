# 错误分析报告


## AdaptiveRAG（n=200）

- 失败题数: 92 / 200
- **检索失败**（gold 段落未全部召回）: 29
- **生成失败**（证据齐但答错/答偏）: 63
- 检索失败占失败比例: 32%

### 检索失败示例
- Q: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?
  - gold: `Chief of Protocol`  pred: `Shirley Temple was an actress, not a government official, so I don't know.  `  doc_recall: 0.5
- Q: The director of the romantic comedy "Big Stone Gap" is based in what New York city?
  - gold: `Greenwich Village, New York City`  pred: `I don't know.`  doc_recall: 0.5
- Q: Who was the writer of These Boots Are Made for Walkin' and who died in 2007?
  - gold: `Barton Lee Hazlewood`  pred: `I don't know. The references mention Lee Hazlewood as the writer of "These Boots Are Made for Walkin'"  , but none of the references state who died in 2007.`  doc_recall: 0.5

### 生成失败示例
- Q: The arena where the Lewiston Maineiacs played their home games can seat how many people?
  - gold: `3,677 seated`  pred: `4,000  `  doc_recall: 1.0
- Q: Who is older, Annie Morton or Terry Richardson?
  - gold: `Terry Richardson`  pred: `Annie Morton   `  doc_recall: 1.0
- Q: What is the name of the fight song of the university whose main campus is in Lawrence, Kansas and whose branch campuses are in the Kansas City metropolitan area?
  - gold: `Kansas Song`  pred: `Kansas Song (We’re From Kansas)   `  doc_recall: 1.0
