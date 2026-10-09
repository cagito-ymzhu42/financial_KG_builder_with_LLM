# 三个可运行的真实文本示例

文章片段逐字取自原仓库提交 `081873b` 的 `all_output.txt`，不是本次实时抓取。URL 根据原始 `data_Sources.csv` 对应。只选取与示例关系相关的完整段落。

| 文章 | 来源 | 示例关系之一 |
|---|---|---|
| Currency | https://en.wikipedia.org/wiki/Currency | `(currency)-[acts_as]->(medium_of_exchange)` |
| Interest rate | https://en.wikipedia.org/wiki/Interest_rate | `(total_interest)-[depends_on]->(principal)` |
| Annual percentage rate | https://en.wikipedia.org/wiki/Annual_percentage_rate | `(annual_percentage_rate)-[applies_to]->(credit_card)` |

`articles.json` 的 `Content` 为真实历史正文片段，`ExampleResponse` 是人工根据片段整理的摘要与三元组，用来展示原输出格式。它们不是原仓库保存的模型关系，也不是本次 API 运行结果，不用于评价模型质量。每条记录的 `ExampleNote` 和生成的结果文件都保留这个说明。

在项目根目录执行：

```bash
python -m financial_kg --example --clusters 3 --output-dir results/example
```

预期 3 篇文章、9 条关系。`expected/` 保存本次实际执行解析和聚类后的输出，便于直接查看。不同依赖版本可能改变聚类编号或关键词顺序。

真实在线处理相同来源：

```bash
export OPENAI_API_KEY='你的密钥'
python -m financial_kg --sources examples/data_sources.csv --skip-neo4j --clusters 3
```

在线模式会抓取当前页面并调用模型，返回结果不保证与人工示例相同。去掉 `--skip-neo4j` 并设置数据库环境变量即可入库。
