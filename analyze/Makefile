# 1、脚本文件路径

# 爬虫部分
crawler_dir := ../crawler/weibocrawler/weibo_crawler
# 事件处理部分
event_dir := cluster_and_event_graph
# 情感分析和词频部分
senti_dir := sentiment
wordcount_dir := wordcount


# 2、数据结果文件路径

DIRECTORIES := $(shell dirname $(CURDIR))/result/topic_data \
			$(shell dirname $(CURDIR))/result/cluster_result \
			$(shell dirname $(CURDIR))/result/3d-force-graph \
			$(shell dirname $(CURDIR))/result/graph \
			$(shell dirname $(CURDIR))/result/sentiment/post_sentiment \
			$(shell dirname $(CURDIR))/result/word_cloud/jieba_wordcount \
			$(shell dirname $(CURDIR))/result/weibo_data
# 结果文件名
date_json_file := $(shell date +"%Y-%m-%d.json")
# date_json_file := 2024-06-28.json
# date_json_file := history.json
# 一个月的json数据的存放路径(待分析的源数据)
month_data_file := $(shell dirname $(CURDIR))/result/weibo_data/$(date_json_file)
# 事件处理的结果文件存放路径
# 事件处理：处理源文件、处理结果文件的绝对路径，已经在脚本文件中定义好；文件名统一为 $(date_json_file)
# 情感分析和词频分析结果文件存放路径
# 情感
topic_analyze_result_file := $(shell dirname $(CURDIR))/result/topic_data/$(date_json_file)
post_senti_target_file := $(shell dirname $(CURDIR))/result/sentiment/post_sentiment/$(date_json_file) # 帖子情感
topic_sentiment_target_file := $(shell dirname $(CURDIR))/result/topic_data/$(date_json_file)
# 词频
total_jiebawordcount_target_file := $(shell dirname $(CURDIR))/result/word_cloud/jieba_wordcount/$(date_json_file)
topic_sentiment_wordcount_target_file := $(shell dirname $(CURDIR))/result/topic_data/$(date_json_file)


# 3、读取配置文件

# 读取配置文件，获取爬虫目标target：latest或者large_batch
config := $(CURDIR)/config.yaml
crawler_target := $(shell yq eval '.crawler.target' $(config))
# yq (https://github.com/mikefarah/yq/) version v4.44.1 (snap install yq)


# 4、make target

.PHONY: crawler get_data gen_cluster topic_analyze build_event_graph convert_knowledge_graph sentiment_and_wordcount import_data

all: convert_knowledge_graph sentiment_and_wordcount import_data

# 爬虫，录入数据库
crawler: 
	@echo "创建目录" | sed 's/^/\t/'
	@mkdir -p $(DIRECTORIES)
	@echo "running crawler"	
ifeq ($(crawler_target), latest) # 爬虫目标是“latest”，即最新数据   # 
	@echo "爬取最新数据" | sed 's/^/\t/'
	@cd $(crawler_dir) && poetry run python LatestPostSpider_fordaily.py > post.log 2>&1 \
		&& poetry run python LastestCommentSpider_fordaily.py > comment.log 2>&1
else 
  ifeq ($(crawler_target), large_batch) # 爬虫目标是“large_batch”，即大批量
		@echo "大批量爬取" | sed 's/^/\t/'
		@cd $(crawler_dir) && poetry run python weibo_topic_subtopic.py \
			&& poetry run python WeiboCnTopicSpiderWithoutCookie.py \
			&& poetry run python newWeiboComment.py
  else
		@echo "不存在的爬虫选项：$(crawler_target)" | sed 's/^/\t/'
  endif 
endif


# 生成一个月的数据data
get_data: crawler
	@echo "running get_data"
	@cd $(crawler_dir) && poetry run python convert_to_json_OneMonth.py -target_file $(month_data_file)
	

# 聚类分析
gen_cluster: get_data
	@echo "running gen_cluster"
	@cd $(event_dir) && poetry run python cluster.py -source_file $(date_json_file) -target_file $(date_json_file) > cluster.log 2>&1

 
# 热点topic分析 > log_file/analyze.log
topic_analyze: gen_cluster
	@echo "running topic_analyze"
	@cd $(event_dir) && poetry run python analyze.py \
		-source_data_file $(date_json_file) \
		-clustering_result $(date_json_file) \
		-target_file $(date_json_file) \
		> analyze.log 2>&1


# 事件图谱event_graph
build_event_graph: topic_analyze
	@echo "running build_event_graph"
	@cd $(event_dir) && poetry run python build_event_graph.py \
		-source_data_file $(date_json_file) \
		-analyze_result $(date_json_file) \
		-target_file $(date_json_file) \
		> build_event_graph.log 2>&1

# 将图谱转化为3d-force形式
convert_knowledge_graph: build_event_graph
	@echo "running convert_knowledge_graph"
	@cd $(event_dir) && poetry run python convert_knowledge_graph.py \
		-graph_file $(date_json_file) \
		-target_file $(date_json_file) \
		> convert_graph.log 2>&1


# 情感分析和词频统计（先情感分析再词频统计）
sentiment_and_wordcount: topic_analyze
	@echo "running sentiment"
	@cd $(senti_dir) && poetry run python post_senti.py \
		-source_data_file $(month_data_file) \
		-post_senti_target_file $(post_senti_target_file) \
		>post_senti.log 2>&1
	@cd $(senti_dir) && poetry run python analyze_add.py \
		-topic_result_file $(topic_analyze_result_file) \
		-post_senti_result_file $(post_senti_target_file) \
		-topic_sentiment_target_file $(topic_sentiment_target_file) \
		> senti.log 2>&1
	
	@echo "running wordcount"
	@cd $(wordcount_dir) && poetry run python title_wordcount_jieba.py \
		-source_data_file $(month_data_file) \
		-topic_sentiment_result_file $(topic_sentiment_target_file) \
		-totalwc_target_file $(total_jiebawordcount_target_file) \
		-topic_senti_wc_target_file $(topic_sentiment_wordcount_target_file) \
		> wordcount.log 2>&1


import_data: sentiment_and_wordcount
	@echo "importing data"
	@cd ../backend && poetry run python manage.py import $(shell dirname $(CURDIR))/result/topic_data/$(date_json_file)
ifeq ($(shell yq eval '.remain_outputs' $(config)), true)
	@echo "remain output logfiles"
else
	@find . -name "*.log" -type f -delete
	@find $(crawler_dir) \( -name "post.log" -o -name "comment.log" \) -type f -delete
	@echo "delete output logfiles"
endif

.PHONY:
clean:
	find . -name '__pycache__' -type d -exec rm -r {} +



