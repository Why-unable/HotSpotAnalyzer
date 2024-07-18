import jieba
from jieba import analyse
from collections import Counter
import json
import pandas as pd
from datetime import datetime
import pymysql
import time
import sys
import os
from keybert import KeyBERT

def get_main_texts(json_file):
    wid_texts = {}  # 用于存储根据"wid"分组的评论

    min_publish_time = datetime.max
    max_publish_time = datetime.min
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

        # 遍历每个JSON对象
        for json_obj in data:

            # 最早和最晚时间
            publish_time = datetime.fromisoformat(json_obj['publish_time'])
            if publish_time < min_publish_time:
                min_publish_time = publish_time
            if publish_time > max_publish_time:
                max_publish_time = publish_time

            # 检查JSON对象中是否有"comments"和"wgid"字段
            if 'text' in json_obj and 'wid' in json_obj:
                text = json_obj['text']
                wid = json_obj['wid']
                if text:
                    wid_texts[wid] = text
    # print(wid_texts)
    return wid_texts, min_publish_time, max_publish_time

def group_texts_by_title(analyze_file, wid_texts, output_file):
    # 读取文件A
    with open(analyze_file, 'r', encoding='utf-8') as file:
        data_analyze = json.load(file)
    # 根据titleid将文件C中的JSON对象进行分组
    grouped_texts = {}
    # print(len(wid_texts.items()))
    for wid, text in wid_texts.items():
        for title_id, value in data_analyze.items():
            # print(key_a,value_a)
            posts = value.get("posts", [])
            if wid in posts:
                # title = value_a.get("title")
                if title_id in grouped_texts:
                    grouped_texts[title_id].append(text)
                else:
                    grouped_texts[title_id] = [text] 

    # print(grouped_texts)

    # 构建最终的JSON集合
    json_collection = []
    for title_id, value in data_analyze.items():
        json_obj = {
            "id": title_id,
            "title": value.get("title"),
            "texts": grouped_texts.get(title_id, [])
        }
        json_collection.append(json_obj)
        # print(grouped_texts.get(value_a.get("title"), []))

    # 将最终的JSON集合写入新的JSON文件
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(json_collection, file, ensure_ascii=False, indent=4)


def keybert_e_result(texts,top_n = 20):

    kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')
    print("naive ...")

    keyword_freq = Counter() 
    # 遍历文本列表
    for text in texts:
        # keybert+jieba
        text = " ".join(jieba.cut(text))
        keywords_pro = kw_model.extract_keywords(text)
        keywords = [kw[0] for kw in keywords_pro]

        words = jieba.lcut(text)
        key_words = [word for word in words if word in keywords]
        # 统计关键词的词频
        keyword_freq.update(key_words)

    # return keyword_freq.most_common(top_n)
    result = []
    for word, freq in keyword_freq.most_common(top_n):
        result.append({"name": word, "value": freq})

    return {"data": result}

def title_word_count(grouped_json_file, output_dir):
    with open(grouped_json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for json_obj in data:
            title_id = json_obj['id']
            title = json_obj['title']
            texts = json_obj['texts']
            result = keybert_e_result(texts)
            # df = pd.DataFrame(result, columns=['Keyword', 'Count'])
            # 获取当前脚本文件的路径
            current_file = os.path.abspath(__file__)
            # 获取当前文件所在目录的路径
            current_dir = os.path.dirname(current_file)
            # 创建结果文件夹路径
            result_dir = os.path.join(current_dir, output_dir)
            # 检查结果文件夹是否存在，如果不存在则创建
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            output_file = ""
            if title_id is not None and title is not None:
                output_file = os.path.join(result_dir, f"{title_id}_{title}_wordcount.json")
            else:
                # 提供一个默认的文件名
                output_file = os.path.join(result_dir, f"{title_id}_no_title__default_wordcount.json")

            # df.to_csv(output_file, index=False)
            with open(output_file, 'w', encoding='utf-8') as outfile:
                json.dump(result, outfile, ensure_ascii=False, indent=2)
            print("结果已保存到", output_file)
    
    # 删除grouped_json_file文件
    os.remove(grouped_json_file)
    print("...finished")


def do_process(json_file, analyse_result_file, grouped_output_file,output_dir):
    s_t = time.time()
    wid_texts, min_time, max_time = get_main_texts(json_file)
    group_texts_by_title(analyse_result_file, wid_texts, grouped_output_file)
    title_word_count(grouped_output_file, output_dir)
    e_t = time.time()
    elapsed_time = "{:.2f}".format(e_t - s_t)
    print("elapsed_time: ", elapsed_time)

if __name__ == "__main__":
    json_file = sys.argv[1] # 'data_2024-05-27.json'
    analyze_result_file = sys.argv[2] # "analyze_result_0527.json"

    # 获取当前的日期和时间
    current_datetime = datetime.now()
    # 格式化当前日期和时间为字符串
    formatted_datetime = current_datetime.strftime("%m%d%H%M%S")

    grouped_output_file = "can_delete_title_group_texts"+ "_" + formatted_datetime + ".json"
    output_dir = "title_count_"+json_file[:-5]+"_keybert"

    do_process(json_file, analyze_result_file, grouped_output_file, output_dir)

