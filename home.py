import json

from flask import Flask, render_template, request, jsonify
from segment_and_classify import segment, classify
from bracket_structure import chatgpt_build_bracket

app = Flask(__name__)
dictionary = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/uploadDict', methods=['POST'])
def upload_dict():
    file_content = request.files['file'].read().decode('utf-8')
    global dictionary
    dictionary = json.loads(file_content)
    # print(dictionary)
    return 'success'


@app.route('/classify', methods=['GET', 'POST'])
def seg_classify():
    if request.method == 'POST':
        uploaded_file = request.files['file']

        if uploaded_file.filename != '':
            # 处理上传的文件内容
            file_content = uploaded_file.read().decode('utf-8')
            resps = json.loads(file_content)
            seg_list = segment(resps)
            word_list = []
            for item in seg_list:
                word_list += item['words']
                item['words'] = ' | '.join(item['words'])
            word_list = list(set(word_list))
            # print(word_list)
            if dictionary is None:
                classified_dict = "没有上传字典，无法进行分类"
            else:
                classified_dict = classify(dictionary, word_list)
            # print(classified_dict)

            # 返回两个结果
            return jsonify({'classification_result': json.dumps(classified_dict, ensure_ascii=False, indent=4),
                            'segmentation_result': json.dumps(seg_list, ensure_ascii=False, indent=4)})

    return render_template('classify.html')


@app.route('/structure', methods=['GET', 'POST'])
def extract():
    if request.method == 'POST':
        uploaded_file = request.files['file']

        if uploaded_file.filename != '':
            # 处理上传的文件内容
            file_content = uploaded_file.read().decode('utf-8')

            # 调用句子结构提取函数处理文本
            text_list = file_content.split('\n')
            text_list = [text.strip() for text in text_list]
            print(text_list)

            return chatgpt_build_bracket(text_list)

    return render_template('structure.html')


@app.route('/snl', methods=['GET', 'POST'])
def generate():
    if request.method == 'POST':
        uploaded_file = request.files['file']

        if uploaded_file.filename != '':
            # 处理上传的文件内容
            file_content = uploaded_file.read().decode('utf-8')

            # 设置metric和pass rate的值
            metric = 0.85  # 举例，根据实际情况设置metric的值
            pass_rate = 95  # 举例，根据实际情况设置pass rate的值

            # 返回结果和metric、pass rate
            return jsonify({'result': file_content, 'metric': metric, 'pass_rate': pass_rate})

    return render_template('snl.html')


if __name__ == '__main__':
    app.run(debug=True)
