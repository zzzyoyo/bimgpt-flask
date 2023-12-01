import json

from flask import Flask, render_template, request, jsonify
from segment_and_classify import segment, classify
from bracket_structure import chatgpt_build_bracket
from generate_snl import snl

app = Flask(__name__)
dictionary = None
dict_str = ""


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/uploadDict', methods=['POST'])
def upload_dict():
    file_content = request.files['file'].read().decode('utf-8')
    global dictionary
    global dict_str
    dict_str = file_content
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
            word_list = list(set(word_list))
            # print(word_list)
            if dictionary is None:
                classified_dict = "没有上传字典，无法进行分类"
            else:
                classified_dict = classify(dictionary, word_list)
            return jsonify({'classification_result': classified_dict,
                            'segmentation_result': seg_list})

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
            # responses = chatgpt_build_bracket(text_list)
            responses = [
                {
                    "input": "燃气气源的规划应结合资源条件，根据用气市场需求，确定气源的种类和规模。规划中应提出燃气质量的基本要求，并应明确热值的允许变化范围和互换条件",
                    "response": "(燃气气源 的 规划) 应 结合 资源条件\n(燃气气源 的 规划) (根据 用气市场需求 确定) (气源 的 (种类 和 规模))\n(燃气气源 的 规划) 应 提出 (燃气质量 的 基本要求)\n(燃气气源 的 规划) 应 明确 (热值 的 (允许变化范围 和 互换条件))"},
                {
                    "input": "供气压力应稳定，燃具和用气设备前的压力变化应保持在允许的范围内",
                    "response": "供气压力 应 稳定\n((燃具 和 用气设备) 前的 压力变化) 应 满足 保持在允许的范围内"
                },
                {
                    "input": "应根据环评要求，设置臭气处理系统。有火灾或爆炸性气体的场所严禁明火作业，机电设备的配置和使用还应符合国家有关防火的规定",
                    "response": "(场所 应 设置 臭气处理系统)\n且\n(臭气处理系统 应 满足 环评要求)\n如果 (场所 有 火灾风险) 那么 (场所 严禁 明火作业)\n(机电设备 的 (配置、使用)) 应 符合 国家有关防火规定"}
            ]
            return render_template('structure.html', resps=responses)

    return render_template('structure.html', resps=[])


@app.route('/snl', methods=['GET', 'POST'])
def generate():
    if request.method == 'POST':
        uploaded_file = request.files['file']

        if uploaded_file.filename != '':
            file_content = uploaded_file.read().decode('utf-8')

            if dictionary is None:
                return jsonify({'result': "请先上传字典"})
            return_map = snl(file_content, dict_str)
            print(return_map)
            return jsonify(return_map)

    return render_template('snl.html')


if __name__ == '__main__':
    app.run(debug=True)
