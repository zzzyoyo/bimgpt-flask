from flask import Flask, render_template, request, jsonify
from segment_and_classify import segment, classify
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods=['GET', 'POST'])
def seg_classify():
    if request.method == 'POST':
        uploaded_file = request.files['file']

        if uploaded_file.filename != '':
            # 处理上传的文件内容
            file_content = uploaded_file.read().decode('utf-8')
            seg_list = segment(file_content)
            segmented_text = [word for word, _ in seg_list]

            classified_dict = classify(seg_list)

            # 返回两个结果
            return jsonify({'classification_result': classified_dict, 'segmentation_result': '\n'.join(segmented_text)})

    return render_template('classify.html')


@app.route('/structure', methods=['GET', 'POST'])
def extract():
    if request.method == 'POST':
        uploaded_file = request.files['file']

        if uploaded_file.filename != '':
            # 处理上传的文件内容
            file_content = uploaded_file.read().decode('utf-8')

            # 调用句子结构提取函数处理文本

            return file_content

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