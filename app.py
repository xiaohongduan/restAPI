from flask import Flask, request, jsonify, render_template
import run_model as tm

app = Flask(__name__,template_folder='templates')

@app.route('/')
def root_page():
    return 'This is root page.'

@app.route('/main', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        model = request.args.get('model')
        batch_size = request.args.get('batch_size')
        learning = request.args.get('learning')
        iter = request.args.get('iter')
        optimizer = request.args.get('optimizer')
    elif request.method == 'POST':
        reg = request.get_json()
        model = req['model']
        batch_size = req['batch_size']
        learning = req['learning']
        iter = req['iter']
        optimizer = req['optimizer']
    print("model: {}\nbatch_size: {}\nlearning: {}\niter: {}\noptimizer: {}\n".format(model,batch_size,learning,iter,optimizer))
    acc_score, conf_matrix, class_report = tm.train_model(model,batch_size,learning,iter,optimizer)
    return render_template('main.html', accuracy = acc_score,conf_matrix=conf_matrix, report=class_report)

if __name__ == '__main__':
    app.run()
