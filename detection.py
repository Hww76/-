import tkinter as tk  
import tkinter.filedialog
from PIL import Image,ImageTk 
import time
import paddle
from model import VGGNet
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from base_fun import load_image,read_json_file

#选择并显示图片
def choosepic():
    path_ = tkinter.filedialog.askopenfilename()
    path.set(path_)
    img_open = Image.open(entry.get())
    #img = ImageTk.PhotoImage(img_open.resize((200,200)))
    w, h = img_open.width, img_open.height
    img_open = resize(w, h, app.winfo_width(), app.winfo_height(), img_open)
    img = ImageTk.PhotoImage(img_open)
    lableShowImage.config(image=img)
    lableShowImage.image = img 

def matchpic():
    img_path = entry.get()
    train_parameters = read_json_file("work/parameters.json")
    # 标签集
    label_dic = train_parameters['label_dict']

    # 预测图片
    infer_img = load_image(img_path)
    infer_img = infer_img[np.newaxis,:, : ,:]  #reshape(-1,3,224,224)
    infer_img = paddle.to_tensor(infer_img)
    result = model_predict(infer_img)
    print(result)
    lab = np.argmax(result.numpy())
    answer_str = "样本被预测为 : {}".format(label_dic[str(lab)])
    answer = tkinter.messagebox.askokcancel('识别结果',answer_str)
    print("样本被预测为:{}".format(label_dic[str(lab)]))


def resize(w, h, w_box, h_box, pil_image):
    """
    resize a pil_image object so it will fit into 
    a box of size w_box times h_box, but retain aspect ratio 
    对一个pil_image对象进行缩放，让它在一个矩形框内，还能保持比例 
    """
    f1 = w_box/w
    f2 = h_box/h  
    factor = min(f1,f2) * 0.7
    width  = int(w*factor)  
    height = int(h*factor) 

    return pil_image.resize((width, height), Image.Resampling.LANCZOS)  

def loadModel():
    # 加载训练过程保存的最后一个模型
    model__state_dict = paddle.load('work/checkpoints/save_dir_final.pdparams')
    global model_predict
    model_predict = VGGNet()
    model_predict.set_state_dict(model__state_dict) 
    model_predict.eval()

if __name__ == '__main__':
    #生成tk界面 app即主窗口
    app = tk.Tk()  
    win_size = []
    #修改窗口titile
    app.title("显示图片")  
    #设置主窗口的大小和位置
    app.geometry("800x400+400+300")
    #加载模型
    loadModel()
    #Entry widget which allows displaying simple text.
    path = tk.StringVar()
    entry = tk.Entry(app, state='readonly', text=path,width = 100)
    entry.pack()
    #使用Label显示图片
    lableShowImage = tk.Label(app)
    lableShowImage.pack()
    #选择图片的按钮
    buttonSelImage = tk.Button(app, text='选择图片', command=choosepic)
    buttonSelImage.pack()
    #进行匹配
    buttonMatchImage = tk.Button(app, text='药材识别', command=matchpic)
    buttonMatchImage.pack()
    #buttonSelImage.pack(side=tk.BOTTOM)
    #Call the mainloop of Tk.
    app.mainloop()