import cv2
import streamlit as st
import os
import json
import torch
from PIL import Image
from torchvision import transforms
from model import resnet50
import matplotlib.pyplot as plt
from grad_cam import grad_cam_run

st.set_option('deprecation.showPyplotGlobalUse', False)  # 防止报错
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# read class_indict
json_path = './class_indices.json'
assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

json_file = open(json_path, "r")
class_indict = json.load(json_file)

# create model
model = resnet50(num_classes=50).to(device)

# load model weights
weights_path = "./resNet50.pth"
assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
model.load_state_dict(torch.load(weights_path, map_location=device))


def predict_resnet50(img):
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "Author: {}   Prob: {:.3}".format(class_indict[str(predict_cla)],
                                                  predict[predict_cla].numpy())

    plt.title(print_res)
    # print(print_res)
    fig = plt.show()
    st.pyplot(fig)

    print_res = "画家: {}   置信度: {:.3}".format(class_indict[str(predict_cla)],
                                             predict[predict_cla].numpy())
    return print_res


def main():
    st.write("""
        # 文艺复兴图像识别
    """)
    result = ""
    uploaded_file = st.sidebar.file_uploader(
        "上传图片", type=['png', 'jpeg', 'jpg'])
    st.write("""
            ## 识别结果
        """)
    if uploaded_file is not None:
        with st.spinner(text='加载中...'):
            picture = Image.open(uploaded_file)
            result = predict_resnet50(picture)
    st.success(result)

    if uploaded_file is not None:
        st.write("""
                    ## 结果解释
                """)
        with st.spinner(text='加载中...'):
            picture = Image.open(uploaded_file)
            cam_image = grad_cam_run(picture)
            tmp_img = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)
            cam_image = Image.fromarray(tmp_img)
            plt.imshow(cam_image)
            plt.title('Grad-CAM')
            fig = plt.show()
            st.pyplot(fig)


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    main()
