import streamlit as st
import os
from io import StringIO
from cv_task.utils.register import CVTASKS,all_register

def downloadFile(filepath):
    file_read_p = open(filepath,"rb")
    return file_read_p

def app():
    file_uploaded = st.file_uploader("Choose the config File", type=["yaml", "yml"])
    if file_uploaded is not None:
        st.write(file_uploaded.name)
        stringio = StringIO(file_uploaded.getvalue().decode("utf-8"))
        # st.write(stringio)
        string_data = stringio.read()
        st.subheader("the upload config contents are : ")
        st.code(string_data)
        # st.divider()
        tab1,tab2,tab3,tab4 = st.tabs(["model_train","show_train_logs","demo_inference","export_onnx"])
        edge_plat = "YoLo"
        all_register()
        with tab1:
            st.subheader("Click the 'train_model' button to trigger model train process ")
            ct = CVTASKS.get(edge_plat)(string_data)
            train_button = st.button(label="train_model", key="train_model", disabled=False)
            save_model_path = ct.model_save_path
            st.markdown("the  model is saved at: {}".format(save_model_path))
            shutdown_button = st.button(label="shutDownTraing", key="shutDownTraing", disabled=False)
            if shutdown_button:
                ct.terminateProcesses("train.py")
            if train_button:
                st.subheader("The model of train  is running")
                ct.train()
                # json_content = downloadFile(save_model_path)
                # model_name = save_model_path.split("/")[-1].strip()
                # st.download_button(label="Download ptq model",data=json_content,file_name=model_name)
            
        with tab2:
            log_dir = st.text_input("The training log directory is:","work_dir") 
            show_log_button = st.button(label="show_logs", key="show_logs", disabled=False)
            st.markdown("Please open the website to view the details: http://localhost:6006/")
            if show_log_button:
                ct.showLogs(log_dir)
                
        with tab3:
            st.subheader("Click the 'model_infer' button to trigger model inference ") 
            image_path = st.text_input("The test image path or directory :","demo.jpg") 
            check_point = st.text_input("The model check point name:","epoch_10.pth")
            test_button = st.button(label="model_infer", key="model_infer", disabled=False)
            if test_button:
                ct.test(image_path,check_point)
                tmp_dir = ct.image_save_dir
                cols = st.columns(3)
                if os.path.exists(tmp_dir):
                    image_names = os.listdir(tmp_dir)
                    images_names = image_names[:50] if len(image_names)>50 else image_names
                    for img_id,tmp in enumerate(image_names):
                        imgpath = os.path.join(tmp_dir,tmp.strip())
                        with cols[img_id %3]:
                            st.image(imgpath)
        with tab4:
            st.subheader("Click the 'convert' button to trigger model convert ") 
            image_shape = st.text_input("The image shape of onnx model exported are: image_height,image_width","320,640") 
            check_point = st.text_input("The export model check point name:","epoch_100.pth")
            convert_button = st.button(label="convert", key="convert", disabled=False)
            if "," in image_shape:
                image_shape = image_shape.strip().split(",")
            else:
                image_shape = image_shape.strip().split()
            if len(image_shape) ==2:
                if convert_button:
                    ct.export(check_point,image_shape)
                    save_model_path = ct.export_model_path
                    st.markdown("the onnx model is saved at: {}".format(save_model_path))
                    json_content = downloadFile(save_model_path)
                    model_name = save_model_path.split("/")[-1].strip()
                    st.download_button(label="Download onnx model",data=json_content,file_name=model_name)
            else:
                st.write("Please input the right number,the length of image shape should be: {}".format(2))
            