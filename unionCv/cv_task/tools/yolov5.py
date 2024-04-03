import os
import sys
import yaml
import json
from loguru import logger
from string import Template
from cv_task.basecore.basemodel import BaseClientTrainer
from cv_task.utils.register import CVTASKS

@CVTASKS.register(name="YoLo")
class YoLo(BaseClientTrainer):
    def __init__(self,configfile):
        super().__init__(configfile)
        self.python_train = "/home/lixy/prjs/git-prj/mmyolo/tools/train.py"
        self.python_infer = "/home/lixy/prjs/git-prj/mmyolo/demo/image_demo.py"
        self.export_onnx = "/home/lixy/prjs/git-prj/mmyolo/projects/easydeploy/tools/export_onnx.py"
        self.template_notune_file = "/home/lixy/prjs/git-prj/detect_tools/unionCv/cv_task/config/template_config.py"
        self.template_finetune_file = "/home/lixy/prjs/git-prj/detect_tools/unionCv/cv_task/config/template_finetune_config.py"
        self.config_dir = "/home/lixy/prjs/git-prj/mmyolo/configs/custom_dataset"
        self.optimize_anchor = "/home/lixy/prjs/git-prj/mmyolo/tools/analysis_tools/optimize_anchors.py"
        self.voc_convert = "/home/lixy/prjs/git-prj/fastComputerVison/unionCv/mmyolo/tools/dataset_converters/voc2coco.py"
        self.coco_split = "/home/lixy/prjs/git-prj/fastComputerVison/unionCv/mmyolo/tools/misc/coco_split.py"
        self.default_anchors = [
            [[60, 60], [154, 91], [143, 162]],  # P3/8
            [[242, 160], [189, 287],[391, 207]],  # P4/16
            [[353, 337], [539, 341], [443, 432]]  # P5/32
            ]
        self.getSaveModelPath()

    def genTrainConfigFile(self,template_filepath: str,config_filepath: str,data_dict: dict) -> None:
        tmp_file = open(template_filepath,"r")
        tmpl = Template(tmp_file.read())
        cfg_file = self.refileOpen(config_filepath,"w")
        cfg_cnt = []
        cfg_cnt.append(tmpl.substitute(
            work_dir=data_dict["workRoot"],
            data_dir=data_dict["dataRoot"],
            ann_file_train=data_dict["trainFile"],
            ann_file_val=data_dict["valFile"],
            class_name_str=data_dict["classNames"],
            color_show=data_dict["colors"],
            anchors_list=data_dict["anchors"],
            batch_size=data_dict["batchSize"],
            worker_nums=data_dict["workers"]
        ))
        cfg_file.writelines(cfg_cnt)
        cfg_file.close()
    
    def getSaveModelPath(self):
        config_file = self.configSet["configFileName"]
        save_dir = self.configSet["inputConfig"]["workRoot"]
        file_basename_no_extension = os.path.splitext(config_file)[0]
        self.model_save_path = os.path.join(save_dir,file_basename_no_extension)
        self.checkDir(self.model_save_path)
        self.config_file_path = os.path.join(self.config_dir,config_file)
        
    def train(self):
        logger.info("*************** yaml file:")
        logger.info(self.configSet)
        config_file = self.configSet["configFileName"]
        gpu_nums = int(self.configSet["gpuNums"])
        save_dir = self.model_save_path #self.configSet["inputConfig"]["workRoot"]
        self.checkDir(save_dir)
        if not config_file.endswith((".py")):
            logger.info("Error: the config file must be end with .py: {}".format(config_file))
        # self.config_file_path = os.path.join(self.config_dir,config_file)
        #=============
        #generate config
        tmp_dict = self.configSet["inputConfig"]
        config_dict = dict()
        for tmp_key in tmp_dict.keys():
            config_dict[tmp_key] = tmp_dict[tmp_key]
        config_dict["workRoot"] = self.model_save_path
        class_names = tmp_dict["classNames"]
        color_bar = []
        for i in range(len(class_names)):
            color_bar.append((20,220,60))
        config_dict["colors"] = color_bar
        config_dict["anchors"] = self.default_anchors
        if self.configSet["finetune"]:
            template_file = self.template_finetune_file
        else:
            template_file = self.template_notune_file
        self.genTrainConfigFile(template_file,self.config_file_path,config_dict)
        #=========gen anchors
        gen_anchor_sh = "python3 {} {} --algorithm v5-k-means --input-shape 640 640  --prior-match-thr 4.0 --out-dir {}".format(self.optimize_anchor,self.config_file_path,save_dir)
        anchor_path = json_path = os.path.join(save_dir, 'anchor_optimize_result.json')
        logger.info("begin to optime  anchors")
        os.system(gen_anchor_sh)
        logger.info("optime finished")
        with open(anchor_path,'r') as f:
            jsonitems = json.load(f)
            f.close()
        config_dict["anchors"] = jsonitems
        logger.info("begin to generate config file ")
        self.genTrainConfigFile(template_file,self.config_file_path,config_dict)
        logger.info("config file finished ")
        
        if int(self.configSet["resume"])>0:
            train_sh = "python3 -m torch.distributed.launch --nproc_per_node={} {}  {} --resume".format(gpu_nums,self.python_train,self.config_file_path)
        else:
            train_sh = "python3 -m torch.distributed.launch --nproc_per_node={} {}  {} ".format(gpu_nums,self.python_train,self.config_file_path)
 
        train_sh_file = os.path.join(save_dir,"train.sh")
        with self.refileOpen(train_sh_file,"w") as f:
            f.write("#!/home/lixy/tools/mmyolo_env/bin/\n")
            f.write(train_sh+"\n")
            f.close()
        
        run_sh = "bash {}".format(train_sh)
        logger.info("begin to run  train")
        os.system(train_sh)
        logger.info("train finished")

    def test(self,image_path:str,model_name:str) ->None:
        self.image_save_dir = os.path.join(self.model_save_path,"demo_images")
        check_point = os.path.join(self.model_save_path,model_name)
        self.checkDir(self.image_save_dir)
        test_sh = "python3 {} {} {} {} --out-dir {}".format(self.python_infer,image_path,self.config_file_path,check_point,self.image_save_dir)
        test_sh_file = os.path.join(self.model_save_path,"test.sh")
        with self.refileOpen(test_sh_file,"w") as f:
            f.write("#!/home/lixy/tools/mmyolo_env/bin/\n")
            f.write(test_sh+"\n")
            f.close()
        logger.info("begin to run  test")
        os.system(test_sh)
        logger.info("test finished")
    
    def export(self,model_name: str,image_shape: list) ->None:
        imgh,imgw = image_shape
        check_point = os.path.join(self.model_save_path,model_name)
        convert_sh = "python3 {} {} {} --work-dir {} --img-size {} {} --simplify --backend ONNXRUNTIME --device cpu --model-only".format(self.export_onnx,\
                        self.config_file_path,check_point,self.model_save_path,imgh,imgw)
        cvt_sh_file = os.path.join(self.model_save_path,"cvt.sh")
        with self.refileOpen(cvt_sh_file,"w") as f:
            f.write("#!/home/lixy/tools/mmyolo_env/bin/\n")
            f.write(convert_sh+"\n")
            f.close()
        logger.info("begin to run  convert")
        os.system(convert_sh)
        logger.info("convert finished")
        self.export_model_path = check_point.replace(".pth",".onnx")

    def convertXmlData(self) ->None:
        data_root = self.configSet["inputConfig"]["dataRoot"]
        convert_sh = "python3 {}  --data-dir {} ".format(self.voc_convert, data_root)
        dataset_name = data_root.split("/")[-1]
        coco_json_file = os.path.join(data_root,"annotations/"+dataset_name+".json")
        out_dir = os.path.join(data_root,"annotations")
        spl_sh = "python3 {} --json {} --out-dir {} --ratios 0.9 0.1".format(self.coco_split,coco_json_file,out_dir)
        logger.info("begin to run data convert")
        os.system(convert_sh)
        os.system(spl_sh)
        logger.info("data convert finished")

    def showLogs(self,logdir:str)->None:
        cnts = os.listdir(logdir.strip())
        dir_list = []
        for tmp in cnts:
            tmp_dir = os.path.join(logdir,tmp)
            if os.path.isdir(tmp_dir):
                dir_list.append(tmp)
        # print(dir_list)
        if len(dir_list)>0:
            dir_list = sorted(dir_list)
            log_dir = os.path.join(logdir,dir_list[-1],"vis_data")
        else:
            log_dir = logdir
        show_sh = "tensorboard --logdir={}".format(log_dir.strip())
        os.system(show_sh)
