import os
import sys
import yaml
import psutil
from abc import abstractmethod

class BaseClientTrainer():
    def __init__(self,configfile: str) -> None:
        self.loadConfigFromFile(configfile)

    def loadConfigFromFile(self,filepath):
        if os.path.isfile(filepath):
            with open(filepath,encoding='utf-8') as f:
                self.configSet = yaml.load(f,Loader=yaml.FullLoader)
        else:
            self.configSet = yaml.load(filepath,Loader=yaml.FullLoader)
    @staticmethod
    def terminateProcesses(script_name):
        # # 遍历所有进程
        # for proc in psutil.process_iter(['pid', 'ppid', 'cmdline']):
        #     cmdline = proc.info['cmdline']
        #     if cmdline and script_name in " ".join(cmdline):
        #         # 获取进程 PID 和 PPID
        #         target_pid = proc.info['pid']
        #         target_ppid = proc.info['ppid']

        #         # 终止进程和父进程
        #         psutil.Process(target_pid).terminate()
        #         psutil.Process(target_ppid).terminate()

        #         print(f"Terminated process with PID {target_pid} and its parent PID {target_ppid} running {script_name}")

        # 搜索包含指定脚本名称的进程
        for proc in psutil.process_iter(['pid', 'cmdline']):
            cmdline = proc.info['cmdline']
            if cmdline and script_name in " ".join(cmdline):
                # 获取进程 PID
                target_pid = proc.info['pid']

                # 终止进程
                psutil.Process(target_pid).terminate()
                print(f"Terminated process with PID {target_pid} running {script_name}")

    @staticmethod
    def refileOpen(filepath: str,mode: str):
        descriptor = os.open(path=filepath,flags=os.O_WRONLY | os.O_CREAT | os.O_TRUNC ,mode=0o777)
        fh = open(descriptor, mode)
        return fh

    @staticmethod
    def genYamlFile(filepath: str,data_dict : dict) -> None:
        with open(filepath,'w',encoding="utf-8") as f:
            yaml.dump(data=data_dict,stream=f,allow_unicode=True)

    @staticmethod
    def checkDir(filedir: str) -> None:
        if not os.path.exists(filedir):
            os.makedirs(filedir,mode=0o777)

    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError("train not implemented")
    @abstractmethod
    def genTrainConfigFile(template_filepath: str,config_filepath: str,data_dict: dict) -> None:
        raise NotImplementedError("train not implemented")
        