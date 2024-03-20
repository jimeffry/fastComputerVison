from cv_task.utils.register import CVTASKS,all_register

if __name__=="__main__":
    all_register()
    pstr = "YoLo"
    cf = "cv_task/config/config_carb.yml"
    ct = CVTASKS.get(pstr)(cf)
    ct.convert()