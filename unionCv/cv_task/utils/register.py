from cv_task.utils.registry import Registry

CVTASKS = Registry("cv_tasks")

def all_register():
    try:
        import cv_task.tools
    except AssertionError:
        pass