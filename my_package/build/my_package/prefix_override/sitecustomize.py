import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/mnt/c/research/program/ADMM_with_ICNN_on_ROS/my_package/install/my_package'
