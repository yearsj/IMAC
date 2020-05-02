from os.path import expanduser
HOME_DIR = expanduser("/data/chenzhuo/drift/java")

MOA_DIR = '{home_dir}/execmoa'.format(home_dir = HOME_DIR)
MOA_STUMP = "java -cp {moa_dir}/moa.jar -javaagent:{moa_dir}/sizeofag.jar".format(moa_dir = MOA_DIR)

INDEX_COL = 'learning evaluation instances'

