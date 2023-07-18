from rpy2.robjects.packages import importr
def read_r():
    base = importr('base')
    motifs_obj = base.readRDS('./CP_motifs_PWM.rds')
    print(motifs_obj[0][0])
read_r()


