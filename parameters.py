################################################################################
# Data

def reduce_data_randomly(from_label, to_label, filter):
    dis = {}

    for l in range(from_label, to_label):
        dis[l] = filter

    dis[0] = 0
    return dis


PATH_TO_DATA41 = '/data/self-driving-car/course/project3/data/data41/'
PATH_TO_DATA42 = '/data/self-driving-car/course/project3/data/data42/'

PATH_TO_DATA44 = '/data/self-driving-car/course/project3/data/data44/'
PATH_TO_DATA45 = '/data/self-driving-car/course/project3/data/data45/'

PATH_TO_DATA51 = '/data/self-driving-car/course/project3/data/data51/'

PATH_TO_DATA61 = '/data/self-driving-car/course/project3/data/data61/'
PATH_TO_DATA62 = '/data/self-driving-car/course/project3/data/data62/'

PATH_TO_DATA63 = '/data/self-driving-car/course/project3/data/data63/'
PATH_TO_DATA64 = '/data/self-driving-car/course/project3/data/data64/'

PATH_TO_DATA71 = '/data/self-driving-car/course/project3/data/data71/'
PATH_TO_DATA72 = '/data/self-driving-car/course/project3/data/data72/'

PATH_TO_DATA80 = '/data/self-driving-car/course/project3/data/data80/'
PATH_TO_DATA81 = '/data/self-driving-car/course/project3/data/data81/'
PATH_TO_DATA82 = '/data/self-driving-car/course/project3/data/data82/'
PATH_TO_DATA83 = '/data/self-driving-car/course/project3/data/data83/'
PATH_TO_DATA84 = '/data/self-driving-car/course/project3/data/data84/'

# Track 1
DATA01 = [PATH_TO_DATA42, {-1: 0.6, 0: 0.1, 1: 0.6}]         # T1, lane
DATA02 = [PATH_TO_DATA45, {-1: 0.6, 0: 0.1, 1: 0.6}]         # T1, lane
DATA03 = [PATH_TO_DATA71, {-1: 0.6, 0: 0.1, 1: 0.6}]         # T1, lane
DATA04 = [PATH_TO_DATA41, reduce_data_randomly(-5, 5, 0.1)]  # T1, recovery
DATA05 = [PATH_TO_DATA44, reduce_data_randomly(-5, 5, 0.1)]  # T1, recovery
DATA06 = [PATH_TO_DATA51, reduce_data_randomly(-5, 5, 0.1)]  # T1, recovery, off road
DATA07 = [PATH_TO_DATA72, reduce_data_randomly(-5, 5, 0.1)]  # T1, recovery, off road

# Track 2
DATA51 = [PATH_TO_DATA61, {-1: 0.6, 0: 0.1, 1: 0.6}]         # T2, lane
DATA52 = [PATH_TO_DATA64, {-1: 0.6, 0: 0.1, 1: 0.6}]         # T2, lane
DATA53 = [PATH_TO_DATA81, {-1: 0.6, 0: 0.1, 1: 0.6}]         # T2, lane
DATA54 = [PATH_TO_DATA82, {-1: 0.6, 0: 0.1, 1: 0.6}]         # T2, lane
DATA55 = [PATH_TO_DATA83, {-1: 0.6, 0: 0.1, 1: 0.6}]         # T2, lane
DATA56 = [PATH_TO_DATA62, reduce_data_randomly(-5, 5, 0.1)]  # T2, recovery
DATA57 = [PATH_TO_DATA63, reduce_data_randomly(-5, 5, 0.1)]  # T2, recovery
DATA58 = [PATH_TO_DATA72, reduce_data_randomly(-5, 5, 0.1)]  # T1, recovery
DATA59 = [PATH_TO_DATA80, reduce_data_randomly(-5, 5, 0.1)]  # T2, recovery
DATA60 = [PATH_TO_DATA84, reduce_data_randomly(-5, 5, 0.1)]  # T2, recovery

DATA = [DATA01, DATA02, DATA03, DATA51, DATA52, DATA53, DATA54, DATA55]  # M1, M2, M3
#DATA = [DATA01, DATA02, DATA03]
DATA = [DATA01, DATA02, DATA03, DATA04, DATA05, DATA06, DATA07, DATA51, DATA52, DATA53, DATA54, DATA55, DATA56, DATA57,
        DATA59, DATA60]  # M4

# DATA = [DATA51, DATA52, DATA53, DATA54, DATA55]  # X4
# DATA = [DATA55, DATA60, DATA03, DATA07]  # X4

LOG_FILE = 'driving_log.csv'

################################################################################
# Training
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 160

NUM_CLASSES = 1

TRAIN_BATCH_SIZE = 1500
TRAIN_KERAS_BATCH_SIZE = 1000
TRAIN_KERAS_EPOCH = 4

TEST_DATA = 0.05

DROPOUT = 0.3
L2_REGULARIZATION = 0.0001

################################################################################
# Pre-processing & data generation

THRASHOLD_STEERING_ANGLE = 1
THRESHOLD_SPEED = 0.5

STEERING_ANGLE_ADJUSTMENT_MU = 0.20
STEERING_ANGLE_ADJUSTMENT_SIGMA = 0.1

SUPPRESS_LABEL_0 = False
GENERATED_CLONES = 2
GENERATOR_TRANSFORM = 100
