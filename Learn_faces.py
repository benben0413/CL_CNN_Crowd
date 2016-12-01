import sys

from CNN.CNN import *


dataset_path = 'Datasets/TLand_2016_faces_13516pos_4279neg.pkl'
cnn = CNN('weights/weights.pkl')

if(os.path.isfile(dataset_path)):
    print ("Loading dataset...")
    f = open(dataset_path, 'rb')
    Train, Valid, Test = cPickle.load(f)
    f.close()
else:
    print "Dataset not found"
    sys.exit()


cnn.fit([Train, Valid])

# fill_image_weights()
