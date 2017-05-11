#!//anaconda/envs/carnd-term1/bin/python
"""
Program to train a vehicle detection classifier and apply it to a video
Usage:
vehicle_detection.py train <modelname> <images_folder>
  modelname      name of file to save the trained model
  images_folder  path of where the training images are present
                 image_folder/cars/*.jpg
                 image_folder/not-cars/*.jpg

vehicle_detection.py process <modelname> <video>
  modelname    name of pretrained classifier
  video        name of video to process


 Training:
  Load training Image
  split test training data
  train classifier (linear?)
  save classifier model
 Process Video:
  select pre-trained classifier
  process video fames using classifier
    bounding boxes
  write output video

"""
import glob
import time
import pickle
import yaml
import click
from munch import Munch
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import hog
from scipy.ndimage.measurements import label
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from moviepy.editor import VideoFileClip


class VehicleDetection():
    def __init__(self, model, params, save_processed=False):
        self.model = model
        self.hp = Munch.fromDict(params).vehicle_detection
        self.windows = None
        self.frame = 0
        self.save_processed = save_processed
        self.img_name = ""

    def generate_windows(self, img):
        """
        function that takes an image,
        start and stop positions in both x and y,
        window size (x and y dimensions),
        and overlap fraction (for both x and y)
        If x and/or y start/stop positions not defined, set to image size
        """
        windows = []
        for scale in self.hp.search_scales:
            if not scale.xrange[1]:
                scale.xrange[1] = img.shape[1]
            if not scale.yrange[1]:
                scale.yrange[1] = img.shape[0]

            xstep = scale.winsize[0]*(1 - scale.overlap[0])
            ystep = scale.winsize[1]*(1 - scale.overlap[1])
            for x in range(scale.xrange[0],
                           int(scale.xrange[1]-xstep), int(xstep)):
                for y in range(scale.yrange[0],
                               int(scale.yrange[1]-ystep), int(ystep)):
                    windows.append(((x, y),
                                    (x+scale.winsize[0], y+scale.winsize[1])))
        self.windows = windows

    def draw_boxes(self, oimg, boxes, prefix=""):
        """
        draw bounding boxes on image
        """
        if not self.save_processed:
            return
        img = np.copy(oimg)
        for box in boxes:
            color = (
                   np.random.randint(5, 255),
                   np.random.randint(5, 255),
                   np.random.randint(5, 255),
                  )
            cv2.rectangle(img, box[0], box[1],
                          color,
                          self.hp.box_thickness)
        self.model.dbg_img.add(img, note=prefix)

    def search_windows(self, img):
        on_windows = []
        for window in self.windows:
            test_img = cv2.resize(img[window[0][1]:window[1][1],
                                      window[0][0]:window[1][0]], (64, 64))
            features = self.model.image_features(test_img)
            X = self.model.scaler.transform(np.array(features).reshape(1, -1))
            y = self.model.clf.predict(X)
            if y == 1:
                on_windows.append(window)
        return on_windows

    def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient,
                  pix_per_cell, cell_per_block, spatial_size, hist_bins):
        draw_img = np.copy(img)
        img = img.astype(np.float32)/255

        img_tosearch = img[ystart:ystop,:,:]
        ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
            
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
        nfeat_per_block = orient*cell_per_block**2
        
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        
        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
        
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
              
                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
                test_prediction = svc.predict(test_features)
                
                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                    
        return draw_img

    def add_heat(self, heatmap, boxes):
        """
        Add +1 for all pixels inside each bbox
        Each "box" takes the form ((x1, y1), (x2, y2))
        """
        for box in boxes:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        return heatmap

    def apply_threshold(self, heatmap):
        """
        Zero out pixels below threshold
        """
        heatmap[heatmap <= self.hp.threshold] = 0
        return heatmap

    def draw_labeled_bboxes(self, img, labels):
        """
        Iterate through all detected cars
        Find pixels with each car_number label value
        Identify x and y values of those pixels
        Define a bounding box based on min/max x and y
        Draw the box on the image
        """
        for car_number in range(1, labels[1]+1):
            nonzero = (labels[0] == car_number).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                    (np.max(nonzerox), np.max(nonzeroy)))
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
        if self.save_processed:
            self.model.dbg_img.add(img, note="labelled")
        return img

    def _heatmap(self, image, boxes):
        heatmap = np.zeros_like(image[:, :, 0]).astype(np.float)
        heatmap = self.add_heat(heatmap, boxes)
        heatmap = self.apply_threshold(heatmap)
        heatmap = np.clip(heatmap, 0, 255)
        return heatmap

    def process_frame(self, image, name=None):
        self.frame += 1
        if not name:
            name = "frame_{}.jpg".format(self.frame)
        if not self.windows:
            self.generate_windows(image)

        with DebugImage(name, (28, 40), 5, 2,
                        self.save_processed) as self.model.dbg_img:
            if self.save_processed:
                self.model.dbg_img.add(image, note="input")
            self.draw_boxes(image, self.windows, prefix="allwin")
            if self.save_processed:
                self.model.image_features(image, save=True)
            detected = self.search_windows(image)
            # print("detected: {}".format(len(detected)))
            self.draw_boxes(image, detected, prefix="detected_win")
            heatmap = self._heatmap(image, detected)
            if self.save_processed:
                self.model.dbg_img.add(heatmap, note="heatmap", cmap="hot")
            labels = label(heatmap)
            processed_img = self.draw_labeled_bboxes(np.copy(image), labels)
        return processed_img

    def process_clip(self, path):
        video = VideoFileClip(path)
        processed = video.fl_image(self.process_frame)
        return processed


class DebugImage():
    def __init__(self, name, size, rows, cols, save=True):
        self.name = name
        self.rows = rows
        self.cols = cols
        self.figsize = size
        self.picnum = 0

    def __enter__(self):
        self.fig = plt.figure(figsize=self.figsize)
        return self

    def __exit__(self, *args):
        self.fig.savefig("output_images/{}.jpg".format(self.name))

    def add(self, img, note, cmap=None):
        self.picnum += 1
        plot = self.fig.add_subplot(self.rows, self.cols, self.picnum)
        plot.imshow(img, cmap=cmap)
        print("saving {}-{}".format(note, self.name))
        plt.title(note)


class Model():
    """
    Classifier Model
    """
    def __init__(self, name, hyperparams, path):
        """
        init
        """
        self.name = name
        self.hp = Munch.fromDict(hyperparams).model
        self.path = path
        self.dbg_img = None
        print(self)

    def __str__(self):
        s = "Model: {}\n".format(self.name)
        s += " trained-images: {}\n".format(self.path)
        s += " params:\n"
        s += yaml.dump(self.hp.toDict())
        return s

    # -- feature extraction -------------------------------------
    def _hog(self, img, channel, feature_vec=True, save=False):
        _img = img[:, :, channel]
        rv = hog(_img, orientations=self.hp.hog.orient,
                 pixels_per_cell=(self.hp.hog.pix_per_cell,
                                  self.hp.hog.pix_per_cell),
                 cells_per_block=(self.hp.hog.cell_per_block,
                                  self.hp.hog.cell_per_block),
                 transform_sqrt=True,
                 visualise=save,
                 feature_vector=False)
        if save:
            self.dbg_img.add(rv[1], cmap="hot",
                             note="hog_c{}".format(channel))
            rv = rv[0]
        return rv.ravel()

    def get_hog_features(self, img, save=False):
        """
        Define a function to return HOG features and visualization
        Call with two outputs if vis==True
        """
        if self.hp.hog.channel == 'ALL':
            features = []
            for channel in range(img.shape[2]):
                features.append(self._hog(img, channel, save=save))
            features = np.ravel(features)
        else:
            features = self._hog(img, self.hp.hog.channel, save=save)
        return features

    def bin_spatial(self, img, save=False):
        """
        function to compute binned color features
        Use cv2.resize().ravel() to create the feature vector
        """
        features = cv2.resize(img, (self.hp.spatialbin.size,
                                    self.hp.spatialbin.size)).ravel()
        return features

    def color_hist(self, img, bins_range=(0, 256), save=False):
        """
        function to compute color histogram features
        Compute the histogram of the color channels separately
        """
        bins = self.hp.colorhist.bins
        c1 = np.histogram(img[:, :, 0], bins=bins, range=bins_range)
        c2 = np.histogram(img[:, :, 1], bins=bins, range=bins_range)
        c3 = np.histogram(img[:, :, 2], bins=bins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((c1[0], c2[0], c3[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    def _color_img(self, img):
        if self.hp.color_space != 'RGB':
            if self.hp.color_space == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif self.hp.color_space == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif self.hp.color_space == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif self.hp.color_space == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif self.hp.color_space == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(img)
        return feature_image

    def image_features(self, img, save=False):
        """
        function to extract features from a single image window
        This function is very similar to extract_features()
        just for a single image rather than list of images
        """
        features = []
        img = self._color_img(img)
        if save:
            self.dbg_img.add(img, note=self.hp.color_space)
        if self.hp.spatialbin.enabled:
            features.append(self.bin_spatial(img, save=save))
        if self.hp.colorhist.enabled:
            features.append(self.color_hist(img, save=save))
        if self.hp.hog.enabled:
            features.append(self.get_hog_features(img, save=save))
        return np.concatenate(features)

    def extract_features(self, img_paths):
        return [self.image_features(mpimg.imread(img)) for img in img_paths]

    def _get_classifier_model(self):
        if self.hp.classifier == "linear":
            clf = LinearSVC()
        else:
            assert(False)
        return clf

    def train(self):
        # Read in cars and notcars
        cars = glob.glob(self.path + "/vehicles/*/*.png")
        notcars = glob.glob(self.path + "/non-vehicles/*/*.png")
        self.save_training_samples(cars, notcars)
        car_features = self.extract_features(cars)
        notcar_features = self.extract_features(notcars)

        # features
        Xin = np.vstack((car_features, notcar_features)).astype(np.float64)
        self.scaler = StandardScaler().fit(Xin)
        X = self.scaler.transform(Xin)

        # labels
        y = np.hstack((np.ones(len(car_features)),
                       np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.hp.test_train_split,
                             random_state=np.random.randint(0, 100))
        self.clf = self._get_classifier_model()
        t = time.time()
        self.clf.fit(X_train, y_train)
        t2 = time.time()
        print("Trained in ", round(t2-t, 2))
        print('Test Accuracy ', round(self.clf.score(X_test, y_test), 4))

    def save_training_samples(self, cars, notcars):
        images = []
        images.extend(cars[0:2])
        images.extend(notcars[0:2])
        # for _ in range(2):
        #     images.append(cars[np.random.randint(0, len(cars)-1)])
        # for _ in range(2):
        #     images.append(notcars[np.random.randint(0, len(notcars)-1)])
        with DebugImage(self.name, (32, 40), 4, 5,
                        True) as self.dbg_img:
            for image in images:
                _image = mpimg.imread(image)
                self.dbg_img.add(_image, note=image)
                _ = self.image_features(_image, save=True)
        self.dbg_img = None

    def save(self):
        pickle.dump(self, file=open(self.name+".p", "wb"))

    @classmethod
    def load(cls, name):
        M = pickle.load(file=open(name+".p", "rb"))
        print(M)
        return M


# -- CLI entry points ---------------------------------------------------------
@click.group()
def cli():
    pass


@click.command()
@click.option('--model', help='Name of model being trained. Hyperparams are\
                               read from <model>.yml. Trained model is \
                               saved as <model>.p')
@click.option('--images', type=click.Path(exists=True),
              help="path to images. images/*/*/*.png")
def train(model, images):
    click.echo('Training the {} model'.format(model))
    params = yaml.load(open(model+".yml", "r"))
    click.echo("Params:")
    click.echo(params["model"])
    model = Model(name=model, hyperparams=params, path=images)
    model.train()
    model.save()


@click.command()
@click.option('--model', help='Name of model being loaded from <model>.p')
@click.option('--config', help='image processing configs')
@click.argument('video', type=click.Path(exists=True))
def process(model, video, config):
    click.echo('Process the video')
    params = yaml.load(open(config+".yml", "r"))
    M = Model.load(model)
    vd = VehicleDetection(M, params)
    out = vd.process_clip(video)
    out.write_videofile('output_images/output_' + video, audio=False)


@click.command()
@click.option('--model', help='Name of model being loaded from <model>.p')
@click.option('--config', help='image processing configs')
def test(model, config):
    click.echo('Running on test images')
    params = yaml.load(open(config+".yml", "r"))
    M = Model.load(model)
    vd = VehicleDetection(M, params, save_processed=True)
    for image in glob.glob("test_images/*.jpg"):
        name = model + "_" + image.split("/")[-1].split(".")[0]
        vd.process_frame(mpimg.imread(image), name=name)


cli.add_command(train)
cli.add_command(process)
cli.add_command(test)


# -- Main ---------------------------------------------------------------------
# Training:
#  Load training Image
#  split test training data
#  train classifier (linear?)
#  save classifier model
# Process Video:
#  select pre-trained classifier
#  process video fames using classifier
#    bounding boxes
#  write output video

if __name__ == '__main__':
    cli()
