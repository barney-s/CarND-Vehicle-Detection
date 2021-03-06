#!//anaconda/envs/carnd-term1/bin/python
"""
Author: Barni S

Program to train a vehicle detection classifier and apply it to a video
./vehicle_detection.py --help for more details

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

    def find_cars(self, img, scale):
        search_p = self.hp.search_scales[0]
        detected = []
        scaler = self.model.scaler
        clf = self.model.clf
        pix_per_cell = self.model.hp.hog.pix_per_cell
        cell_per_block = self.model.hp.hog.cell_per_block
        ystart = search_p.yrange[0]
        ystop = search_p.yrange[1]
        xstart = search_p.xrange[0]
        xstop = search_p.xrange[1]

        img_tosearch = img[ystart:ystop, xstart:xstop, :]
        ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_BGR2YCrCb)
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch,
                                         (np.int(imshape[1]/scale),
                                          np.int(imshape[0]/scale)))

        # Define blocks and steps as above
        nxblocks = (ctrans_tosearch[:, :, 0].shape[1] // pix_per_cell)\
            - cell_per_block + 1
        nyblocks = (ctrans_tosearch[:, :, 0].shape[0] // pix_per_cell)\
            - cell_per_block + 1
        # nfeat_per_block = self.hp.hog.orient*cell_per_block**2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        nxblocks_per_window = (search_p.winsize[0] // pix_per_cell) - cell_per_block + 1
        nyblocks_per_window = (search_p.winsize[1] // pix_per_cell) - cell_per_block + 1
        nxsteps = (nxblocks - nxblocks_per_window) // search_p.overlap[0]
        nysteps = (nyblocks - nyblocks_per_window) // search_p.overlap[1]

        # Compute individual channel HOG features for the entire image
        hog1 = self.model.get_hog_features(ctrans_tosearch, channel=0, ravel=False, feature_vec=False)
        hog2 = self.model.get_hog_features(ctrans_tosearch, channel=1, ravel=False, feature_vec=False)
        hog3 = self.model.get_hog_features(ctrans_tosearch, channel=2, ravel=False, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*search_p.overlap[1]
                xpos = xb*search_p.overlap[0]
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nyblocks_per_window,
                                 xpos:xpos+nxblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nyblocks_per_window,
                                 xpos:xpos+nxblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos+nyblocks_per_window,
                                 xpos:xpos+nxblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+search_p.winsize[1],
                                                    xleft:xleft+search_p.winsize[0]],
                                    (64, 64))

                # Get color features
                spatial_features = self.model.bin_spatial(subimg)
                hist_features = self.model.color_hist(subimg)

                # Scale features and make a prediction
                allfeatures = np.concatenate((spatial_features,hist_features,hog_features))
                # allfeatures = np.hstack((spatial_features,
                #                          hist_features,
                #                          hog_features)).reshape(1, -1)
                test_features = scaler.transform(allfeatures)
                test_prediction = clf.predict(test_features)
                if test_prediction[0] == 1:
                    xleft = np.int(xleft*scale)
                    ydraw = np.int(ytop*scale)
                    winx_draw = np.int(search_p.winsize[0]*scale)
                    winy_draw = np.int(search_p.winsize[1]*scale)
                    detected.append(((xleft+xstart, ydraw+ystart),
                                     (xleft+xstart+winx_draw, ydraw+winy_draw+ystart)))
        return detected

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

    def _process_frame_search_windows(self, image):
        self.draw_boxes(image, self.windows, prefix="allwin")
        if self.save_processed:
            self.model.image_features(image, save=True)
        detected = self.search_windows(image)
        return detected

    def _process_frame_hog_subsampling(self, image):
        detected = []
        search_p = self.hp.search_scales[0]
        for scale in search_p.scales:
            detected.extend(self.find_cars(image, scale))
        return detected

    def process_frame(self, image, name=None):
        self.frame += 1
        _image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if not name:
            name = "frame_{}.jpg".format(self.frame)
        if not self.windows:
            self.generate_windows(image)

        with DebugImage(name, (28, 40), 5, 2,
                        self.save_processed) as self.model.dbg_img:
            if self.save_processed:
                self.model.dbg_img.add(image, note="input")
            if self.hp.search_mode == "windows":
                detected = self._process_frame_search_windows(_image)
            else:
                detected = self._process_frame_hog_subsampling(image)
            self.draw_boxes(image, detected, prefix="detected_win")
            heatmap = self._heatmap(image, detected)
            if self.save_processed:
                self.model.dbg_img.add(heatmap, note="heatmap", cmap="hot")
            labels = label(heatmap)
            processed_img = self.draw_labeled_bboxes(image, labels)
        return processed_img

    def process_clip(self, path):
        video = VideoFileClip(path) # .subclip(0.0, 0.2)
        processed = video.fl_image(self.process_frame)
        return processed


class DebugImage():
    def __init__(self, name, size, rows, cols, save=True):
        self.name = name
        self.rows = rows
        self.cols = cols
        self.figsize = size
        self.picnum = 0
        self.save = save

    def __enter__(self):
        if self.save:
            self.fig = plt.figure(figsize=self.figsize)
        return self

    def __exit__(self, *args):
        if self.save:
            self.fig.savefig("output_images/{}.jpg".format(self.name))

    def add(self, img, note, cmap=None):
        if not self.save:
            return
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
    def _hog(self, img, channel, feature_vec=True, save=False, ravel=True):
        _img = img[:, :, channel]
        rv = hog(_img, orientations=self.hp.hog.orient,
                 pixels_per_cell=(self.hp.hog.pix_per_cell,
                                  self.hp.hog.pix_per_cell),
                 cells_per_block=(self.hp.hog.cell_per_block,
                                  self.hp.hog.cell_per_block),
                 # transform_sqrt=True,
                 visualise=save,
                 feature_vector=feature_vec)
        if save:
            self.dbg_img.add(rv[1], cmap="hot",
                             note="hog_c{}".format(channel))
            rv = rv[0]
        if ravel:
           rv = rv.ravel()
        return rv

    def get_hog_features(self, img, save=False, channel=None, ravel=True, feature_vec=True):
        """
        Define a function to return HOG features and visualization
        Call with two outputs if vis==True


        """
        if channel is None:
            channel = self.hp.hog.channel
        if channel == 'ALL':
            features = []
            for channel in range(img.shape[2]):
                features.append(self._hog(img, channel, save=save, feature_vec=feature_vec))
            features = np.ravel(features)
        else:
            features = self._hog(img, channel, save=save, ravel=ravel, feature_vec=feature_vec)
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
        if self.hp.color_space != 'BGR':
            if self.hp.color_space == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            elif self.hp.color_space == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
            elif self.hp.color_space == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
            elif self.hp.color_space == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            elif self.hp.color_space == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            elif self.hp.color_space == 'RGB':
                feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
        return [self.image_features(cv2.imread(img)) for img in img_paths]

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
                _image = cv2.imread(image)
                self.dbg_img.add(_image, note=image)
                _ = self.image_features(_image, save=True)
        self.dbg_img = None

    def save(self):
        pickle.dump(self, file=open("models/" + self.name+".p", "wb"))

    @classmethod
    def load(cls, name):
        M = pickle.load(file=open(name, "rb"))
        print(M)
        return M


# -- CLI entry points ---------------------------------------------------------
@click.group()
def cli():
    pass


@click.command()
@click.option('--params', help='Hyperparams of the training model.\
                               Trained model is saved under models/<model>.p')
@click.option('--images', type=click.Path(exists=True),
              help="path to images. Should have images/vehicles/*/*.png,\
                    images/non-vehicles/*/*.png")
def train(params, images):
    name = ".".join(params.split("/")[-1].split(".")[:-1])
    click.echo('Training the {} model'.format(name))
    params = yaml.load(open(params, "r"))
    click.echo("Params:")
    click.echo(params["model"])
    model = Model(name=name, hyperparams=params, path=images)
    model.train()
    model.save()


@click.command()
@click.option('--model', help='Path of model file')
@click.option('--config', help='vehicle detection window configs')
@click.argument('video',
                # help='path to video file. Output is saved\
                #      with output_ prefix',
                type=click.Path(exists=True))
def process(model, video, config):
    click.echo('Process the video')
    params = yaml.load(open(config, "r"))
    M = Model.load(model)
    vd = VehicleDetection(M, params)
    out = vd.process_clip(video)
    out.write_videofile('output_images/output_' + video, audio=False)


@click.command()
@click.option('--model', help='Path of the model file')
@click.option('--config', help='vehicle detection window configs')
def test(model, config):
    click.echo('Running on test images')
    params = yaml.load(open(config, "r"))
    M = Model.load(model)
    vd = VehicleDetection(M, params, save_processed=True)
    for image in glob.glob("test_images/*.jpg"):
        name = ".".join(model.split("/")[-1].split(".")[:-1]) + "_" +\
            image.split("/")[-1].split(".")[0]
        vd.process_frame(mpimg.imread(image), name=name)


# -- Main ---------------------------------------------------------------------
cli.add_command(train)
cli.add_command(process)
cli.add_command(test)

if __name__ == '__main__':
    cli()
