import sys
sys.path.append('../')
from util.header import *


class BaseDataset(Dataset):
    def __init__(self, opt, mode):
        super(BaseDataset, self).__init__()
        self.opt = opt
        self.mode = mode
        if self.mode == 'Train':
            self.data_file = self.opt.train_pth
        elif self.mode == 'Val':
            self.data_file = self.opt.val_pth
        else:
            self.data_file = self.opt.test_pth
        self.images, self.labels = self._load_data()
        self.transformer = self._get_transformer()
        self.data_size = len(self.images)
        self.sigma = opt.sigma
        self.flip = False
        self.heatmap_size = opt.heatmap_size
        self.num_joints = 17
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
                           [13, 14], [15, 16]]

    def _get_transformer(self):
        transform_list = []
        # resize
        transform_list.append(
            transforms.Resize(self.opt.input_size, Image.BICUBIC))

        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(self.opt.mean,
                                                   self.opt.std))
        return transforms.Compose(transform_list)

    def _load_joints(self, json_pth):
        '''
        parse joints json file
        '''
        file = open(json_pth, 'r')
        file = json.dumps(eval(file.read()))
        inf_json = json.loads(file)
        inf_joints = inf_json[0]['keypoints']
        labels = []
        for name, label in inf_joints.items():
            temp = np.array(list(map(float, label)))
            # all coords will join for train
            temp[-1] = 1
            labels.append(temp)
        return np.array(labels)

    def _load_data(self):
        images, labels = list(), list()
        if not os.path.exists(self.data_file):
            raise ValueError("data file is not exists!")
        rf = open(self.data_file, 'r')
        lines = rf.readlines()
        for line in lines:
            inf = line.strip().split(' ')
            img_pth = self.opt.img_dir + inf[0]
            # pdb.set_trace()
            json_pth = img_pth + '.json'
            if not os.path.isfile(img_pth) or not os.path.isfile(json_pth):
                continue
            if not os.path.exists(img_pth) or not os.path.exists(json_pth):
                continue

            images.append(img_pth)
            labels.append(self._load_joints(json_pth))
        return images, labels

    def _load_image(self, image_file):
        input = cv2.imread(image_file, cv2.IMREAD_COLOR)
        return input

    def _fliplr_joints(self, joints, joints_vis, width):
        # Flip horizontal
        joints[:, 0] = width - joints[:, 0] - 1

        # Change left-right parts
        for pair in self.flip_pairs:
            joints[pair[0], :], joints[pair[1], :] = joints[
                pair[1], :], joints[pair[0], :].copy()
            joints_vis[pair[0], :], joints_vis[pair[1], :] = joints_vis[
                pair[1], :], joints_vis[pair[0], :].copy()

        return joints * joints_vis, joints_vis

    def _scale_transform(self, pt, width, height):
        pt[0] = pt[0] / width * self.opt.input_size[0]
        pt[1] = pt[1] / height * self.opt.input_size[1]
        return pt

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        target = np.zeros(
            (self.num_joints, self.heatmap_size[1], self.heatmap_size[0]),
            dtype=np.float32)

        tmp_size = self.sigma * 3

        # convert the coords of input size to the coords of gaussian size
        for joint_id in range(self.num_joints):
            feat_stride = [
                self.opt.input_size[0] / self.heatmap_size[0],
                self.opt.input_size[1] / self.heatmap_size[1]
            ]
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * self.sigma**2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight

    def __getitem__(self, index):
        image_file = self.images[index % self.data_size]
        input = self._load_image(image_file)
        height, width = input.shape[0], input.shape[1]

        labels_inf = self.labels[index % self.data_size]

        # init joints
        # pdb.set_trace()
        joints = np.zeros((self.num_joints, 3), dtype=np.float)
        joints_vis = np.zeros((self.num_joints, 3), dtype=np.float)
        for idx in range(self.num_joints):
            # coords
            joints[idx, 0] = labels_inf[idx][0]
            joints[idx, 1] = labels_inf[idx][1]
            joints[idx, 2] = 0
            # visible
            joints_vis[idx, 0] = labels_inf[idx][2]
            joints_vis[idx, 1] = labels_inf[idx][2]
            joints_vis[idx, 2] = 0
        if self.opt.mode == 'Train':
            if self.flip and random.random() <= 0.5:
                # image flip
                input = input[:, ::-1, :]
                # coords flip
                joints, joints_vis = self._fliplr_joints(
                    joints, joints_vis, width)
        # scale the raw coords for input size
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = self._scale_transform(joints[i, 0:2], width,
                                                       height)

        # print(joints)
        # transform
        input = self.transformer(Image.fromarray(input))

        # generate gaussian map
        target, target_weight = self.generate_target(joints, joints_vis)
        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)
        meta = {
            'image_file': image_file,
            'joints': joints,
            'joints_vis': joints_vis,
            'width': width,
            'height': height,
        }
        return input, target, target_weight, meta

    def __len__(self):
        return self.data_size


class PoseDataLoader():
    def __init__(self, opt):
        self.opt = opt

        # load dataset
        if opt.mode == "Train":
            logging.info("Load Train Dataset...")
            self.train_set = BaseDataset(self.opt, "Train")
            logging.info("Load Validate Dataset...")
            self.val_set = BaseDataset(self.opt, "Val")
        else:
            logging.info("Load Test Dataset...")
            self.test_set = BaseDataset(self.opt, "Test")

    def GetTrainSet(self):
        if self.opt.mode == "Train":
            return self._DataLoader(self.train_set, shuffle=True)
        else:
            raise ("Train Set DataLoader NOT implemented in Test Mode")

    def GetValSet(self):
        if self.opt.mode == "Train":
            return self._DataLoader(self.val_set, shuffle=False)
        else:
            raise ("Validation Set DataLoader NOT implemented in Test Mode")

    def GetTestSet(self):
        if self.opt.mode == "Test":
            return self._DataLoader(self.test_set)
        else:
            raise ("Test Set DataLoader NOT implemented in Train Mode")

    def _DataLoader(self, dataset, shuffle=False):
        dataloader = DataLoader(dataset,
                                batch_size=self.opt.batch_size,
                                shuffle=shuffle,
                                num_workers=self.opt.load_thread,
                                pin_memory=False,
                                drop_last=False)
        return dataloader
