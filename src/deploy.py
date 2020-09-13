from util.header import *
from util.evaluate import *
from util.util import *
from options.options import Options
from models.model import load_model
from data.loader import PoseDataLoader
# import util
import pdb


def test(model, test_set, opt):
    acc = AverageMeter()
    for i, data in enumerate(test_set):
        inputs, targets, targets_weight, meta = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()
            targets_weight = targets_weight.cuda()

        outputs = model(inputs)
        # cal accuracy
        _, avg_acc, cnt, pred = accuracy_pck(outputs.detach().cpu().numpy(),
                                             targets.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # get pred coords of raw img
        width = meta['width']
        height = meta['height']
        image_file = meta['image_file']
        preds, maxvals = get_final_preds(opt, outputs.detach().clone().cpu().numpy(), width, height)
        draw_batch_image_with_joints(preds,targets_weight, image_file)
        # pdb.set_trace()
    logging.info('acc of pck = %.4f' % (acc.avg))

def main():
    # parse options
    op = Options()
    opt = op.parse()

    # save log to disk
    if opt.mode == "Test":
        log_path = opt.out_dir + "/test.log"

    # log setting
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    fh = logging.FileHandler(log_path, 'a')
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logging.getLogger().addHandler(fh)
    logging.getLogger().addHandler(ch)
    logging.getLogger().setLevel(logging.INFO)

    # load train or test data
    data_loader = PoseDataLoader(opt)
    test_set = data_loader.GetTestSet()

    # load model
    model = load_model(opt)
    model.eval()

    # use cuda
    if torch.cuda.is_available():
        model = model.cuda(opt.device_ids[0])
        cudnn.benchmark = True

    # Test model
    if opt.mode == "Test":
        test(model, test_set, opt)


if __name__ == "__main__":
    main()
