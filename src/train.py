from util.header import *
from util.evaluate import *
from util.util import *
from loss.mse import JointsMSELoss
from data.loader import PoseDataLoader
from models.model import load_model, save_model
from options.options import Options
from util import util

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

    logging.info('acc of pck = %.4f' % (acc.avg))


def train(model, criterion, train_set, val_set, optimizer, scheduler, opt):
    logging.info("####################Train Model###################")
    # loss_avg =
    for epoch in range(opt.sum_epoch):
        epoch_start_t = time.time()
        epoch_batch_iter = 0
        logging.info('Begin of epoch %d' % (epoch))
        for i, data in enumerate(train_set):
            inputs, targets, targets_weight, meta = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()
                targets_weight = targets_weight.cuda()
            # cal output of CNN
            outputs = model(inputs)
            # cal loss
            loss = criterion(outputs, targets, targets_weight)
            # cal gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_batch_iter += 1

            # display train loss
            if epoch_batch_iter % opt.display_train_freq == 0:
                util.print_loss(loss, epoch, epoch_batch_iter, opt)

        # display validate accuracy
        if epoch_batch_iter % opt.display_validate_freq == 0:
            logging.info('Validate of epoch %d' % (epoch))
            test(model, val_set, opt)

        # adjust learning rate
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        logging.info('learning rate = %.7f epoch = %d' % (lr, epoch))

        # save model
        if epoch % opt.save_epoch_freq == 0 or epoch == opt.sum_epoch - 1:
            logging.info('saving the model at the end of epoch %d' % (epoch))
            save_model(model, opt, epoch)
    logging.info("--------Optimization Done--------")


def main():
    # parse options
    op = Options()
    opt = op.parse()

    # save log to disk
    if opt.mode == "Train":
        log_path = opt.out_dir + "/train.log"

    # save options to disk
    util.opt2file(opt, opt.out_dir + "/opt.txt")

    # log setting
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    fh = logging.FileHandler(log_path, 'a')
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logging.getLogger().addHandler(fh)
    logging.getLogger().addHandler(ch)
    log_level = logging.INFO
    logging.getLogger().setLevel(log_level)

    # load train or test data
    data_loader = PoseDataLoader(opt)
    if opt.mode == "Train":
        train_set = data_loader.GetTrainSet()
        val_set = data_loader.GetValSet()

    # load model
    model = load_model(opt)

    # define loss function
    criterion = JointsMSELoss(opt)

    # define optimizer
    if opt.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                            opt.lr,
                            betas=(0.9, 0.999),
                            eps=1e-08,
                            weight_decay=opt.weight_decay,
                            amsgrad=False)
    else:
        optimizer = optim.SGD(model.parameters(),
                            opt.lr,
                            momentum=opt.momentum,
                            weight_decay=opt.weight_decay)
    # define laerning rate scheluer
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=opt.lr_decay_in_epoch,
                                          gamma=opt.gamma)

    # use cuda
    if len(opt.device_ids) == 1:
        model = model.cuda(opt.device_ids[0])
        cudnn.benchmark = True
    elif len(opt.device_ids) > 1:
        model = nn.DataParallel(model.cuda(opt.device_ids[0]),
                                device_ids=opt.device_ids)
        cudnn.benchmark = True

    # Train model
    if opt.mode == "Train":
        train(model, criterion, train_set, val_set, optimizer, scheduler, opt)


if __name__ == "__main__":
    main()
