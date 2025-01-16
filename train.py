import time
from core.utils import AverageMeter, process_data_item, run_model, calculate_accuracy

def train_epoch(e, data_loader, model, criterion, optimizer, opt, class_names, writer):
    k, epoch = e
    print("# ---------------------------------------------------------------------- #")
    print('Training at epoch {}'.format(epoch))
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    end_time = time.time()

    for i, data_item in enumerate(data_loader):
        visual, target, audio, text, batch_size = process_data_item(opt, data_item)
        data_time.update(time.time() - end_time)
        output, loss = run_model(opt, [visual, target, audio, text], model, criterion, i, print_attention=False)
        acc,_ = calculate_accuracy(output, target)
        losses.update(loss.item(), batch_size)
        accuracies.update(acc, batch_size)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

    # ---------------------------------------------------------------------- #
    print("Epoch Time: {:.2f} min".format(batch_time.sum/ 60))
    print("Total Time: {:.2f} h".format(batch_time.sum * opt.n_epochs*5/ (3600*4)))
    print("Train loss: {:.4f}".format(losses.avg))
    print("Train acc: {:.4f}".format(accuracies.avg))

    writer.add_scalar(f'train/loss/{k}_fold', losses.avg, epoch)
    writer.add_scalar(f'train/acc/{k}_fold', accuracies.avg, epoch)